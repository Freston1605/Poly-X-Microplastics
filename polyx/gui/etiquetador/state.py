"""Estado del Etiquetador — imágenes, anotaciones, undo/redo."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


CLASS_NAMES_DEFAULT = ["PET", "PP", "LDPE"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class BBox:
    class_id: int
    cx: float   # normalizado [0, 1]
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> "BBox":
        parts = line.strip().split()
        return cls(int(parts[0]), float(parts[1]), float(parts[2]),
                   float(parts[3]), float(parts[4]))


class LabelerState(QObject):
    images_loaded      = Signal(list)   # List[Path]
    image_changed      = Signal(int)    # current_idx
    annotations_changed = Signal()
    classes_changed    = Signal(list)   # List[str]
    active_class_changed = Signal(int)
    progress_changed   = Signal()       # cambió el conjunto de revisadas

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.current_idx: int = -1
        self.class_names: list[str] = list(CLASS_NAMES_DEFAULT)
        self.active_class: int = 0
        self.root_folder: Optional[Path] = None

        self._annotations: dict[str, list[BBox]] = {}
        self._undo_stacks: dict[str, list[list[BBox]]] = {}
        self._redo_stacks: dict[str, list[list[BBox]]] = {}
        # Imágenes que el operador declaró revisadas. Un conteo censal necesita
        # distinguir "revisada, cero partículas" de "todavía no mirada": ambas
        # tienen cero cajas, pero solo la primera es un dato.
        self._reviewed: set[str] = set()

    # ── Propiedades ───────────────────────────────────────────
    @property
    def current_image(self) -> Optional[Path]:
        if 0 <= self.current_idx < len(self.images):
            return self.images[self.current_idx]
        return None

    @property
    def current_boxes(self) -> list[BBox]:
        img = self.current_image
        return self._annotations.get(str(img), []) if img else []

    # ── Edición de anotaciones ───────────────────────────────
    def set_current_boxes(self, boxes: list[BBox], push_undo: bool = True):
        img = self.current_image
        if img is None:
            return
        key = str(img)
        if push_undo:
            self._push_undo(key)
        self._annotations[key] = list(boxes)
        self.annotations_changed.emit()

    def _push_undo(self, key: str):
        self._undo_stacks.setdefault(key, []).append(
            copy.deepcopy(self._annotations.get(key, []))
        )
        self._redo_stacks.setdefault(key, []).clear()

    def undo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._undo_stacks.get(key, [])
        if stack:
            self._redo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    def redo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._redo_stacks.get(key, [])
        if stack:
            self._undo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    # ── Navegación ───────────────────────────────────────────
    def load_images(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        self.images = imgs
        self.root_folder = folder
        self._annotations.clear()
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        # Recupera el avance de sesiones anteriores: un .txt en disco significa
        # que esa imagen ya fue revisada, tenga o no cajas.
        self._reviewed = {str(p) for p in imgs if self._label_path_for(p).exists()}
        self.images_loaded.emit(imgs)
        if imgs:
            self.current_idx = -1  # forzar goto a emitir image_changed
            self.goto(0)
        else:
            self.current_idx = -1

    def goto(self, idx: int):
        if not self.images:
            return
        idx = max(0, min(idx, len(self.images) - 1))
        if idx == self.current_idx:
            return
        self.save_current()
        self.current_idx = idx
        self._load_labels_for(self.images[idx])
        self.image_changed.emit(idx)
        self.annotations_changed.emit()

    def next_image(self):
        self.goto(self.current_idx + 1)

    def prev_image(self):
        self.goto(self.current_idx - 1)

    # ── Revisión ─────────────────────────────────────────────
    def is_reviewed(self, img: Path) -> bool:
        return str(img) in self._reviewed

    def mark_reviewed(self, revisada: bool = True):
        """Declara la imagen actual como revisada, aunque no tenga partículas."""
        img = self.current_image
        if img is None:
            return
        key = str(img)
        if revisada:
            self._reviewed.add(key)
            self._save_labels_for(img, self.current_boxes, forzar=True)
        else:
            self._reviewed.discard(key)
            txt = self._label_path_for(img)
            if txt.exists() and not self.current_boxes:
                txt.unlink()
        self.progress_changed.emit()

    def n_reviewed(self) -> int:
        return len(self._reviewed)

    def next_unreviewed(self) -> int:
        """Índice de la siguiente imagen sin revisar; -1 si no queda ninguna."""
        n = len(self.images)
        for salto in range(1, n + 1):
            i = (self.current_idx + salto) % n
            if str(self.images[i]) not in self._reviewed:
                return i
        return -1

    # ── Persistencia ─────────────────────────────────────────
    def save_current(self):
        img = self.current_image
        if img:
            self._save_labels_for(img, self.current_boxes)

    def _label_path_for(self, img: Path) -> Path:
        # Si imagen está en images/, guardar en labels/ hermana
        if img.parent.name.lower() == "images":
            lbl_dir = img.parent.parent / "labels"
        else:
            lbl_dir = img.parent
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return lbl_dir / (img.stem + ".txt")

    def _save_labels_for(self, img: Path, boxes: list[BBox], forzar: bool = False):
        """Escribe el .txt. Sin cajas y sin revisar, NO crea archivo.

        Crear .txt vacíos al solo pasar por encima falsearía el avance: una
        imagen apenas ojeada quedaría registrada como revisada con cero
        partículas, que es un dato distinto.
        """
        if not boxes and not forzar and str(img) not in self._reviewed:
            return
        txt = self._label_path_for(img)
        content = "\n".join(b.to_yolo_line() for b in boxes)
        txt.write_text(content + "\n" if content else "", encoding="utf-8")
        if boxes:
            self._reviewed.add(str(img))

    def _load_labels_for(self, img: Path):
        key = str(img)
        if key in self._annotations:
            return
        txt = self._label_path_for(img)
        boxes: list[BBox] = []
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        boxes.append(BBox.from_yolo_line(line))
                    except Exception:
                        pass
        self._annotations[key] = boxes

    def save_classes_txt(self):
        """Escribe classes.txt en la raíz y en cada subcarpeta con imágenes.

        Con carpetas anidadas (un directorio por testigo) no basta la raíz: las
        herramientas YOLO buscan classes.txt junto a las etiquetas.
        """
        if not self.images:
            return
        contenido = "\n".join(self.class_names) + "\n"
        destinos = {self._label_path_for(p).parent for p in self.images}
        if self.root_folder is not None:
            destinos.add(self.root_folder)
        for d in destinos:
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / "classes.txt").write_text(contenido, encoding="utf-8")
            except OSError:
                pass

    # ── Clases ───────────────────────────────────────────────
    def set_active_class(self, cls_id: int):
        self.active_class = cls_id
        self.active_class_changed.emit(cls_id)

    def set_class_names(self, names: list[str]):
        self.class_names = names
        self.classes_changed.emit(names)
