"""Page 2 — Dataset. Loads data.yaml, validates the structure, preview, auto-split."""
from __future__ import annotations
from pathlib import Path
import random
import shutil

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QFrame, QSpinBox, QComboBox, QProgressDialog,
    QListWidget, QListWidgetItem,
)

from ._base import TrainerPage
from polyx.gui import theme as T
from polyx.gui import iconos
from polyx.model.paths import IMAGE_EXTS
from polyx.model.dataset_audit import (
    audit_dataset, LEVEL_OK, LEVEL_WARN, LEVEL_ERROR,
)
from polyx.model.i18n import tr
from ..state import validate_training_config


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, d: dict):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)


def _resolve_split_path(yaml_dir: Path, data: dict, split_value) -> Path | None:
    """Resolves the path of a split applying Ultralytics' logic.

    YOLO convention:
        path: ../datasets/coco128     <- root (optional)
        train: images/train2017       <- relative to 'path' if it exists, else to yaml_dir
        val:   images/val2017
    """
    if not split_value:
        return None
    p = Path(split_value)
    if p.is_absolute():
        return p
    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate
        # If it does not exist relative to `path`, try relative to the yaml
    return (yaml_dir / p).resolve()


def _count_split(yaml_dir: Path, data: dict, split_value) -> int:
    p = _resolve_split_path(yaml_dir, data, split_value)
    if p is None:
        return 0
    if p.is_dir():
        n = 0
        for ext in IMAGE_EXTS:
            n += sum(1 for _ in p.rglob(f"*{ext}"))
        return n
    if p.is_file():
        # list .txt file with image paths
        try:
            return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:
            return 0
    return 0


def _find_label_for(img_path: Path) -> Path | None:
    """Finds the YOLO .txt of an image, following the Ultralytics convention:
    replaces the 'images' segment with 'labels' in the path."""
    # 1) Replace 'images' -> 'labels' in the path
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            new_parts = parts[:i] + ["labels"] + parts[i+1:]
            candidate = Path(*new_parts).with_suffix(".txt")
            if candidate.exists():
                return candidate
            break
    # 2) Same directory
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same
    # 3) labels/ sibling of the parent
    sibling = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if sibling.exists():
        return sibling
    # 4) labels/ inside the same directory
    nested = img_path.parent / "labels" / (img_path.stem + ".txt")
    if nested.exists():
        return nested
    return None


class _PreviewTile(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(220, 150)
        self.setStyleSheet(
            f"background: {T.BG_SOFT}; border: 1px solid {T.RULE}; border-radius: 6px;"
        )
        self.setAlignment(Qt.AlignCenter)
        self.setText(tr("(vacío)"))


class DatasetPage(TrainerPage):
    PAGE_ICON = "dataset"
    PAGE_TITLE = tr("Dataset")
    PAGE_DESCRIPTION = (
        tr("Carga tu data.yaml. Validamos la estructura, contamos imágenes y mostramos una "
        "vista previa con cajas. Si tu dataset no está dividido, usa Auto-split.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Source ──
        c1, l1 = self.card(tr("data.yaml"), "reporte")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.ed_yaml = QLineEdit()
        self.ed_yaml.setPlaceholderText(tr("Ruta a data.yaml…"))
        self.ed_yaml.editingFinished.connect(self._reload_yaml)
        row.addWidget(self.ed_yaml, 1)
        btn = QPushButton("…")
        btn.setFixedWidth(36)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        l1.addLayout(row)

        self.lbl_yaml_status = QLabel(tr("Sin data.yaml cargado."))
        self.lbl_yaml_status.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(self.lbl_yaml_status)
        self.body.addWidget(c1)

        # ── Counts ──
        c2, l2 = self.card(tr("Conteos por split"), "resultados")
        grid = QGridLayout(); grid.setHorizontalSpacing(20); grid.setVerticalSpacing(8)
        self.lbl_train = QLabel(tr("Train: —")); grid.addWidget(self.lbl_train, 0, 0)
        self.lbl_val   = QLabel(tr("Val: —"));   grid.addWidget(self.lbl_val,   0, 1)
        self.lbl_test  = QLabel(tr("Test: —"));  grid.addWidget(self.lbl_test,  0, 2)
        self.lbl_classes = QLabel(tr("Clases: —")); grid.addWidget(self.lbl_classes, 1, 0, 1, 3)
        for w in (self.lbl_train, self.lbl_val, self.lbl_test, self.lbl_classes):
            w.setStyleSheet(f"color: {T.INK2}; font-size: 11pt; border: none;")
        l2.addLayout(grid)
        self.body.addWidget(c2)

        # ── Preview ──
        c3, l3 = self.card(tr("Vista previa (6 imágenes random con cajas)"), "ver")
        prev_grid = QGridLayout(); prev_grid.setSpacing(10)
        self.tiles: list[_PreviewTile] = []
        for i in range(6):
            t = _PreviewTile()
            self.tiles.append(t)
            prev_grid.addWidget(t, i // 3, i % 3)
        l3.addLayout(prev_grid)
        btn_refresh = QPushButton(tr("Otras 6"))
        btn_refresh.setIcon(iconos.icon("recargar", 15, T.INK2))
        btn_refresh.clicked.connect(self._refresh_preview)
        l3.addWidget(btn_refresh, 0, Qt.AlignLeft)
        self.body.addWidget(c3)

        # ── Auto-split ──
        c4, l4 = self.card(tr("Auto-split (si tu dataset no está dividido)"), "piezas")
        info = QLabel(
            tr("Toma una carpeta con images/ y labels/ y genera train/val/test automáticamente "
            "con un nuevo data.yaml. No mueve archivos: usa listas .txt.")
        )
        info.setWordWrap(True); info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l4.addWidget(info)

        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.ed_split_src = QLineEdit()
        self.ed_split_src.setPlaceholderText(tr("Carpeta raíz con images/ y labels/…"))
        srow.addWidget(self.ed_split_src, 1)
        bbtn = QPushButton("…"); bbtn.setFixedWidth(36)
        bbtn.clicked.connect(self._browse_split_src); srow.addWidget(bbtn)
        l4.addLayout(srow)

        prow = QHBoxLayout(); prow.setSpacing(12)
        prow.addWidget(QLabel(tr("Train %:")))
        self.sb_train = QSpinBox(); self.sb_train.setRange(50, 95); self.sb_train.setValue(80)
        prow.addWidget(self.sb_train)
        prow.addWidget(QLabel(tr("Val %:")))
        self.sb_val = QSpinBox(); self.sb_val.setRange(5, 30); self.sb_val.setValue(15)
        prow.addWidget(self.sb_val)
        prow.addWidget(QLabel(tr("Test %:")))
        self.sb_test = QSpinBox(); self.sb_test.setRange(0, 25); self.sb_test.setValue(5)
        prow.addWidget(self.sb_test)
        prow.addStretch(1)
        btn_split = QPushButton(tr("Auto-split + generar data.yaml"))
        btn_split.setIcon(iconos.icon("piezas", 15, T.ON_ACCENT))
        btn_split.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        btn_split.setCursor(Qt.PointingHandCursor)
        btn_split.clicked.connect(self._auto_split)
        prow.addWidget(btn_split)
        l4.addLayout(prow)
        self.body.addWidget(c4)

        # ── Dataset validation ──
        c5, l5 = self.card(tr("Validación del dataset"), "buscar")
        info_val = QLabel(
            tr("Verifica que el dataset sea correcto antes de entrenar. "
            "Se comprueba automáticamente al cargar data.yaml.")
        )
        info_val.setWordWrap(True)
        info_val.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l5.addWidget(info_val)

        # Validation labels
        self._val_checks: list[QLabel] = []
        val_checks_names = [
            "data.yaml cargado y válido",
            tr("Split train con imágenes"),
            tr("Split val con imágenes"),
            "Labels encontradas (≥ 80 %)",
            "Clases definidas en YAML",
            tr("Parámetros de entrenamiento válidos"),
        ]
        val_grid = QGridLayout(); val_grid.setSpacing(6)
        for i, name in enumerate(val_checks_names):
            icon = QLabel("○")
            icon.setFixedWidth(20)
            icon.setStyleSheet(f"color: {T.INK3}; font-size: 11pt; border: none;")
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
            val_grid.addWidget(icon, i, 0)
            val_grid.addWidget(lbl, i, 1)
            self._val_checks.append((icon, lbl))
        l5.addLayout(val_grid)

        btn_val = QPushButton(tr("Validar ahora"))
        btn_val.setIcon(iconos.icon("buscar", 15, T.INK2))
        btn_val.clicked.connect(self._validate_dataset)
        l5.addWidget(btn_val, 0, Qt.AlignLeft)

        # Deep analysis panel (class distribution, etc.)
        self.lbl_audit_title = QLabel(tr("Análisis profundo"))
        self.lbl_audit_title.setStyleSheet(
            f"color: {T.INK2}; font-weight: 600; font-size: 10.5pt; "
            f"border: none; margin-top: 8px;")
        l5.addWidget(self.lbl_audit_title)
        self.lbl_dist = QLabel("")
        self.lbl_dist.setWordWrap(True)
        self.lbl_dist.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        l5.addWidget(self.lbl_dist)
        self.audit_list = QListWidget()
        self.audit_list.setMinimumHeight(150)
        self.audit_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {T.RULE}; border-radius: 6px; "
            f"background: {T.BG_SOFT}; }}")
        self.audit_list.setWordWrap(True)
        l5.addWidget(self.audit_list)
        self.body.addWidget(c5)

        # ── Reactivity (M003/T03) ────────────────────────────────
        # Parameter, model or augmentation changes revalidate the
        # configuration row on the fly (cheap mode: it does not touch the disk). A
        # dataset change (new data.yaml) triggers the full audit.
        self.state.params_changed.connect(self._on_config_changed)
        self.state.aug_changed.connect(self._on_config_changed)
        self.state.model_changed.connect(self._on_config_changed)
        self.state.dataset_changed.connect(self._on_dataset_changed)

    # ──────────────────────────────────────────
    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar data.yaml", "", "YAML (*.yaml *.yml)"
        )
        if f:
            self.ed_yaml.setText(f)
            self._reload_yaml()

    def _browse_split_src(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta raíz con images/ y labels/")
        if d:
            self.ed_split_src.setText(d)

    def _reload_yaml(self):
        t = self.ed_yaml.text().strip()
        if not t:
            return
        p = Path(t)
        if not p.exists():
            self.lbl_yaml_status.setText(tr("✗ Archivo no encontrado."))
            self.lbl_yaml_status.setStyleSheet(f"color: {T.ERR_TX}; font-size: 10pt; border: none;")
            return
        try:
            d = _load_yaml(p)
        except Exception as e:
            self.lbl_yaml_status.setText(f"✗ Error parseando YAML: {e}")
            self.lbl_yaml_status.setStyleSheet(f"color: {T.ERR_TX}; font-size: 10pt; border: none;")
            return

        yaml_dir = p.parent
        train_p = d.get("train"); val_p = d.get("val"); test_p = d.get("test")
        n_train = _count_split(yaml_dir, d, train_p)
        n_val = _count_split(yaml_dir, d, val_p)
        n_test = _count_split(yaml_dir, d, test_p)

        names = d.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]

        self.state.dataset.yaml_path = p
        self.state.dataset.train_count = n_train
        self.state.dataset.val_count = n_val
        self.state.dataset.test_count = n_test
        self.state.dataset.class_names = list(names) if names else []
        self.state.dataset_changed.emit()

        self.lbl_yaml_status.setText(f"✓ {p}")
        self.lbl_yaml_status.setStyleSheet(f"color: {T.OK_TX}; font-size: 10pt; border: none;")
        self.lbl_train.setText(f"Train: {n_train}")
        self.lbl_val.setText(f"Val: {n_val}")
        self.lbl_test.setText(f"Test: {n_test}")
        self.lbl_classes.setText(
            f"Clases ({len(self.state.dataset.class_names)}): "
            + ", ".join(self.state.dataset.class_names[:20])
        )
        self._refresh_preview()

    def _refresh_preview(self):
        yaml_path = self.state.dataset.yaml_path
        if not yaml_path:
            return
        try:
            d = _load_yaml(yaml_path)
        except Exception:
            return
        yaml_dir = yaml_path.parent

        # root to resolve relative paths inside .txt lists
        root_value = d.get("path")
        if root_value:
            root = Path(root_value)
            if not root.is_absolute():
                root = (yaml_dir / root).resolve()
        else:
            root = yaml_dir

        train_p = d.get("train")
        if not train_p:
            return
        train_abs = _resolve_split_path(yaml_dir, d, train_p)
        if train_abs is None or not train_abs.exists():
            # Show a message in the tiles
            for tile in self.tiles:
                tile.setPixmap(QPixmap())
                tile.setText(tr("(no se encontró\nla carpeta de train)"))
            return

        imgs: list[Path] = []
        if train_abs.is_dir():
            for ext in IMAGE_EXTS:
                imgs.extend(list(train_abs.rglob(f"*{ext}"))[:200])
        elif train_abs.is_file():
            for line in train_abs.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pp = Path(line)
                if not pp.is_absolute():
                    # Ultralytics resolves the lists relative to `path` or the yaml
                    pp = (root / pp).resolve()
                    if not pp.exists():
                        pp = (yaml_dir / line).resolve()
                if pp.exists():
                    imgs.append(pp)
        if not imgs:
            for tile in self.tiles:
                tile.setPixmap(QPixmap())
                tile.setText(tr("(sin imágenes\nen train)"))
            return

        random.shuffle(imgs)
        sample = imgs[:6]
        for tile, img_path in zip(self.tiles, sample):
            self._draw_preview(tile, img_path)
        for tile in self.tiles[len(sample):]:
            tile.setPixmap(QPixmap()); tile.setText(tr("(vacío)"))

    def _draw_preview(self, tile: _PreviewTile, img_path: Path):
        pm = QPixmap(str(img_path))
        if pm.isNull():
            tile.setText("✗")
            return
        # Look for YOLO .txt files with the Ultralytics convention
        boxes = []
        label_path = _find_label_for(img_path)
        if label_path is not None:
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(float(parts[0]))
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        boxes.append((cid, cx, cy, w, h))
            except Exception:
                pass
        # draw
        img = QImage(pm)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        names = self.state.dataset.class_names
        for cid, cx, cy, w, h in boxes:
            name = names[cid] if cid < len(names) else str(cid)
            color = QColor(T.CLASS_COLOR_HEX.get(name, "#33aaff"))
            painter.setPen(QPen(color, max(2, img.width() // 300)))
            x = (cx - w / 2) * img.width(); y = (cy - h / 2) * img.height()
            painter.drawRect(int(x), int(y), int(w * img.width()), int(h * img.height()))
        painter.end()
        scaled = QPixmap.fromImage(img).scaled(tile.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        tile.setPixmap(scaled)

    # ── Validation ────────────────────────────────────────────
    def _set_check(self, idx: int, ok: bool | None, msg: str = ""):
        icon_lbl, text_lbl = self._val_checks[idx]
        if ok is True:
            icon_lbl.setText("✓")
            icon_lbl.setStyleSheet(f"color: {T.OK_TX}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.OK_TX}; font-size: 10pt; border: none;")
        elif ok is False:
            icon_lbl.setText("✗")
            icon_lbl.setStyleSheet(f"color: {T.ERR_TX}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.ERR_TX}; font-size: 10pt; border: none;")
        else:
            icon_lbl.setText("○")
            icon_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        if msg:
            text_lbl.setToolTip(msg)

    def _validate_dataset(self):
        """Validates the dataset and updates the validation indicators."""
        # Reset
        for i in range(len(self._val_checks)):
            self._set_check(i, None)

        yaml_path = self.state.dataset.yaml_path
        if not yaml_path or not yaml_path.exists():
            self._set_check(0, False, "No hay data.yaml cargado.")
            self._validate_config_fast()
            return
        try:
            d = _load_yaml(yaml_path)
        except Exception as e:
            self._set_check(0, False, str(e))
            self._validate_config_fast()
            return
        self._set_check(0, True)

        yaml_dir = yaml_path.parent
        n_train = self.state.dataset.train_count
        n_val   = self.state.dataset.val_count
        self._set_check(1, n_train > 0,
                        f"{n_train} imagen(es) en train." if n_train else "Sin imágenes en train.")
        self._set_check(2, n_val > 0,
                        f"{n_val} imagen(es) en val." if n_val else "Sin imágenes en val.")

        # Check labels for the first 50 train images
        train_p = d.get("train")
        train_abs = _resolve_split_path(yaml_dir, d, train_p)
        if train_abs and train_abs.exists():
            imgs: list[Path] = []
            if train_abs.is_dir():
                for ext in IMAGE_EXTS:
                    imgs.extend(list(train_abs.rglob(f"*{ext}"))[:50])
            if imgs:
                with_label = sum(1 for img in imgs if _find_label_for(img))
                pct = with_label / len(imgs) * 100
                ok_labels = pct >= 80
                self._set_check(3, ok_labels,
                                f"{pct:.0f} % de imágenes tienen label ({with_label}/{len(imgs)} revisadas).")
            else:
                self._set_check(3, None, "No se encontraron imágenes para revisar.")
        else:
            self._set_check(3, None, "No se pudo acceder al split train.")

        names = d.get("names", [])
        if isinstance(names, dict):
            names = list(names.values())
        has_names = bool(names)
        self._set_check(4, has_names,
                        f"Clases: {', '.join(str(n) for n in names[:10])}" if has_names
                        else "No hay clave 'names' en el YAML.")

        # Deep analysis (class distribution, per-class validation, etc.)
        self._run_deep_audit(yaml_path)
        self._validate_config_fast()

    # ── Reactive validation ─────────────────────────────────────
    def _on_config_changed(self):
        """Cheap re-validation when parameters, model or augmentation change.

        Runs on the UI thread but is O(1): it does not touch the disk. The
        full dataset audit runs in ``_on_dataset_changed``.
        """
        self._validate_config_fast()

    def _on_dataset_changed(self):
        """Full re-validation when the dataset changes (new data.yaml)."""
        self._validate_dataset()

    def _validate_config_fast(self):
        """Row 6: model and parameters within range, without auditing the disk."""
        ok, msgs = validate_training_config(self.state, con_dataset=False)
        if ok:
            self._set_check(5, True,
                            tr("Modelo, hiperparámetros y augmentación dentro de rango."))
        else:
            self._set_check(5, False, "\n".join(msgs))

    def _run_deep_audit(self, yaml_path):
        """Runs the deep audit and dumps the findings into the panel."""
        self.audit_list.clear()
        try:
            audit = audit_dataset(yaml_path)
        except Exception as e:
            self.lbl_dist.setText("")
            it = QListWidgetItem(f"No se pudo analizar: {e}")
            it.setForeground(QColor(T.ERR))
            self.audit_list.addItem(it)
            return

        # Compact per-class distribution table (train vs val)
        tr = audit.per_class.get("train", {})
        vl = audit.per_class.get("val", {})
        if audit.class_names:
            parts_2 = []
            for c in audit.class_names:
                t = tr.get(c, 0)
                v = vl.get(c, 0)
                marker = "  ⚠️sin val" if (t > 0 and v == 0) else ""
                parts_2.append(f"<b>{c}</b>: train {t} / val {v}{marker}")
            self.lbl_dist.setText("Distribución por clase &nbsp; — &nbsp; "
                                  + " &nbsp;•&nbsp; ".join(parts_2))
        else:
            self.lbl_dist.setText("")

        colors = {LEVEL_ERROR: T.ERR, LEVEL_WARN: T.WARN, LEVEL_OK: T.OK}
        icons = {LEVEL_ERROR: "✗", LEVEL_WARN: "⚠", LEVEL_OK: "✓"}
        for f in audit.findings:
            txt = f"{icons.get(f.level,'•')}  {f.title}"
            if f.detail:
                txt += f"\n     {f.detail}"
            it = QListWidgetItem(txt)
            it.setForeground(QColor(colors.get(f.level, T.INK2)))
            self.audit_list.addItem(it)

    # ── Auto-split ────────────────────────────────────────────
    def _auto_split(self):
        src = self.ed_split_src.text().strip()
        if not src:
            QMessageBox.warning(self, tr("Auto-split"), tr("Selecciona la carpeta raíz primero."))
            return
        root = Path(src)
        imgs_dir = root / "images"
        if not imgs_dir.exists():
            QMessageBox.warning(self, tr("Auto-split"),
                                f"No existe {imgs_dir}. Necesitas images/ y labels/ dentro de la carpeta.")
            return
        # collect images
        all_imgs: list[Path] = []
        for ext in IMAGE_EXTS:
            all_imgs.extend(imgs_dir.rglob(f"*{ext}"))
        if not all_imgs:
            QMessageBox.warning(self, tr("Auto-split"), tr("No se encontraron imágenes en images/."))
            return

        tp, vp, sp = self.sb_train.value(), self.sb_val.value(), self.sb_test.value()
        if tp + vp + sp > 100:
            QMessageBox.warning(self, tr("Auto-split"), tr("Los porcentajes suman > 100."))
            return

        random.seed(42)
        random.shuffle(all_imgs)
        N = len(all_imgs)
        n_tr = int(N * tp / 100)
        n_vl = int(N * vp / 100)
        train_list = all_imgs[:n_tr]
        val_list   = all_imgs[n_tr:n_tr + n_vl]
        test_list  = all_imgs[n_tr + n_vl:] if sp > 0 else []

        # Save lists
        for fname, lst in [("train.txt", train_list), ("val.txt", val_list), ("test.txt", test_list)]:
            if not lst: continue
            (root / fname).write_text(
                "\n".join(str(p.relative_to(root)) for p in lst), encoding="utf-8"
            )

        # Read classes.txt if it exists
        cls_path = root / "classes.txt"
        names: list[str] = []
        if cls_path.exists():
            names = [l.strip() for l in cls_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        d = {
            "path": str(root),
            "train": "train.txt",
            "val": "val.txt",
        }
        if test_list: d["test"] = "test.txt"
        if names:     d["names"] = names

        yaml_out = root / "data.yaml"
        _save_yaml(yaml_out, d)
        QMessageBox.information(
            self, tr("Auto-split"),
            f"Listo:\n• Train: {len(train_list)}\n• Val: {len(val_list)}\n"
            f"• Test: {len(test_list)}\n\nGenerado: {yaml_out}"
        )
        self.ed_yaml.setText(str(yaml_out))
        self._reload_yaml()
