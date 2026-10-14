"""Hilo de entrenamiento en background con Ultralytics YOLO."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QThread, Signal

from ..core import observabilidad as obs

from .state import TrainerState, EpochMetrics
from .dominio import elegir_mejor_en_real


class TrainerRunner(QThread):
    """Lanza un entrenamiento Ultralytics en thread aparte y emite señales por época."""

    epoch_metrics = Signal(object)         # EpochMetrics
    log_line = Signal(str)
    progress = Signal(int, int)             # epoch, total
    finished_ok = Signal(str)               # ruta best.pt
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: TrainerState, parent=None):
        super().__init__(parent)
        self.state = state

    def _informar_comparacion(self, comparativa: list[dict], p) -> None:
        """Resume en el log las arquitecturas entrenadas con la misma config.

        Si hubo reeleccion por dominio, la comparacion se hace con el F1 sobre
        sedimento real, que es con lo que se va a detectar de verdad, y no con el
        mAP global que mezcla laboratorio y terreno.
        """
        hay_real = any(c.get("f1_real") is not None for c in comparativa)
        self.log_line.emit("")
        self.log_line.emit("=" * 60)
        self.log_line.emit("  COMPARACION DE ARQUITECTURAS")
        self.log_line.emit("=" * 60)
        self.log_line.emit(
            f"  Configuracion identica: imgsz={p.imgsz} · batch={p.batch} · "
            f"epocas={p.epochs} · lr0={p.lr0}")
        self.log_line.emit("")
        cab = f"  {'familia':<10}{'peso base':<16}{'mAP50':>9}{'epoca':>8}"
        if hay_real:
            cab += f"{'F1 real':>10}"
        self.log_line.emit(cab)
        self.log_line.emit("  " + "-" * (len(cab) - 2))
        for c in comparativa:
            fila = (f"  {c['familia']:<10}{c['peso_base']:<16}"
                    f"{c['map50']:>9.4f}{c['epoca_mejor']:>8}")
            if hay_real:
                fr = c.get("f1_real")
                fila += f"{fr:>10.4f}" if fr is not None else f"{'-':>10}"
            self.log_line.emit(fila)

        # Con reeleccion por dominio se compara con el F1 en sedimento real: es
        # el numero que describe lo que van a hacer estos pesos en las fotos de
        # terreno. El mAP global mezcla laboratorio y terreno y no sirve aqui.
        clave = ("f1_real" if hay_real else "map50")
        etiqueta = ("F1 en sedimento real" if hay_real else "mAP50")
        utiles = [c for c in comparativa if c.get(clave) is not None]
        if not utiles:
            utiles, clave, etiqueta = comparativa, "map50", "mAP50"
        mejor = max(utiles, key=lambda c: c[clave])
        peor = min(utiles, key=lambda c: c[clave])
        d = mejor[clave] - peor[clave]
        self.log_line.emit("")
        self.log_line.emit(
            f"  Mejor: {mejor['familia']} ({etiqueta} {mejor[clave]:.4f}, "
            f"+{d:.4f} sobre {peor['familia']})")
        if hay_real:
            self.log_line.emit(
                "  Los pesos a usar para detectar son los best_real.pt.")
        # Sin repeticiones no se puede separar arquitectura de azar de semilla.
        self.log_line.emit(
            "  Con un entrenamiento por arquitectura, una diferencia pequena")
        self.log_line.emit(
            "  no distingue el diseno del azar de inicializacion.")
        self.log_line.emit(
            "  Los pesos quedaron en runs_train/; comparalos en la pestana Comparar.")
        self.log_line.emit("=" * 60)

    def run(self):
        _t0 = __import__("time").perf_counter()
        try:
            import torch
            from ultralytics import YOLO
        except Exception as e:
            self.failed.emit(f"No se pudo importar Ultralytics / PyTorch: {e}")
            return

        try:
            st = self.state
            mdl = st.model
            ds = st.dataset
            p = st.params
            a = st.aug

            if not ds.yaml_path or not Path(ds.yaml_path).exists():
                self.failed.emit("Falta data.yaml. Cárgalo en la pestaña Dataset.")
                return
            log = self._log()
            obs.set_run_id(mdl.base_weights_name())
            obs.evento(log, "train.config", modelo=mdl.nombre, dataset=str(ds.yaml_path),
                       epochs=p.epochs, batch=p.batch, imgsz=p.imgsz,
                       device=getattr(p, "device", None),
                       optimizador=getattr(p, "optimizer", None))

            # ── Diagnóstico explícito de hardware ──
            cuda_ok = torch.cuda.is_available()
            self.log_line.emit("=" * 60)
            self.log_line.emit(f"[HW] PyTorch {torch.__version__}  ·  CUDA build: {torch.version.cuda}")
            self.log_line.emit(f"[HW] torch.cuda.is_available(): {cuda_ok}")
            if cuda_ok:
                dev_name = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info(0)
                self.log_line.emit(
                    f"[HW] GPU 0: {dev_name}  ·  "
                    f"VRAM libre {free/1024**3:.2f} / {total/1024**3:.2f} GB"
                )
            self.log_line.emit(f"[HW] Device solicitado: '{p.device}'")

            # Resolver device efectivo
            req = (p.device or "0").strip().lower()
            if not cuda_ok and req != "cpu":
                self.log_line.emit("[ATENCION] CUDA no disponible — se forzará entrenamiento en CPU.")
                effective_device = "cpu"
            elif req in ("cpu",):
                self.log_line.emit("[INFO] Entrenando en CPU (por configuración).")
                effective_device = "cpu"
            else:
                # 'cuda:0' es más explícito que '0' y evita ambigüedades
                effective_device = f"cuda:{req}" if req.isdigit() else req
                self.log_line.emit(f"[INFO] Entrenando en GPU: {effective_device}")
            self.log_line.emit("=" * 60)

            # Que familias se entrenan. Con pesos personalizados no tiene
            # sentido comparar arquitecturas: el .pt ya fija una.
            usa_custom = bool(mdl.custom_weights and Path(mdl.custom_weights).exists())
            if usa_custom:
                familias = [mdl.family]
                if mdl.comparar_familias:
                    self.log_line.emit(
                        "[ATENCION] Hay pesos personalizados cargados: se ignora "
                        "la comparacion de arquitecturas y se entrena solo ese .pt.")
            else:
                familias = mdl.familias_a_entrenar()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = p.run_name.strip() or f"train_{stamp}"

            proj_dir = Path(__file__).resolve().parents[2] / "runs_train"
            proj_dir.mkdir(parents=True, exist_ok=True)

            if len(familias) > 1:
                self.log_line.emit("=" * 60)
                self.log_line.emit(
                    f"[INFO] Comparacion de arquitecturas: se entrenaran "
                    f"{len(familias)} modelos ({', '.join(familias)}) con "
                    f"identica configuracion.")
                self.log_line.emit(
                    "[INFO] Van en secuencia, no en paralelo: comparten GPU.")
                self.log_line.emit("=" * 60)

            # Los callbacks se registran una vez pero sirven a varias corridas,
            # asi que leen el nombre desde aqui en vez de capturarlo por cierre.
            run_actual = {"nombre": base_name}

            # Callback: progreso por época
            def _on_train_epoch_end(trainer):
                try:
                    ep = int(trainer.epoch) + 1
                    metrics = trainer.metrics or {}
                    em = EpochMetrics(
                        epoch=ep,
                        precision=float(metrics.get("metrics/precision(B)", 0.0)),
                        recall=float(metrics.get("metrics/recall(B)", 0.0)),
                        map50=float(metrics.get("metrics/mAP50(B)", 0.0)),
                        map50_95=float(metrics.get("metrics/mAP50-95(B)", 0.0)),
                        box_loss=float(getattr(trainer, "loss_items", [0])[0]) if getattr(trainer, "loss_items", None) is not None else 0.0,
                        lr=float(trainer.optimizer.param_groups[0]["lr"]) if trainer.optimizer else 0.0,
                    )
                    st.history.append(em)
                    if em.map50 > st.best_map50:
                        st.best_map50 = em.map50
                        st.best_epoch = ep
                        st.epochs_no_improve = 0
                    else:
                        st.epochs_no_improve += 1
                    self.epoch_metrics.emit(em)
                    self.progress.emit(ep, p.epochs)
                except Exception as e:
                    self.log_line.emit(f"[WARN] callback error: {e}")

            def _on_train_start(trainer):
                self.log_line.emit(f"[INFO] Entrenamiento iniciado.")
                self.log_line.emit(f"[INFO] Run: {run_actual['nombre']}")
                self.log_line.emit(f"[INFO] Imgsz: {p.imgsz} · Batch: {p.batch} · Epochs: {p.epochs}")
                # Confirmar device EFECTIVO del modelo (esto es la verdad)
                try:
                    model_dev = next(trainer.model.parameters()).device
                    self.log_line.emit(f"[HW✓] Modelo Ultralytics colocado en: {model_dev}")
                    if "cuda" not in str(model_dev) and effective_device != "cpu":
                        self.log_line.emit(
                            "[ALERTA] Pediste GPU pero Ultralytics movió el modelo a CPU. "
                            "Causas tipicas: OOM al probar, AMP check fallido, o driver "
                            "incompatible. Revisa la salida arriba."
                        )
                except Exception as e:
                    self.log_line.emit(f"[WARN] No se pudo verificar device del modelo: {e}")

            def _on_train_end(trainer):
                self.log_line.emit(f"[INFO] Entrenamiento finalizado.")

            # Cache: ultralytics acepta True/False/'ram'/'disk'
            cache_val: object = p.cache
            if isinstance(cache_val, str) and cache_val.lower() in ("false", "0", "no"):
                cache_val = False

            train_kwargs = dict(
                data=str(ds.yaml_path),
                imgsz=int(p.imgsz),
                epochs=int(p.epochs),
                batch=int(p.batch),
                lr0=float(p.lr0),
                weight_decay=float(p.weight_decay),
                momentum=float(p.momentum),
                warmup_epochs=int(p.warmup_epochs),
                close_mosaic=int(p.close_mosaic),
                label_smoothing=float(p.label_smoothing),
                patience=int(p.patience),
                save_period=int(p.save_period),
                amp=bool(p.amp),
                cos_lr=bool(p.cos_lr),
                cache=cache_val,
                device=effective_device,
                workers=int(p.workers),
                # Augmentación
                hsv_h=float(a.hsv_h),
                hsv_s=float(a.hsv_s),
                hsv_v=float(a.hsv_v),
                fliplr=float(a.fliplr),
                mosaic=float(a.mosaic),
                mixup=float(a.mixup),
                copy_paste=float(a.copy_paste),
                # Proyecto
                project=str(proj_dir),
                exist_ok=True,
                verbose=True,
                plots=True,
            )

            comparativa = []
            ultimo_best = ""

            for i, familia in enumerate(familias, start=1):
                if usa_custom:
                    weights = str(mdl.custom_weights)
                else:
                    weights = mdl.peso_de(familia)

                # Con una sola familia se respeta el nombre tal cual lo puso el
                # usuario; con varias hay que desambiguar o la segunda corrida
                # sobrescribiria a la primera (exist_ok=True).
                nombre = base_name if len(familias) == 1 else f"{base_name}_{familia}"
                run_actual["nombre"] = nombre
                st.run_dir = proj_dir / nombre

                # El historial es por corrida: si no se limpia, las curvas de la
                # segunda arquitectura se dibujarian encima de las de la primera.
                st.history.clear()
                st.best_map50 = 0.0
                st.best_epoch = 0
                st.epochs_no_improve = 0

                if len(familias) > 1:
                    self.log_line.emit("")
                    self.log_line.emit("=" * 60)
                    self.log_line.emit(
                        f"[INFO] Modelo {i} de {len(familias)}: {familia} ({weights})")
                    self.log_line.emit("=" * 60)

                self.log_line.emit(f"[INFO] Cargando modelo base: {weights}")
                model = YOLO(weights)
                model.add_callback("on_train_start", _on_train_start)
                model.add_callback("on_train_epoch_end", _on_train_epoch_end)
                model.add_callback("on_train_end", _on_train_end)

                self.log_line.emit("[INFO] Iniciando model.train(...)")
                model.train(name=nombre, **train_kwargs)

                best_path = proj_dir / nombre / "weights" / "best.pt"

                # Reelegir el checkpoint contra el dominio que importa. best.pt
                # sale del mAP global, que aqui lo dominan las placas de
                # laboratorio; para detectar en fotos de terreno eso es el
                # criterio equivocado.
                peso_final = best_path
                f1_real = None
                if getattr(p, "elegir_por_dominio_real", True):
                    try:
                        elegido, tabla = elegir_mejor_en_real(
                            proj_dir / nombre, Path(ds.yaml_path),
                            imgsz=int(p.imgsz), batch=int(p.batch),
                            device=effective_device, log=self.log_line.emit)
                        if elegido is not None:
                            peso_final = elegido
                            f1_real = tabla[0]["f1"] if tabla else None
                    except Exception as exc:
                        # Que falle la reeleccion no puede tirar un entrenamiento
                        # de horas: se avisa y se sigue con best.pt.
                        self.log_line.emit(
                            f"[WARN] No se pudo reelegir checkpoint: {exc}. "
                            f"Se conserva best.pt.")

                if peso_final.exists():
                    ultimo_best = str(peso_final)
                comparativa.append({
                    "familia": familia,
                    "peso_base": weights,
                    "run": nombre,
                    "best": str(peso_final) if peso_final.exists() else "",
                    "map50": st.best_map50,
                    "epoca_mejor": st.best_epoch,
                    "f1_real": f1_real,
                })

            if len(comparativa) > 1:
                self._informar_comparacion(comparativa, p)

            obs.evento(log, "train.finish", ms=int((__import__("time").perf_counter() - _t0) * 1000),
                       weights=str(ultimo_best))
            self.finished_ok.emit(ultimo_best)

        except KeyboardInterrupt:
            self.aborted.emit()
            obs.evento(self._log(), "train.aborted")
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            obs.evento_error(self._log(), "train.error", e)

    def _log(self):
        return obs.logger_para("trainer")
