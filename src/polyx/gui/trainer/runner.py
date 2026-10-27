"""Background training thread with Ultralytics YOLO."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from polyx.core import observability as obs

from .dominio import choose_better_in_real
from .state import EpochMetrics, TrainerState


class TrainerRunner(QThread):
    """Launches an Ultralytics training on a separate thread and emits per-epoch signals."""

    epoch_metrics = Signal(object)  # EpochMetrics
    log_line = Signal(str)
    progress = Signal(int, int)  # epoch, total
    finished_ok = Signal(str)  # path to best.pt
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: TrainerState, parent=None):
        super().__init__(parent)
        self.state = state

    def _report_comparison(self, comparison: list[dict], p) -> None:
        """Summarizes in the log the architectures trained with the same config.

        If there was a domain re-choice, the comparison is made with the F1
        on real sediment, which is what will actually be detected, and not
        with the global mAP that mixes laboratory and field.
        """
        there_is_real = any(c.get("f1_real") is not None for c in comparison)
        self.log_line.emit("")
        self.log_line.emit("=" * 60)
        self.log_line.emit("  COMPARACION DE ARQUITECTURAS")
        self.log_line.emit("=" * 60)
        self.log_line.emit(
            f"  Configuracion identica: imgsz={p.imgsz} · batch={p.batch} · "
            f"epocas={p.epochs} · lr0={p.lr0}"
        )
        self.log_line.emit("")
        header_line = f"  {'familia':<10}{'peso base':<16}{'mAP50':>9}{'epoca':>8}"
        if there_is_real:
            header_line += f"{'F1 real':>10}"
        self.log_line.emit(header_line)
        self.log_line.emit("  " + "-" * (len(header_line) - 2))
        for c in comparison:
            row = f"  {c['familia']:<10}{c['peso_base']:<16}{c['map50']:>9.4f}{c['epoca_mejor']:>8}"
            if there_is_real:
                fr = c.get("f1_real")
                row += f"{fr:>10.4f}" if fr is not None else f"{'-':>10}"
            self.log_line.emit(row)

        # With domain re-selection the F1 on real sediment is the benchmark: it is
        # the number that describes what these weights will do on field photos.
        # The global mAP mixes laboratory and field and is useless here.
        key_2 = "f1_real" if there_is_real else "map50"
        label = "F1 en sedimento real" if there_is_real else "mAP50"
        useful = [c for c in comparison if c.get(key_2) is not None]
        if not useful:
            useful, key_2, label = comparison, "map50", "mAP50"
        best = max(useful, key=lambda c: c[key_2])
        worst = min(useful, key=lambda c: c[key_2])
        d = best[key_2] - worst[key_2]
        self.log_line.emit("")
        self.log_line.emit(
            f"  Mejor: {best['familia']} ({label} {best[key_2]:.4f}, "
            f"+{d:.4f} sobre {worst['familia']})"
        )
        if there_is_real:
            self.log_line.emit("  Los pesos a usar para detectar son los best_real.pt.")
        # Without repetitions architecture cannot be separated from random seed.
        self.log_line.emit("  Con un entrenamiento por arquitectura, una diferencia pequena")
        self.log_line.emit("  no distingue el diseno del azar de inicializacion.")
        self.log_line.emit(
            "  Los pesos quedaron en runs_train/; comparalos en la pestana Comparar."
        )
        self.log_line.emit("=" * 60)

    def _diagnose_hardware(self, torch, p) -> str:
        """Emit hardware diagnostics and resolve the effective device string."""
        cuda_ok = torch.cuda.is_available()
        torch_cuda = getattr(torch.version, "cuda", None)
        torch_hip = getattr(torch.version, "hip", None)
        build_str = (
            torch_cuda
            if torch_cuda is not None
            else (torch_hip if torch_hip is not None else "CPU-only")
        )
        backend_name = "ROCm" if (cuda_ok and torch_hip is not None) else "CUDA"
        self.log_line.emit("=" * 60)
        self.log_line.emit(
            f"[HW] PyTorch {torch.__version__}  ·  {backend_name} build: {build_str}"
        )
        self.log_line.emit(f"[HW] torch.cuda.is_available(): {cuda_ok}")
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            self.log_line.emit(
                f"[HW] GPU 0: {dev_name}  ·  "
                f"VRAM libre {free / 1024**3:.2f} / {total / 1024**3:.2f} GB"
            )
        self.log_line.emit(f"[HW] Device solicitado: '{p.device}'")

        req = (p.device or "0").strip().lower()
        if not cuda_ok and req != "cpu":
            self.log_line.emit(
                "[ATENCION] GPU (CUDA/ROCm) no disponible — se forzará entrenamiento en CPU."
            )
            effective_device = "cpu"
        elif req in ("cpu",):  # noqa: FURB171
            self.log_line.emit("[INFO] Entrenando en CPU (por configuración).")
            effective_device = "cpu"
        else:
            effective_device = f"cuda:{req}" if req.isdigit() else req
            self.log_line.emit(f"[INFO] Entrenando en GPU: {effective_device}")
        self.log_line.emit("=" * 60)
        return effective_device

    def run(self):
        _t0 = __import__("time").perf_counter()
        try:
            import torch
            from ultralytics import YOLO  # noqa: F401
        except Exception as e:  # noqa: BLE001
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
            obs.set_run_id(mdl.base_weights_name)
            obs.event(
                log,
                "train.config",
                modelo=mdl.name,
                dataset=str(ds.yaml_path),
                epochs=p.epochs,
                batch=p.batch,
                imgsz=p.imgsz,
                device=getattr(p, "device", None),
                optimizador=getattr(p, "optimizer", None),
            )

            effective_device = self._diagnose_hardware(torch, p)

            # Which families are trained. With custom weights,
            # checking architectures makes little sense: the .pt already fixes one.
            use_custom = bool(mdl.custom_weights and Path(mdl.custom_weights).exists())
            if use_custom:
                families = [mdl.family]
                if mdl.compare_families:
                    self.log_line.emit(
                        "[ATENCION] Hay pesos personalizados cargados: se ignora "
                        "la comparacion de arquitecturas y se entrena solo ese .pt."
                    )
            else:
                families = mdl.families_to_train()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = p.run_name.strip() or f"train_{stamp}"

            proj_dir = Path(__file__).resolve().parents[4] / "runs_train"
            proj_dir.mkdir(parents=True, exist_ok=True)

            if len(families) > 1:
                self.log_line.emit("=" * 60)
                self.log_line.emit(
                    f"[INFO] Comparacion de arquitecturas: se entrenaran "
                    f"{len(families)} modelos ({', '.join(families)}) con "
                    f"identica configuracion."
                )
                self.log_line.emit("[INFO] Van en secuencia, no en paralelo: comparten GPU.")
                self.log_line.emit("=" * 60)

            # Callbacks are registered once but serve several runs,
            # so they read the name from here instead of capturing it in a closure.
            # "aborted" is set from the epoch callback when the user
            # asks to stop: the current family finishes and the remaining ones do not start.
            run_current = {"nombre": base_name, "abortado": False}

            # Callback: per-epoch progress
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
                        box_loss=float(getattr(trainer, "loss_items", [0])[0])
                        if getattr(trainer, "loss_items", None) is not None
                        else 0.0,
                        lr=float(trainer.optimizer.param_groups[0]["lr"])
                        if trainer.optimizer
                        else 0.0,
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
                    # Per-epoch telemetry: one record per epoch so that
                    # trainer.jsonl can follow a run in progress.
                    # obs.evento never raises, it is safe inside the callback.
                    obs.event(
                        log,
                        "train.progress",
                        epoca=ep,
                        total=p.epochs,
                        map50=round(em.map50, 4),
                        precision=round(em.precision, 4),
                        recall=round(em.recall, 4),
                        caja=round(em.box_loss, 4),
                    )
                    # Stop requested by the user: we cut at the epoch
                    # boundary (not midway) so that no half-written checkpoints are left.
                    # consume_abort() consumes the flag only once.
                    if st.consume_abort():
                        trainer.stop = True
                        run_current["abortado"] = True
                        self.log_line.emit(
                            "[INFO] Solicitud de detención recibida: "
                            "se parará al final de esta época."
                        )
                except Exception as e:  # noqa: BLE001
                    self.log_line.emit(f"[WARN] callback error: {e}")

            def _on_train_start(trainer):
                self.log_line.emit("[INFO] Entrenamiento iniciado.")
                self.log_line.emit(f"[INFO] Run: {run_current['nombre']}")
                self.log_line.emit(
                    f"[INFO] Imgsz: {p.imgsz} · Batch: {p.batch} · Epochs: {p.epochs}"
                )
                # Confirm the EFFECTIVE device of the model (this is the truth)
                try:
                    model_dev = next(trainer.model.parameters()).device
                    self.log_line.emit(f"[HW✓] Modelo Ultralytics colocado en: {model_dev}")
                    if "cuda" not in str(model_dev) and effective_device != "cpu":
                        self.log_line.emit(
                            "[ALERTA] Pediste GPU pero Ultralytics movió el modelo a CPU. "
                            "Causas tipicas: OOM al probar, AMP check fallido, o driver "
                            "incompatible. Revisa la salida arriba."
                        )
                except Exception as e:  # noqa: BLE001
                    self.log_line.emit(f"[WARN] No se pudo verificar device del modelo: {e}")

            def _on_train_end(trainer):
                self.log_line.emit("[INFO] Entrenamiento finalizado.")

            # Cache: ultralytics accepts True/False/'ram'/'disk'
            cache_val: object = p.cache
            if isinstance(cache_val, str) and cache_val.lower() in ("false", "0", "no"):
                cache_val = False

            train_kwargs = dict(  # noqa: C408
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
                # Augmentation
                hsv_h=float(a.hsv_h),
                hsv_s=float(a.hsv_s),
                hsv_v=float(a.hsv_v),
                fliplr=float(a.fliplr),
                mosaic=float(a.mosaic),
                mixup=float(a.mixup),
                copy_paste=float(a.copy_paste),
                # Project
                project=str(proj_dir),
                exist_ok=True,
                verbose=True,
                plots=True,
            )

            comparison = []
            last_best = ""

            for i, family_2 in enumerate(families, start=1):
                # If the user aborted the previous family, we do not start more
                # architectures: the abort applies to the whole run, not per family.
                if run_current["abortado"]:
                    break
                if use_custom:  # noqa: SIM108
                    weights = str(mdl.custom_weights)
                else:
                    weights = mdl.weight_of(family_2)

                # With a single family the name is kept exactly as the
                # user typed it; with several it must be disambiguated or the second run
                # would overwrite the first one (exist_ok=True).
                name_2 = base_name if len(families) == 1 else f"{base_name}_{family_2}"
                run_current["nombre"] = name_2
                st.run_dir = proj_dir / name_2

                # History is per run: if it is not cleared, the curves of the
                # second architecture would be drawn on top of the first one's.
                st.history.clear()
                st.best_map50 = 0.0
                st.best_epoch = 0
                st.epochs_no_improve = 0

                if len(families) > 1:
                    self.log_line.emit("")
                    self.log_line.emit("=" * 60)
                    self.log_line.emit(
                        f"[INFO] Modelo {i} de {len(families)}: {family_2} ({weights})"
                    )
                    self.log_line.emit("=" * 60)

                self.log_line.emit(f"[INFO] Cargando modelo base: {weights}")
                from polyx.core.training import TrainingConfig
                from polyx.core.training import train as _core_train

                cfg = TrainingConfig(
                    weights=weights,
                    data_yaml=str(ds.yaml_path),
                    project=str(proj_dir),
                    name=name_2,
                    epochs=int(p.epochs),
                    batch=int(p.batch),
                    imgsz=int(p.imgsz),
                    device=effective_device,
                    optimizer="SGD",
                    lr0=float(p.lr0),
                    extra=train_kwargs,
                )
                self.log_line.emit("[INFO] Iniciando model.train(...)")
                best_path = _core_train(
                    cfg,
                    on_log=self.log_line.emit,
                    callbacks={
                        "on_train_start": _on_train_start,
                        "on_train_epoch_end": _on_train_epoch_end,
                        "on_train_end": _on_train_end,
                    },
                )

                # Re-select the checkpoint against the domain that matters. best.pt
                # comes from the global mAP, which here is dominated by the
                # laboratory plates; for detection on field photos that is the
                # wrong criterion.
                weight_final = best_path
                f1_real = None
                if getattr(p, "choose_by_real_domain", True):
                    try:
                        chosen, table = choose_better_in_real(
                            proj_dir / name_2,
                            Path(ds.yaml_path),
                            imgsz=int(p.imgsz),
                            batch=int(p.batch),
                            device=effective_device,
                            log=self.log_line.emit,
                        )
                        if chosen is not None:
                            weight_final = chosen
                            f1_real = table[0]["f1"] if table else None
                    except Exception as exc:  # noqa: BLE001
                        # A failed re-selection cannot throw away an hours-long
                        # training run: it is reported and best.pt is kept.
                        self.log_line.emit(
                            f"[WARN] No se pudo reelegir checkpoint: {exc}. Se conserva best.pt."
                        )

                if weight_final.exists():
                    last_best = str(weight_final)
                comparison.append(
                    {
                        "familia": family_2,
                        "peso_base": weights,
                        "run": name_2,
                        "best": str(weight_final) if weight_final.exists() else "",
                        "map50": st.best_map50,
                        "epoca_mejor": st.best_epoch,
                        "f1_real": f1_real,
                    }
                )

            # User abort: it is NOT a termination. The completion event is not emitted
            # and neither is finished_ok (that would tell the UI there were final
            # weights when there is only a truncated training run).
            if run_current["abortado"]:
                obs.event(
                    log,
                    "train.aborted",
                    ms=int((__import__("time").perf_counter() - _t0) * 1000),
                    familias_completadas=len(comparison),
                )
                self.aborted.emit()
                return

            if len(comparison) > 1:
                self._report_comparison(comparison, p)

            obs.event(
                log,
                "train.complete",
                ms=int((__import__("time").perf_counter() - _t0) * 1000),
                weights=str(last_best),
                familias=len(comparison),
            )
            self.finished_ok.emit(last_best)

        except KeyboardInterrupt:
            self.aborted.emit()
            obs.event(self._log(), "train.aborted")
        except Exception as e:  # noqa: BLE001
            import traceback

            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            obs.event_error(self._log(), "train.error", e)

    def _log(self):
        return obs.logger_for("trainer")
