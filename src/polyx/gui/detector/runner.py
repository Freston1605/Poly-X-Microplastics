"""Qt adapter around the canonical, Qt-free detection engine.

The runner owns the thread and the model loading; everything else -- decode,
calibration, ground truth, prediction, measurement, matching, drawing and the
run folder -- belongs to :func:`polyx.core.detection_run.run_detection`. The
engine numbers its active models densely (``0..N-1``); this adapter maps those
back onto the GUI's real ``model_slots`` indices before publishing results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from polyx.core import observability as obs
from polyx.core.detect import make_detector
from polyx.core.detection_run import (
    ImageResult,
    ModelSpec,
    RunConfig,
    RunFailed,
    RunResult,
    RunStatus,
    run_detection,
)
from polyx.core.i18n import _
from polyx.core.paths import RUNS_DIR

from .state import DetectorState


class DetectorRunner(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(int, object)  # real model slot index, ImageResult
    finished_ok = Signal()
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: DetectorState, parent=None, runs_root: Path | None = None):
        super().__init__(parent)
        self.state = state
        # Where runs are created: the per-user data dir (ADR-0001), unless
        # the caller injects a root (the tests point it at tmp_path).
        self._runs_root = RUNS_DIR if runs_root is None else runs_root
        # Dense engine index -> real index within ``state.model_slots``.
        self._active_indices: list[int] = []

    def _load_models(self, slots, _log) -> None:
        """Lazily resolve and load every active model. Raises on the first failure."""
        for s in slots:
            if s.loaded is None:
                s.loaded = make_detector(s.model_type or "yolo", str(s.path), alias=s.alias)
            try:
                s.loaded.load()
                obs.event(_log, "model.loaded", alias=s.alias, file=str(s.path))
            except Exception as exc:
                obs.event_error(_log, "model.error", exc, alias=s.alias, file=str(s.path))
                raise

    def _create_run_dir(self, state: DetectorState) -> str:
        """Create this run's folder under the runs root and return its stamp.

        The stamp is what ``obs.set_run_id`` correlates every event of the
        batch with.
        """
        self._runs_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.run_dir = self._runs_root / f"detect_{stamp}"
        state.run_dir.mkdir(parents=True, exist_ok=True)
        return stamp

    def _build_config(self, state: DetectorState, slots, output_dir: Path) -> RunConfig:
        """Translate the GUI state into the engine's run configuration.

        The GUI resolves scales per photo: the plate on the photo, else the
        manual value. That is exactly the engine's ``auto`` policy;
        pre-canonical index CSVs are not read.
        """
        params = state.params
        models: list[ModelSpec] = []
        for slot in slots:
            if slot.loaded is None:
                raise RuntimeError(f"model {slot.alias!r} is not loaded")
            models.append(
                ModelSpec(
                    detector=slot.loaded,
                    alias=slot.alias,
                    alias_generated=slot.alias_generated,
                    path=slot.path,
                    model_type=slot.model_type or "yolo",
                    names=dict(slot.names or {}),
                    sha256=slot.sha256,
                )
            )
        return RunConfig(
            images=list(state.images),
            models=models,
            output_dir=output_dir,
            params=params,
            gt_folder=state.gt_folder,
            calibration=state.calibration,
        )

    def _on_progress(self, done: int, total: int, image_path: Path) -> None:
        """Forward the engine's (models x images) counter to the Qt signal."""
        self.progress.emit(done, total, Path(image_path).name)

    def _on_result(self, result: ImageResult) -> None:
        """Publish one engine result under its real GUI model slot."""
        real_idx = self._active_indices[result.model_idx]
        self.state.results.setdefault(real_idx, []).append(result)
        self.image_done.emit(real_idx, result)

    def _publish(self, result: RunResult) -> None:
        """Adopt the engine's run record, keeping the streamed results as they are.

        :meth:`_on_result` is the single write path: it appends each
        ``ImageResult`` to ``state.results`` under its real slot as the engine
        produces it, which is what lets the pages show progress. A completed
        run therefore holds every result exactly once, and a run that aborts
        or fails holds exactly the results received before it stopped.

        This final sync only adopts the run-level fields and fills the one gap
        the streaming can lose: a failure between the engine recording a
        result and its callback firing. Missing objects are appended; the
        streamed lists are never rebuilt, so nothing is written twice.
        """
        state = self.state
        state.run_result = result
        state.run_dir = result.run_dir
        for dense_idx, rows in result.results.items():
            published = state.results.setdefault(self._active_indices[dense_idx], [])
            for row in rows:
                if not any(row is entry for entry in published):
                    published.append(row)

    def _log_finish(self, result: RunResult, started: datetime, _log) -> None:
        """Emit the closing observability events of a completed run."""
        ms = int((datetime.now() - started).total_seconds() * 1000)
        execution = result.execution
        shapeless = {
            result.models[index].alias: stats.shapeless
            for index, stats in execution.items()
            if stats.shapeless
        }
        obs.event(
            _log,
            "run.finish",
            ms=ms,
            plates=len(result.images),
            imgsz_used=sorted(
                {int(value) for stats in execution.values() for value in stats.imgsz_used}
            ),
            shapeless=shapeless,
        )
        worst = max(shapeless.items(), key=lambda item: item[1], default=())
        if worst and worst[1] > 0:
            obs.event(
                _log,
                "run.shapeless_high",
                level=logging.WARNING,
                model=worst[0],
                shapeless=worst[1],
            )

    def run(self):
        _log = obs.logger_for("detector")
        obs.set_run_id(None)
        state = self.state
        started = datetime.now()
        try:
            slots = state.active_models()
            params = state.params
            obs.event(
                _log,
                "run.config",
                plates=len(state.images),
                models=[s.alias for s in slots],
                conf=params.conf,
                iou=params.iou_nms,
                imgsz=params.imgsz,
                device=params.device,
                measure_shape=params.measure_shape,
                um_per_px=state.calibration.um_per_px,
            )
            if not slots:
                self.failed.emit(_("No models loaded."))
                return
            if not state.images:
                self.failed.emit(_("No images selected."))
                return

            self._active_indices = [state.model_slots.index(slot) for slot in slots]
            stamp = self._create_run_dir(state)
            obs.set_run_id(stamp)

            self._load_models(slots, _log)

            state.results = {i: [] for i in range(len(state.model_slots))}
            state.run_result = None
            state.verdicts = {}
            config = self._build_config(state, slots, state.run_dir)
            result = run_detection(
                config,
                on_progress=self._on_progress,
                on_result=self._on_result,
                should_abort=state.consume_abort,
            )
        except RunFailed as exc:
            self._publish(exc.result)
            obs.event_error(_log, "run.error", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            obs.event_error(_log, "run.error", exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self._publish(result)
            if result.status is RunStatus.ABORTED:
                self.aborted.emit()
                return
            self._log_finish(result, started, _log)
            self.finished_ok.emit()
