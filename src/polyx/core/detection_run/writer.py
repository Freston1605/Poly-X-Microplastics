"""Write the canonical detection-run folder as results are produced.

The folder belongs to the caller (``RunConfig.output_dir``); this writer only
knows the canonical layout: ``run.json``, ``detections.csv`` and one folder per
model with the triptych views and prediction labels. Writing incrementally is
the point: a failure halfway through a batch must leave a readable partial run,
not an empty directory that only makes sense after a successful return.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from polyx.core.annotate import Triptych, run_label_path, save_view
from polyx.core.labels import bbox_from, write_lines
from polyx.core.measure.calibration import Calibration

from .result import ImageResult, RunResult
from .schema import DETECTION_COLUMNS, build_run_payload, detection_row


class RunWriter:
    """Append-only CSV plus a run record rewritten after every result.

    ``run.json`` is rewritten atomically: a crash between two writes leaves the
    previous valid document, never half a JSON object.
    """

    def __init__(self, run_dir: Path, result: RunResult, provenance_data: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._result = result
        self._provenance = provenance_data or {}
        self.csv_path = self.run_dir / "detections.csv"
        self._csv_handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(
            self._csv_handle, fieldnames=list(DETECTION_COLUMNS), extrasaction="ignore"
        )
        self._csv.writeheader()
        self._csv_handle.flush()
        self._csv_closed = False
        self.write_run_json()

    def write_run_json(self) -> None:
        """Persist the current state of the run record."""
        payload = build_run_payload(self._result, self._provenance)
        target = self.run_dir / "run.json"
        temporary = self.run_dir / ".run.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def record_result(
        self,
        image_result: ImageResult,
        calibration: Calibration | None,
        views: Triptych,
        image_index: int,
        width: int,
        height: int,
    ) -> None:
        """Write one model/image result and publish it in the run record."""
        alias = self._result.models[image_result.model_idx].alias
        folder = self.run_dir / alias
        folder.mkdir(parents=True, exist_ok=True)

        image_result.annotated_path = save_view(
            views.combined, folder, image_index, image_result.image_path.stem, "_annot"
        )
        image_result.pred_path = save_view(
            views.pred, folder, image_index, image_result.image_path.stem, "_pred"
        )
        image_result.gt_path = (
            save_view(views.gt, folder, image_index, image_result.image_path.stem, "_gt")
            if views.gt is not None
            else None
        )

        # Predictions in text, 6 columns (confidence last), never next to the
        # source photo: that is where the manual ground truth lives.
        write_lines(
            run_label_path(folder / "labels", image_index, image_result.image_path.stem),
            [bbox_from(det, width, height) for det in image_result.predictions],
            include_conf=True,
        )

        for det in image_result.predictions:
            self._csv.writerow(detection_row(alias, image_result.image_path, det, calibration))
        self._csv_handle.flush()
        self.write_run_json()

    def record_skip(self) -> None:
        """Publish a skipped image without adding a model result."""
        self.write_run_json()

    def finish(self) -> None:
        """Close the CSV and write the final run record."""
        if not self._csv_closed:
            self._csv_handle.flush()
            self._csv_handle.close()
            self._csv_closed = True
        self.write_run_json()

    def close(self) -> None:
        """Release the CSV handle without rewriting ``run.json``."""
        if not self._csv_closed:
            self._csv_handle.close()
            self._csv_closed = True


__all__ = ["RunWriter"]
