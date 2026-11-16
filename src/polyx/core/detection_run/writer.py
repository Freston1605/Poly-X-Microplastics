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
from polyx.core.labels import bbox_from, write_classes, write_lines
from polyx.core.measure.calibration import Calibration

from .result import ImageResult, RunResult
from .schema import DETECTION_COLUMNS, build_run_payload, detection_row, read_json_object


class RunWriter:
    """Append-only CSV plus a run record rewritten after every result.

    ``run.json`` is rewritten atomically: a crash between two writes leaves the
    previous valid document, never half a JSON object.

    ``update=True`` opens a finished folder to rewrite results in place (the GT
    reload does). It keeps the provenance the folder already recorded and does
    not truncate ``detections.csv``; see :meth:`update_image`.
    """

    def __init__(
        self,
        run_dir: Path,
        result: RunResult,
        provenance_data: dict | None = None,
        *,
        update: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._result = result
        if provenance_data is None and update:
            provenance_data = read_json_object(self.run_dir / "run.json").get("provenance")
        self._provenance = provenance_data or {}
        self.csv_path = self.run_dir / "detections.csv"
        if update:
            # The folder is already the run's own record: rewriting it starts
            # from what is there, so nothing is truncated or recreated here.
            self._csv_handle = None
            self._csv = None
            self._csv_closed = True
            return
        # One class list per model folder: the run can hold models trained on
        # different class sets, and the prediction labels next to it need the
        # right order to reopen as pre-annotation.
        for info in result.models:
            folder = self.run_dir / info.alias
            folder.mkdir(parents=True, exist_ok=True)
            if info.names:
                write_classes(folder, [info.names[key] for key in sorted(info.names)])
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

        self._write_views(image_result, views, image_index, with_predictions=True)

        # Predictions in text, 6 columns (confidence last), never next to the
        # source photo: that is where the manual ground truth lives.
        write_lines(
            run_label_path(folder / "labels", image_index, image_result.image_path.stem),
            [bbox_from(det, width, height) for det in image_result.predictions],
            include_conf=True,
        )

        table, handle = self._csv, self._csv_handle
        assert table is not None and handle is not None  # update mode opens no table
        for det in image_result.predictions:
            table.writerow(detection_row(alias, image_result.image_path, det, calibration))
        handle.flush()
        self.write_run_json()

    def update_image(self, image_result: ImageResult, views: Triptych, image_index: int) -> None:
        """Redraw one image's views and republish it in an existing run folder.

        The prediction views and ``detections.csv`` are deliberately left
        alone: they describe predictions, and a GT reload never runs inference
        -- the table's rows are still exactly the predictions in
        ``image_result``. Rewriting them would only risk replacing columns a
        result reopened from disk cannot rebuild (``has_mask`` is recorded but
        the masks themselves are not persisted). ``run.json`` is the record
        that carries the GT and the counts, so it is rewritten.
        """
        self._write_views(image_result, views, image_index, with_predictions=False)
        self.write_run_json()

    def _write_views(
        self,
        image_result: ImageResult,
        views: Triptych,
        image_index: int,
        *,
        with_predictions: bool,
    ) -> None:
        """Save the views that exist under the run's file-naming contract."""
        folder = self.run_dir / self._result.models[image_result.model_idx].alias
        folder.mkdir(parents=True, exist_ok=True)
        stem = image_result.image_path.stem
        image_result.annotated_path = save_view(views.combined, folder, image_index, stem, "_annot")
        if with_predictions:
            image_result.pred_path = save_view(views.pred, folder, image_index, stem, "_pred")
        image_result.gt_path = (
            save_view(views.gt, folder, image_index, stem, "_gt") if views.gt is not None else None
        )

    def record_skip(self) -> None:
        """Publish a skipped image without adding a model result."""
        self.write_run_json()

    def finish(self) -> None:
        """Close the CSV and write the final run record."""
        if not self._csv_closed:
            assert self._csv_handle is not None  # an open table has an open handle
            self._csv_handle.flush()
            self._csv_handle.close()
            self._csv_closed = True
        self.write_run_json()


__all__ = ["RunWriter"]
