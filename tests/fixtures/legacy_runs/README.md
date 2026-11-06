# Legacy run fixtures

Small, frozen folders written by the two pre-canonical consumers:

- `headless/run/` is the old `run_batch` layout (`summary.json`,
  `centroids.csv`, `images/*.png`). `headless/photos/` holds the original
  photographs the old headless folder did not copy.
- `gui/detect_20260101_120000/` is an old Detector run as the Viewer sees it:
  one folder per model with `labels/{index:04d}_{stem}.txt` and
  `classes.txt`. `gui/photos/` holds the original photograph.

They are text-first on purpose: the tests pin the readers, and the images are
only there so the Viewer and the full report have a real file to open.
