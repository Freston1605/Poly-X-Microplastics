# Dataset card — Poly-X

Labelled YOLO dataset for automated detection of PET, PP and LDPE microplastics
in Nile Red fluorescence images.

## Status

The dataset is **not** committed to the repository. It is maintained locally
and shared through the project owner’s storage. This card describes its
contents and the split procedure.

## Images

| Split | Images | How its images were chosen |
|---|---:|---|
| `train` | 35 | Plate-balanced training split |
| `val` | 11 | Held-out plates and polymer mixes; never seen during training |
| `test` | 10 | Separate hold-out for final reporting |
| `pending/` | 1 | `LDPE 11`, awaiting human annotation; excluded from every split |

All image paths are relative to `data/`, e.g. `data/images/val/`.

## Classes

| Id | Class | Notes |
|---:|---|---|
| 0 | PET | Most distinct polyester |
| 1 | PP | Resembles LDPE; separated mainly by brightness/greenish tinge |
| 2 | LDPE | Often bright and yellowish |

## Domains / provenance

- **Spiked sediment plates** — sediment samples spiked with known polymers
  (Sedcalci and `sed marino` series).
- **Mixtures** (`Mezc N.jpg`) — known multi-polymer mixtures.
- **Pure plates** — individual PET/PP/LDPE plates.

Every split contains representatives from all three domains.

## Known annotation characteristics

The dataset was labelled manually. Measured on the retained data (LDPE 11 is
currently unannotated):

- **PET** — 4,376 boxes; median box area 2352 px² at native resolution.
- **PP** — 5,196 boxes; median box area 682 px².
- **LDPE** — 4,956 boxes; median box area 768 px².
- Spatial distribution is centred on the plate, as expected for plate scans.
- No box extends outside image bounds and no zero-area boxes are present.
- Aspect ratio is largely compact: median 1.19, no boxes ≥ 4:1. Rejecting a
  model purely because it expects "fibres" would be wrong for this material.

## Known limitations

- **PP and LDPE are not fully separable by colour alone.** PP is measured at
  (B=133, G=135, R=29), LDPE at (B=162, G=151, R=62), and about 18.6% of PP
  sampling points sit nearer LDPE's centroid. This is why the assignment rule
  exists and why recall for PP/LDPE is far below PET.
- **PET is larger** (median 49 px side) than PP (26 px) and LDPE (27 px), so
  PET performance partially reflects image size, not only chemistry.
- **Manual annotation noise** matters. Two annotations of a byte-identical
  LDPE image had 1052 vs 698 boxes (~34% disagreement). Duplicates were
  resolved: LDPE 10 is kept as a lower-quality duplicate of LDPE 8 and LDPE
  12 as an unlabelled copy of LDPE 9; both were removed from the dataset.
- **`LDPE 11`** is excluded from `train`, `val` and `test` and lives in
  `data/pending/`. It will be moved back into the tree and added to one of the
  splits only after a human has annotated it. Do not fabricate boxes for it.
- **PE/PS spikings in the plates are not represented as classes.** They are
  not annotated; the dataset is PET/PP/LDPE only.
- **Acquisition shift.** Hand-held/re-exported images differ in compression,
  and some plates contain multiple focal planes; expect domain drift if
  microscopy conditions change.

## Split method

Plate groups are kept intact to avoid leakage. `tools/data/make_splits.py`
contains the exact group assignment and verifies it before moving files. The
assignment was also chosen to keep per-class box counts roughly proportional
across train/val/test (see `tools/data/README.md`).

## Reproduction

```bash
python tools/data/make_splits.py --root data          # dry run
python tools/data/make_splits.py --root data --apply  # writes split
python -m pytest -q
```
