# Architecture Review — Adding Two-Stage Segmentation Models

**Scope:** readiness of the Poly-X detection/measurement pipeline for
two-stage computer-vision segmentation (U-Net, SAM, YOLO-seg, Mask R-CNN, …).
**Reviewer:** code-review skill
**Status:** draft for discussion (not a merge gate)

## 1. What already works in our favour

- A clean `DetectionSource` protocol (`src/polyx/core/detect/protocol.py`) with a
  factory/registry (`factory.py`: `make_detector`, `register_detector`). Adding a
  new backend is a registration call, not a refactor.
- Canonical `Instance` dataclass is the single interchange type between
  detection, measurement, reporting and GUI. Good seam.
- SAM2 already exists as a prompt-driven segmenter (`sam.py`), proving the
  two-stage pattern (detector → segmenter) is reachable.
- Measurement is deliberately mask-based (`core/measure/segmentation.py` + `shape.py` + `drawing.py`) and the
  box is only a fallback. The science already assumes a mask exists.
- Provenance/observability layer (`core/provenance`) records run metadata —
  essential for reproducible two-stage experiments.

## 2. Blocking gaps for segmentation

### 2.1 `Instance` carries no mask

`Instance` has box fields + morphology scalars, but no `mask`, `polygon`, or RLE.
Every segmentation backend therefore has to throw its mask away and re-derive a box,
and morphology then re-segments the box with Otsu — discarding the one thing the
segmenter produced. This is the single biggest blocker.

**Recommendation:** add an optional mask payload to `Instance`:

```python
mask: Optional[np.ndarray] = None          # HxW bool, in full-image coords
mask_rle: Optional[str] = None             # COCO RLE for persistence
mask_source: Optional[str] = None          # "yolo-seg" | "sam2" | "otsu-box"
```

Keep it optional and lazily materialised so detection-only backends are unaffected.
Memory caveat: store RLE (or a cropped mask + offset) for large images, not a full
HxW array per instance.

### 2.2 No segmentation-specific protocol

`DetectionSource.predict` returns boxes. A segmentation model returns masks + boxes.
Today `Sam2Segmenter.predict` raises `NotImplementedError` because the protocol
does not express "I need prompts". Two-stage orchestration (detector → segmenter)
has no first-class home.

**Recommendation:** introduce a parallel protocol:

```python
class SegmentationSource(Protocol):
    def segment(self, image, prompts: SegmentationPrompts) -> List[Instance]: ...
```

and a `TwoStageDetector` composer that wires a `DetectionSource` (stage 1) into a
`SegmentationSource` (stage 2, e.g. SAM2) and merges masks back onto instances.
Keep `DetectionSource` for end-to-end det/seg models (YOLO-seg, Mask R-CNN) that
do both in one pass.

### 2.3 Measurement ignores an available mask

`morfologia.medir` (and `compute_box_size_um`) segment inside the box with Otsu.
If a real mask exists it should be used directly; Otsu should be the fallback
only.

**Recommendation:** change the measurement entry point to accept `mask` first:

```python
def medir_deteccion(det, um_per_px, mask=None):
    if mask is None:
        mask = segmentar_caja(det, img)   # current Otsu path
    ...
```

This keeps the synthetic-shape validation tests intact while letting segmentation
models feed their masks in.

### 2.4 Factory/registry is name→class only

`make_detector(name, weights, **kwargs)` assumes one weights file and one class.
Segmentation models need config files (SAM2 `model_cfg`), class maps, and
pre/post-processing. YOLO-seg needs a different task (`task="segment"`).

**Recommendation:** widen the registry entry to a small factory callable:

```python
_DETECTOR_REGISTRY: Dict[str, Callable[..., DetectionSource]]
```

and let each backend own its construction (SAM2 already does). Add a `task` field
(`"detect" | "segment"`) so the GUI and trainer know which pipeline to run. Also
add `"yolo-seg"` and `"maskrcnn"` entries.

### 2.5 GUI assumes detection semantics

- `models.py` page loads up to 3 `.pt` files; no model-type selector.
- `parameters.py` exposes conf/IoU/imgsz but no segmentation thresholds
  (mask score, min mask area, NMS on masks).
- `results.py`/`report_html.py` render boxes; masks are never drawn or exported.
- Trainer page trains detection only (`yolo` task).

**Recommendation:** add a `ModelSpec` (name, task, weights, config) in
`DetectorState`, and branch the results/report rendering on `task`. Export masks
as PNG overlays + RLE in the run folder alongside `centroids.csv`.

### 2.6 Ground-truth format is box-only

`read_yolo_txt` parses `class cx cy w h`. Segmentation GT is polygon or RLE.
Metrics (`metrics.py` `match_image` uses IoU on boxes) cannot score mask IoU.

**Recommendation:** extend GT reading to optional polygon lines (YOLO-seg format:
`class x1 y1 x2 y2 ...`) and add mask-IoU matching alongside box-IoU, keeping
both reported (box IoU for comparability with prior runs).

### 2.7 Two-stage bookkeeping and provenance

A run that uses "YOLO boxes → SAM2 masks" is methodologically different from
"YOLO-seg end-to-end", but the report does not record which. The report already
records tiling decisions; it should also record stage composition.

**Recommendation:** add a `pipeline` descriptor to the run provenance
(`core/provenance`): list of stages with model name + weights hash + params.
The report's Methods section should render it verbatim.

## 3. Suggested implementation order (smallest viable increments)

1. **Mask on `Instance`** + RLE helpers (no behaviour change).
2. **Measurement accepts a mask** (fallback to Otsu). Tests in
   `tests/test_morfologia.py` keep passing.
3. **`TwoStageDetector`** composer + SAM2 wiring; add `segment_from_boxes` result
   to carry masks.
4. **YOLO-seg backend** (`task=segment`) reading `r.masks` into `Instance.mask`.
5. **GT + metrics** for polygons and mask IoU.
6. **GUI** model-type selector + mask rendering/export.
7. **Trainer** segmentation task (or document that training stays external).

## 4. Risks / things to watch

- **Memory:** per-instance full masks at 4k resolution will blow RAM on a plate.
  Prefer cropped masks + offsets or RLE; decide early.
- **Class maps:** segmentation models may use different class IDs than
  PET/PP/LDPE=0/1/2. Centralise the mapping in `ModelSpec`, do not hard-code.
- **NMS on masks** differs from box NMS; YOLO-seg handles it internally but
  SAM2 outputs need deduplication by mask IoU.
- **Backward compatibility:** old runs have no masks; report code must treat
  `mask is None` as "box-derived" and label it as such (scientific honesty).
- **Test coverage:** add synthetic-image tests for each backend behind a
  `pytest.importorskip` so CI without torch/sam2 still passes.

## 5. Open questions for the maintainer

1. Should SAM2 be an optional refinement stage or the primary segmenter?
2. Do we want per-particle mask export in the HTML report, or only in the run folder?
3. Training of U-Net/Mask R-CNN: in-app or documented external (Colab/notebook)?
4. How to handle overlapping particles — merge masks or keep separate instances?

---

*This review is advisory. No code was changed.*