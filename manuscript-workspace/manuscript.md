# DRAFT — NOT FOR SUBMISSION

This scaffold is intentionally incomplete. Remove this banner only after every local
validator passes and accountable human authors approve the final document.

# Poly-X: an open-source pipeline for automated detection and morphological characterisation of Nile Red-stained microplastics

## Abstract

Poly-X is an open-source desktop application for the automated detection and classification of microplastics (PET, PP, LDPE) in Nile Red fluorescence images acquired under UV excitation. [claim:C001] [evidence:E001]

It integrates four modules: a batch detector, a YOLO trainer, a manual labeller, and a single-image viewer with metric calibration. [claim:C002] [evidence:E001]

The detector reports per-particle position, polymer class, confidence, and morphology (area-equivalent diameter, length, width, aspect ratio, curvature) in micrometres after calibration. [claim:C003] [evidence:E001]

An abstention mechanism separates detection from identification: particles below a confidence threshold are counted but reported as unassignable rather than forced into a polymer class. [claim:C004] [evidence:E001]

The morphology engine measures particles over the segmented mask rather than the bounding box, because the box overestimates area by a factor of 1.87 on the annotated set. [claim:C005] [evidence:E001,E004]

Measured over 7129 annotated particles, the box overestimates area by 1.87x. [claim:C006] [evidence:E001]

Length is taken as the larger of the maximum Feret diameter and, when the particle is slender and non-convex, the geodesic diameter. [claim:C007] [evidence:E001,E004]

Against synthetic shapes of known size the engine achieves a median error of 0.6 percent and a worst case of 4.7 percent. [claim:C008] [evidence:E001,E004]

Real Nile Red emission differs from the interface colours used for boxes: PET 116/58/65, PP 122/125/32, LDPE 181/162/57 (mean RGB). [claim:C009] [evidence:E005]

PP and LDPE share hue and are separated by brightness, which explains the lower recall for the two polyolefins. [claim:C010] [evidence:E001,E005]

## Introduction

Microplastic analysis by Nile Red fluorescence is widely used, but throughput and reproducibility remain limited by manual counting. [claim:C011] [evidence:E002,E003]

Existing pipelines rarely combine detection, training, labelling, and inspection in one tool, and morphological reporting is often derived from bounding boxes. [claim:C012] [evidence:E001]

Poly-X addresses this by wrapping Ultralytics YOLO inference and training in a portable PySide6 desktop application with a paper-quality HTML report. [claim:C013] [evidence:E001,E006]

The present paper describes the design and the quantitative behaviour of the detection and morphology components. [claim:C014] [evidence:E001]

## Methods

### Software and design

Poly-X is written in Python 3.11 and uses Ultralytics 8.3.40 for YOLO inference and training, OpenCV 4.10 for image I/O, NumPy 1.26 for computation, and PySide6 6.7 for the GUI. [claim:C015] [evidence:E001,E006]

All OS-specific behaviour is confined to polyx/model/plataforma.py (device selection, file opening, updater launch). [claim:C016] [evidence:E001]

Images are read and written with cv2.imdecode/imencode over np.fromfile to support accented paths on Windows. [claim:C017] [evidence:E001]

### Detection

The detector runs a YoloModel wrapper around Ultralytics YOLO and returns Detection objects with class_id, class_name, conf, bbox, and optional morphology. [claim:C018] [evidence:E001]

Up to three models can be loaded and compared in parallel. [claim:C019] [evidence:E001]

Ground truth, when present as YOLO .txt, is matched to predictions by IoU at threshold 0.50. [claim:C020] [evidence:E001]

TP, FP, FN and misclassification counts are reported per class. [claim:C021] [evidence:E001]

### Calibration

The viewer supports interactive calibration in two modes: a two-click line (real length) and a three-click circle (real diameter). [claim:C022] [evidence:E001]

The circle dialog defaults to 100000 um, the Petri dish diameter used in the study. [claim:C023] [evidence:E001]

Once calibrated, diameters are reported in micrometres. [claim:C024] [evidence:E001]

### Morphology

Segmentation is performed inside each box with Otsu thresholding on the highest-contrast channel, keeping the central connected component clipped to the box. [claim:C025] [evidence:E001]

Length is the larger of the maximum Feret diameter and the geodesic diameter, the latter used only when the particle is slender (length/thickness >= 4) and non-convex (solidity < 0.90). [claim:C026] [evidence:E001,E004]

The equivalent rectangle L = (P + sqrt(P^2 - 16A))/4 is retained as a descriptor but not used as a size. [claim:C027] [evidence:E001]

The morphology cost is 5.9 ms per particle using only cv2 and numpy. [claim:C028] [evidence:E001]

### Abstention

Below conf_asignacion the particle still counts as detected but is reported as unassignable; the report states the fraction of the batch that went unassigned. [claim:C029] [evidence:E001]

Nile Red is solvatochromic, so emission reflects polarity rather than polymer identity; PET separates cleanly while PP and LDPE, both apolar polyolefins, share hue. [claim:C030] [evidence:E001]

An unknown class was deliberately not trained, because its definition would depend on annotator doubt. [claim:C031] [evidence:E001]

### Labelling

The labeller records three states: reviewed with n particles, reviewed with zero particles, and not reviewed. [claim:C032] [evidence:E001]

A .txt label file is written only when the image is marked reviewed or a box is drawn, so that skimming does not create spurious zero counts. [claim:C033] [evidence:E001]

Minimum box side is 2 px. [claim:C034] [evidence:E001]

The labeller was verified with 552 crops, with startup 0.25 s and interface response 0.06 s. [claim:C035] [evidence:E001]

## Results

### Detection performance

PET recall is 0.98, PP recall 0.70, and LDPE recall 0.54. [claim:C036] [evidence:E001]

PP and LDPE are the dominant confusion, because they share hue and differ in brightness, which depends on exposure, focus, and thickness. [claim:C037] [evidence:E001,E005]

### Morphology accuracy

Against synthetic shapes of known size the morphology engine achieves a median error of 0.6 percent and a worst case of 4.7 percent. [claim:C038] [evidence:E001,E004]

The maximum Feret diameter underestimates length on curved fibres by 34.6 percent on a 180 degree arc. [claim:C039] [evidence:E001]

The geodesic diameter rounds concavities on thick particles, which motivated the slenderness gate. [claim:C040] [evidence:E001]

The equivalent rectangle inflates length by 22.5 percent on a straight line with saw teeth and is undefined for a circle. [claim:C041] [evidence:E001]

A re-analysis of 6638 particles with Feret/geodesic gave a median aspect ratio of 1.58, maximum 8.7, and 1.1 percent fibres, correcting an earlier report of 22 percent fibres. [claim:C042] [evidence:E001]

### Fluorescence

Mean RGB over the interior of 30 boxes per class: PET 116/58/65 (red-salmon), PP 122/125/32 (dull yellowish green), LDPE 181/162/57 (plain yellow, bright). [claim:C043] [evidence:E005]

PP is not orange, contrary to earlier documentation. [claim:C044] [evidence:E001,E005]

## Discussion

The results show that Poly-X provides reproducible, paper-quality reporting of microplastic counts and morphology, while making the detection-versus-identification distinction explicit. [claim:C045] [evidence:E001]

The morphology correction (1.1 percent fibres versus the earlier 22 percent) illustrates the risk of deriving length from bounding boxes. [claim:C046] [evidence:E001]

Limitations: the morphology validation uses synthetic shapes; field performance on irregular, overlapping particles is not yet quantified. [claim:C047] [evidence:E001]

Generalisation to polymers beyond PET, PP, and LDPE, and to non-fluorescent stains, is not supported. [claim:C048] [evidence:E001]

## Author contributions

To be completed by the accountable author before submission.

## Funding

To be completed by the accountable author before submission.

## Competing interests

To be completed by the accountable author before submission.

## Ethics and consent

To be completed by the accountable author before submission.

## Data and code availability

To be completed by the accountable author before submission.

## AI-use disclosure

To be completed by the accountable author before submission.

## References

1. Perez M, Parra S, Ferrada C, Bravo M, Perez PA, Quiroz W (2024). Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. PLoS ONE 19(12): e0314520. https://doi.org/10.1371/journal.pone.0314520 [claim:C049] [evidence:E002]

2. Ferrada C, Perez M, Parra S, Salas E, Sepulveda F, Bravo MA, Quiroz W (2024). Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in Merluccius gayi fish by Nile Red fluorescent staining and image analysis. J. Chil. Chem. Soc. 69(1): 6082-6085. https://doi.org/10.4067/s0717-97072024000106082 [claim:C050] [evidence:E003]