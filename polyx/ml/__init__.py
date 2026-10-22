"""Pure, Qt-free computer-vision engine for Poly-X.

This package holds the entire headless engine: YOLO inference
(:mod:`yolo_wrap`), morphology (:mod:`morfologia`), plate calibration
(:mod:`calibracion`), matching and metrics (:mod:`metrics`,
:mod:`asignacion`), dataset auditing (:mod:`dataset_audit`), observability
(:mod:`observabilidad`), project paths (:mod:`paths`), platform detection
(:mod:`plataforma`), file provenance (:mod:`procedencia`) and translations
(:mod:`i18n`, :mod:`traducciones_en`).

Import contract: this package must stay importable with no PySide6 (or any
Qt binding) module loaded — see ``tests/test_model_package.py``. All engine
code imports from this package directly.
"""
