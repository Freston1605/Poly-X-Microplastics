"""Qt presentation layer for Poly-X.

This package owns every module that touches PySide6 and every module that
renders a report: the theme tokens (:mod:`polyx.gui.theme`), the vector icons
(:mod:`polyx.gui.iconos`), the per-window identity (:mod:`polyx.gui.marca`),
the shared widgets (:mod:`polyx.gui.widgets`), the preferences dialog
(:mod:`polyx.gui.preferencias`), the HTML report generator
(:mod:`polyx.gui.report_html`), the HTML to PDF converter
(:mod:`polyx.gui.pdf_export`) and the update checker
(:mod:`polyx.gui.updates`). The feature modules (:mod:`polyx.gui.detector`,
:mod:`polyx.gui.trainer`, :mod:`polyx.gui.viewer`, :mod:`polyx.gui.labeler`) and the
launcher also belong here and already live in this package (M005).

Import contract: this package is allowed to depend on ``polyx.core`` (the
pure engine), but the engine must never depend back on it. The legacy
re-export package that used to accept the old ``core`` module names is gone
(milestone M005, slice S03): consumers import from ``polyx.core`` and
``polyx.gui`` directly. This ``__init__`` deliberately imports no submodule,
so ``import polyx.gui`` stays cheap and Qt-free.
"""
