# Baseline de identidad de bytes — informe del pipeline (M001, congelado para M004)

```python
"""
Purpose: provenance record for the byte-identity refactor gate of M004.

The benign report snapshot under `benigna/` is THE gate every M004
extraction (S02/S03 section builders pulled out of polyx/gui/report_html.py)
must keep byte-identical. This README records who captured it, when it was
re-verified, and why it must not be regenerated.

Frozen render metadata (re-verified live in M004/S01, pre-refactor):
    sha256  = bf483e638b7e54ab6e21a2cb018a02ee20bf92ebe8d088ac45eba9f3b2316727
    bytes   = 139512
"""
```

## Propósito

Este directorio guarda el baseline de identidad de bytes del informe
generado por `polyx.gui.report_html.generate_report`. Es **LA puerta
(vgate) de refactor de M004**: cada extracción de section builder que se
haga en S02/S03 debe dejar el render benigno byte a byte idéntico a
`benigna/informe.html`. Los tests que hacen cumplir la puerta viven en
`tests/test_render_reporte_benigno.py` (identidad de bytes, sha256 del
render, autoconsistencia del fixture y, desde M004/S01, el pin del tamaño
en bytes vía `TAMANO_BASELINE`).

## Procedencia

- **Captura original:** `scripts/capturar_fixtures_informe.py`, ejecutado
  en **M001/T03** contra el código pre-fix. El script renderiza los dos
  estados de fixture de `tests/fixtures_informe.py` por el pipeline real
  (todas las secciones, wrapper de cronómetro incluido) y persiste cada
  `informe.html` junto a su `informe.sha256`.
- **Re-verificación en vivo:** durante la planificación de **M004/S01**
  (pre-refactor) se recalculó el digest del archivo en el repo:
  `bf483e638b7e54ab6e21a2cb018a02ee20bf92ebe8d088ac45eba9f3b2316727`,
  **139512 B**. Es decir, el snapshot que hay aquí ES el render
  pre-refactor actual: no hay que volver a capturarlo.

## Contrato de idioma

El baseline está en **inglés** y así debe permanecer. La captura original
(M001/T03) usó la resolución por defecto de i18n, que en la máquina de
captura cayó en inglés; forzar el idioma del fixture divergiría del propio
baseline (`es=bb2e9f…`, `en=bf483e…`), y NO forzarlo sería dependiente de
la máquina (locale vacío o `~/.polyx_idioma.json` caería en español). El
test fija `POLYX_IDIOMA=en` explícitamente — el mismo idioma del baseline,
ahora por contrato y no por azar. Detalles en el docstring del módulo
`tests/test_render_reporte_benigno.py`.

## Contenido

- `benigna/informe.html` — render benigno congelado (la puerta de M004).
- `benigna/informe.sha256` — digest sha256 de `benigna/informe.html`
  (línea única, sin prefijo de archivo, igual que produce `sha256sum`).
- `hostil/informe.html` + `hostil/informe.sha256` — snapshot hostil
  pre-fix, usado por los tests de inyección / sumideros escapados
  (`tests/test_render_inyectado.py`,
  `tests/test_reporte_sumideros_escapados.py`).

## ⚠️ NO editar ni regenerar durante M004

El baseline está **congelado para todo M004**. No se debe volver a capturar
durante S02/S03: la captura se hace contra el código actual, así que
regenerar a mitad del refactor convertiría la puerta en un espejo del
código del día (siempre verde) y destruiría su valor como prueba de
identidad. Si un test de identidad falla, el fix está en el código que
extrajo/renderizó, no en el fixture. Ediciones a mano de `informe.html`
también quedan descartadas por `test_el_propio_baseline_coincide_con_su_
registro` (autoconsistencia sha256) y, desde M004/S01, por el pin de
tamaño `TAMANO_BASELINE`.
