# Fixtures geodesia: mascaras reales para el harness de equivalencia

Este directorio alberga mascaras **reales** (uint8, un canal, 0/255) serializadas
desde corridas del detector, para que `tests/test_geodesia.py` las recubra a
todas automaticamente y compare la propagacion geodesica pre/post reescritura
bit a bit sobre ellas.

## Que va aqui

Cada fichero es `<alias>__<stem>__<idx>.npy`, una mascara uint8 recortada al
rectangulo de la particula, tal cual la devuelve `morfologia.segmentar()` sobre
la foto original de una deteccion anotada en `labels/NNNN_<stem>.txt`.

## Como capturar (receta)

En el equipo que tenga los datos `runs/` (esta copia de trabajo no los tiene):

```
python tests/capturar_fixtures_geodesia.py \
    --runs runs/detect_20260812_1030 \
    --fotos F:/Poly-X/fotos_origen \
    [--destino tests/fixtures/geodesia] \
    [--reforzar]
```

El script lee el layout exacto que escribe `polyx/detector/runner.py`
(`labels/NNNN_imgstem.txt` en formato YOLO normalizado), localiza la foto origen
por stem (busqueda recursiva bajo `--fotos`, insensible a mayusculas, match
e exacto primero y luego el stem mas corto que lo contenga), lee la foto con
`np.fromfile` + `cv2.imdecode` (nunca `cv2.imread`: con rutas acentuadas falla
en silencio en Windows) y reproduce la mascara con `morfologia.segmentar()`
exactamente como lo hace `medir_deteccion()` en el pipeline. Serializa un
`.npy` por caja, recortado al rectangulo de lo encendido. Una caja cuya mascara
salga vacia o `None`, o cuyo foto no se encuentre, se SALTA y queda contada en
el resumen final (no se interrumpe el lote por una caja mala).

Sin `--reforzar` nunca sobreescribe un fixture existente cuyo contenido difiera
(si coincide, la pasada es un no-op); con `--reforzar` si lo pisa. La logica de
captura esta probada en `tests/test_captura_geodesia.py` contra un layout de
`runs/detect_*` en miniatura, sin depender de datos reales.

## Estado actual del worktree

Sin `.npy` capturados: el test `test_identidad_sobre_fixtures_reales` se salta
honestamente con ese mensaje. La cobertura sin fixtures recae en los generadores
procedurales de `tests/test_geodesia.py` y en las figuras de `tests/formas.py`,
que son clases de forma de talla conocida.

## TIMING.md

El benchmark de referencia-vs-actual (`tests/bench_geodesia.py`) escribe su
salida en `TIMING.md` aqui mismo, listando las `(alias, stem, idx)` reales que
formen parte del corpus.
