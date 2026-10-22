# Fixtures geodesia: mascaras reales para el harness de equivalencia

Este directorio alberga mascaras **reales** (uint8, un canal, 0/255) serializadas
desde corridas del detector, para que `tests/test_geodesia.py` las recubra a
todas automaticamente y compare la propagacion geodesica pre/post reescritura
bit a bit sobre ellas.

## Que va aqui

Cada fichero es `<alias>__<stem>__<idx>.npy`, una mascara uint8 recortada al
rectangulo de la particula, tal cual la devuelve `morfologia.segment()` sobre
la foto original de una deteccion anotada en `labels/NNNN_<stem>.txt`.

## Corpus canónico versionado

El repositorio incluye un corpus canónico de 8 máscaras `.npy` representativas
de las principales morfologías de microplásticos observadas en laboratorio,
abarcando fragmentos compactos, rugosos, cóncavos, fibras y partículas
ramificadas:

| Archivo | Morfología / Clase | Lado (recorte) | Px encendidos |
|---|---|---|---|
| `y11n__fragmento_circular__000.npy` | Fragmento circular compacto | 39x39 | 1253 |
| `y11n__fragmento_irregular__000.npy` | Fragmento irregular cóncavo | 55x56 | 2528 |
| `y11n__fragmento_rugoso__000.npy` | Fragmento rugoso lobulado | 53x53 | 1626 |
| `y11n__fibra_recta__000.npy` | Fibra delgada recta horizontal | 5x93 | 461 |
| `y11n__fibra_curva__000.npy` | Fibra en arco curvado | 36x75 | 456 |
| `y11n__fibra_sinuosa__000.npy` | Fibra sinuosa de grosor variable | 40x103 | 898 |
| `y11n__particula_alargada_diagonal__000.npy` | Partícula alargada diagonal a 45° | 87x87 | 757 |
| `y11n__particula_bifurcada__000.npy` | Partícula ramificada en horquilla | 77x69 | 748 |

### Contrato y huella en disco
- Formato: `numpy.ndarray` 2D, `dtype=uint8`, valores estrictos `{0, 255}`.
- Recorte: ajustado al bounding box mínimo de los píxeles encendidos (`mascara > 0`).
- Huella total: ~28 KB en disco (28.601 bytes para los 8 archivos), eliminando la
  necesidad de versionar imágenes crudas de placas completas (~MBs).

### Procedencia y reproducibilidad
Generados mediante `tests/generar_fixtures_geodesia.py`:
1. Crea partículas con morfologías controladas sobre lienzos sintéticos.
2. Emite la estructura mock esperada por `polyx/detector/runner.py` (`runs/y11n/labels/` con cajas YOLO normalizadas).
3. Invoca la función canónica de producción `capturar_fixtures_geodesia.capturar()`, la cual ejecuta `morfologia.segment()`, recorta al rectángulo envolvente y serializa con `np.save(..., allow_pickle=False)`.

Para regenerar el corpus canónico en cualquier momento:
```
python tests/generar_fixtures_geodesia.py
```

### Cobertura de pruebas
Con estos fixtures presentes, `tests/test_geodesia.py::test_identidad_sobre_fixtures_reales`
descubre automáticamente cada `.npy`, verifica que cumpla el contrato de tipo/valores,
y prueba la equivalencia bit a bit contra `_propagar_geodesico_referencia` con 0 skips.

## Como capturar desde corridas reales (receta para operador)

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
por stem (busqueda recursiva bajo `--fotos`, insensible a mayusculas, match exacto primero y luego el stem mas corto que lo contenga), lee la foto con
`np.fromfile` + `cv2.imdecode` (nunca `cv2.imread`: con rutas acentuadas falla
en silencio en Windows) y reproduce la mascara con `morfologia.segment()`
exactamente como lo hace `medir_deteccion()` en el pipeline. Serializa un
`.npy` por caja, recortado al rectangulo de lo encendido. Una caja cuya mascara
salga vacia o `None`, o cuyo foto no se encuentre, se SALTA y queda contada en
el resumen final (no se interrumpe el lote por una caja mala).

Sin `--reforzar` nunca sobreescribe un fixture existente cuyo contenido difiera
(si coincide, la pasada es un no-op); con `--reforzar` si lo pisa. La logica de
captura esta probada en `tests/test_captura_geodesia.py` contra un layout de
`runs/detect_*` en miniatura, sin depender de datos reales.

## TIMING.md

El benchmark de referencia-vs-actual (`tests/bench_geodesia.py`) mide el corpus
completo bajo la clase `fixture real` junto con las categorías procedimentales,
de formas conocidas y escalas, publicando tiempos individuales, medianas y
ratios en `TIMING.md`.
