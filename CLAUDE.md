# Poly-X — Suite de detección de microplásticos

**Estado:** Reconstrucción en curso (mayo 2026)  
**Autor:** Cristofher Ferrada · Dr (c) en Ciencias mención Química · PUCV  
**Versión:** 2.0.0  
**GitHub:** https://github.com/CrissFerrada/Poly-X-Microplastics

## 🎯 Propósito

Plataforma de **detección automatizada y clasificación** de microplásticos (PET, PP, LDPE) por fluorescencia Nile Red bajo luz UV (254 nm) usando modelos YOLO v8/v11 (deep learning).

Integra todo el flujo: **entrenamiento → etiquetado → detección → reporte HTML paper-quality**.

### Publicaciones

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. PLoS ONE 19(12): e0314520. https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).** Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in Merluccius gayi fish by Nile Red fluorescent staining and image analysis. J. Chil. Chem. Soc. 69(1): 6082-6085. https://doi.org/10.4067/s0717-97072024000106082

---

## 📦 Estructura

```
polyx/                    # Código fuente
├── __init__.py           # v2.0.0
├── launcher.py           # Shim Qt-free del entry point histórico
├── core/                 # Shim temporal de re-export (se elimina al cerrar M005)
├── model/                # Motor puro, sin Qt
│   ├── paths.py          # Rutas (models/, runs/, data/, assets/)
│   ├── yolo_wrap.py      # Wrapper YOLO (Detection, YoloModel)
│   ├── metrics.py        # TP/FP/FN, IoU, matriz de confusión
│   ├── morfologia.py     # Segmentación y tallas sobre la máscara
│   ├── calibracion.py    # Calibración μm/píxel
│   ├── asignacion.py     # Abstención: detectar ≠ identificar
│   ├── dataset_audit.py  # Auditoría de dataset
│   ├── observabilidad.py # Logs JSONL (logger_para, set_run_id)
│   ├── procedencia.py    # Trazabilidad de la corrida
│   ├── plataforma.py     # Diferencias entre sistemas operativos
│   ├── i18n.py           # tr(), idioma()
│   └── traducciones_en.py
└── gui/                  # Presentación: todo lo que toca Qt
    ├── __main__.py       # python -m polyx.gui
    ├── launcher.py       # Módulo 1: Menú principal (hero + 4 cards)
    ├── theme.py          # Paleta de colores (Segoe UI, Nile Red RGB)
    ├── iconos.py         # Iconos vectoriales (QPainterPath, 40 iconos)
    ├── marca.py          # Identidad por ventana (logo + nombre)
    ├── widgets.py        # LogoBadge, HLine, custom widgets
    ├── preferencias.py   # Diálogo de preferencias (tema, idioma)
    ├── pdf_export.py     # HTML → PDF
    ├── updates.py        # Buscador de actualizaciones
    ├── report_html.py    # Generador reporte (base64, paper-quality)
    ├── detector/         # Módulo 2: Análisis en lote
    │   ├── __main__.py
    │   ├── window.py         # Ventana principal (sidebar + 9 páginas)
    │   ├── state.py          # DetectorState (modelos, imágenes, params)
    │   ├── runner.py         # Lógica de ejecución (batch inference)
    │   └── pages/
    │       ├── _base.py      # BasePage (herencia)
    │       ├── modelos.py    # Carga hasta 3 modelos .pt
    │       ├── imagenes.py   # Selecciona carpeta de imágenes
    │       ├── parametros.py # Conf, IoU, um/px, device (CPU/GPU)
    │       ├── gt_manual.py  # Anota ground truth manual
    │       ├── ejecutar.py   # Botón ▶ run + progreso
    │       ├── resultados.py # Tabla: imagen, clase, conf, tamaño
    │       ├── errores.py    # Matriz confusión, galería FP/FN
    │       ├── comparar.py   # Compara runs (curvas P/R/F1)
    │       └── reporte.py    # Genera HTML + abre navegador
    ├── trainer/          # Módulo 3: Entrenamiento YOLO
    │   ├── __main__.py
    │   ├── window.py         # Ventana principal (sidebar + 9 páginas)
    │   ├── state.py          # TrainerState (config, runs, checkpoints)
    │   ├── runner.py         # Lógica de entrenamiento
    │   ├── hw.py             # Detección GPU, hardware info
    │   └── pages/
    │       ├── _base.py
    │       ├── modelo.py     # YOLO v8/v11, tamaño (nano-xlarge)
    │       ├── dataset.py    # Dataset YAML + split (train/val/test)
    │       ├── parametros.py # Epochs, batch, imgsz, optimizer
    │       ├── augmentacion.py # Flip, rotation, mosaic, etc.
    │       ├── entrenar.py   # Lanzar training + loss curves en vivo
    │       ├── evaluar.py    # Métricas mAP, P, R, F1 por clase
    │       ├── comparar.py   # Comparar múltiples runs
    │       ├── exportar.py   # Export PT→ONNX/TensorRT/CoreML
    │       └── informe.py    # Reporte entrenamiento
    ├── visor/            # Inspección con calibración μm/píxel
    └── etiquetador/      # Anotación YOLO manual

legacy/                   # Versiones anteriores, en la RAIZ (no dentro de polyx/)
├── Detector_Microplastico.py
├── trainer_microplastico.py
├── polyx_viewer.py
└── ...

_diagnostico/             # Scripts sueltos de diagnostico. No son del programa:
                          # se usan a mano para mirar un dataset o un modelo.

manual_screenshots/       # Capturas que empotra el manual
├── es/  en/                 interfaz, un idioma por carpeta
└── web/ win/                figuras de apoyo

# Cadena del manual (archivos sueltos en la raiz):
#   capturar_manual.py   capturas de la interfaz → manual_screenshots/
#   manual_texto_es.py   texto del manual, ES
#   manual_texto_en.py   texto del manual, EN
#   generar_manual.py    motor: arma Manual_PolyX[.en].html y los PDF

models/                   # Pesos YOLO .pt entrenados
├── bestdetectormedium.pt # Modelo por defecto (producción)
└── ...

runs/                     # Resultados de cada ejecución
└── YYYYMMDD_HHMMSS/
    ├── images/          # Imágenes anotadas (PNG)
    ├── centroids.csv    # class, x, y, conf, diam_um
    ├── metrics.json     # TP/FP/FN por clase
    └── annotations/     # YOLO .txt si hay GT

data_microplastico/       # Dataset etiquetado YOLO
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/           # .txt YOLO (class cx cy w h norm)
│   └── val/
└── classes.txt          # 0: PET, 1: PP, 2: LDPE

data/                     # Otros datasets
```

---

## 🔧 Tecnología

| Componente | Versión | Propósito |
|---|---|---|
| **Python** | 3.11.x | Runtime |
| **PySide6** | 6.7.2 | GUI (Qt) |
| **Ultralytics** | 8.3.40 | YOLO inference + training |
| **OpenCV** | 4.10.0.84 | Lectura/anotación imágenes |
| **NumPy** | 1.26.4 | Cálculos |
| **Pillow** | 10.4.0 | Procesamiento PNG/JPEG |
| **Matplotlib** | 3.9.2 | Gráficas (reportes) |
| **pyqtgraph** | 0.13.7 | Plots en vivo (trainer) |

**Hardware:**
- Windows 10/11 · macOS 11 Big Sur o posterior
- 8 GB RAM mínimo
- Aceleración: se elige sola en `model/plataforma.py` → **CUDA** (NVIDIA) →
  **MPS** (GPU integrada de los Mac con Apple Silicon) → **CPU**

**Portabilidad.** Todo lo que difiere entre sistemas está concentrado en
`polyx/model/plataforma.py`, no repartido por el código:

| Antes (solo Windows) | Ahora |
|---|---|
| `os.startfile` (11 sitios) | `abrir_en_el_sistema()` → `startfile` / `open` / `xdg-open` |
| `cmd /c start` | `lanzar_actualizador()` |
| `device = "0"` (CUDA fija) | `dispositivo_disponible()` |
| `actualizar.bat` fijo | `nombre_actualizador()` |

> **Mac Intel:** PyTorch dejó de publicar ruedas x86_64 para macOS desde la 2.3;
> el instalador fija **2.2.2**, la última que existe. Sin MPS (requiere Apple
> Silicon), así que corre en CPU.

> **`.gitattributes` no es cosmético:** fuerza LF en `*.command` y `*.sh`. Con
> `core.autocrlf=true` en Windows y sin él, el script llega al Mac con CRLF y
> bash lo rechaza con `/bin/bash^M: bad interpreter`, un síntoma que no se
> parece en nada a su causa. Los `.command` van con modo `100755`: sin el bit
> de ejecución el doble clic en Finder no hace nada.

---

## 🎨 Temas y paleta (polyx/gui/theme.py)

**Cuatro temas**, elegibles en *Preferencias* (o con `POLYX_TEMA`):

| Tema | Para qué |
|---|---|
| `claro` | El de siempre. Por defecto, y el que muestra el manual |
| `oscuro` | Trabajar de noche y junto a fotos de UV |
| `azul` | Azul profundo, acento más frío |
| `contraste` | AAA. Para proyector y para vista cansada |

Cada tema es un diccionario de **tokens** en `PALETAS`. Al importar el módulo se
resuelve cuál está activo y sus valores se publican como atributos del módulo,
así que el resto del programa sigue escribiendo `T.INK` sin saber que existen
los temas.

> **Por eso `from polyx.gui.theme import INK` está prohibido.** Ese import se queda
> con el color que hubiera al arrancar y no vuelve a cambiar. Hay que importar
> el módulo (`from polyx.gui import theme as T`) y leer `T.INK` en el punto de uso.

### Tokens

```python
INK / INK2 / INK3 / MUTED   # texto, de más a menos contraste
RULE / RULE_SOFT            # bordes: visible y apenas insinuado
BG                          # superficie elevada: tarjetas, barras, campos
BG_SOFT                     # fondo hundido: el lienzo de la ventana
ACCENT / ACCENT_D           # primario y su hover
ON_ACCENT                   # texto ENCIMA del acento; no siempre blanco
OK / WARN / ERR / VIO       # estado, como FONDO de botón relleno
OK_TX / WARN_TX / ...       # el mismo estado, como TEXTO sobre la tarjeta
FIELD_1 / FIELD_2 / RING    # campo del microscopio del launcher
TIP_BG / TIP_FG             # tooltip
```

**Por qué cada estado tiene dos versiones.** El tono que da 5:1 con texto blanco
encima es demasiado apagado para leerse *como* texto sobre un fondo oscuro. Un
solo valor no puede cumplir las dos cosas: elegir uno dejaba media interfaz por
debajo de 4.5:1. En los temas claros ambos coinciden. Todos los pares
texto/fondo de los cuatro temas pasan **4.5:1**.

### Lo que NO sigue al tema

- **`CLASS_COLOR_HEX`** (PET/PP/LDPE) — son datos del dominio. Si cambiaran con
  el tema, dos capturas del mismo análisis no serían comparables.
- **`T.DOC`** — paleta congelada (la clara) para los **informes HTML**. Un
  informe se imprime, se adjunta y se archiva: que saliera oscuro porque quien
  lo generó trabaja de noche gastaría tinta y haría que dos informes del mismo
  análisis no se parecieran. `report_html.py` e `informe.py` usan `T.DOC.*`.

### Cómo se aplica un cambio de tema

| Dónde | Cuándo |
|---|---|
| Launcher | **Al instante** — se reconstruye solo (`_construir_ui`) |
| Módulos ya abiertos | No cambian: son procesos aparte con el color incrustado |
| Módulos que se abran después | Ya salen con el tema nuevo |

Es el mismo contrato que el idioma. Preferencias se guardan en
`~/.polyx_tema.json` (tema + animaciones) y `~/.polyx_idioma.json`.

### Animación

`theme.animaciones()` gobierna el panel del microscopio del launcher. Apagarlo
no es solo estético: el movimiento continuo molesta a quien tiene sensibilidad
vestibular, y en un equipo sin GPU ahorra 30 cuadros por segundo. Con las
animaciones apagadas el panel dibuja un fotograma fijo, no desaparece.

### Iconos

`polyx/gui/iconos.py` dibuja **40 iconos** con `QPainterPath` sobre una rejilla de
24×24, con el trazo proporcional al tamaño pedido.

**No se usan emoji como iconos.** Un emoji lo dibuja la fuente del sistema:
cambia de forma entre Windows 10, 11 y macOS, llega en color fijo —así que no
puede teñirse con el acento del tema— y mezcla estilos, unos planos y otros con
degradado, algo que se nota en cuanto hay cuatro juntos en una rejilla.

```python
from polyx.gui import iconos
lbl.setPixmap(iconos.pixmap("detector", 26, T.ACCENT))
btn.setIcon(iconos.icono("guardar", 15, T.INK2))
```

Para un botón `checkable` —los de la barra lateral— hay
`icono_conmutable(nombre, tam, color_off, color_on)`: Qt guarda un mapa por
estado dentro del propio `QIcon`, de modo que el icono de la página activa se
tiñe de acento junto con su texto, sin escuchar el `toggled`.

Dónde se usan: barras laterales del Detector y del Entrenador
(`SIDEBAR_ITEMS`), cabeceras de página (`PAGE_ICON`), cabeceras de tarjeta
(segundo argumento de `card()`) y los botones de acción de los cuatro módulos.

> **Se conservan como texto** las marcas tipográficas que forman parte de una
> frase: `✓ ✗ ⚠ → ←`. No son iconos, son puntuación, y sustituirlas por
> pictogramas dentro de un `QLabel` obligaría a partir el texto en tres widgets.

> **Cuidado al reescribir archivos con un script.** `Path.write_text()` traduce
> `\n` a `\r\n` en Windows: sobre un archivo que ya venía con CRLF, un barrido
> línea a línea le mete una línea en blanco por cada línea y dobla el archivo.
> Hay que leer y escribir con `newline=""`.

### Valores del tema claro

```python
INK = "#1f2328"          # Texto principal
INK2 = "#424a53"         # Subtítulos
INK3 = "#656d76"         # Muted
ACCENT = "#0969da"       # Botones primarios (azul)
OK = "#1f6b5e"           # Verde (positivo)
WARN = "#9a6700"         # Naranjo (warning)
ERR = "#cf222e"          # Rojo (error)
VIO = "#6639ba"          # Púrpura

# Colores con que se dibujan las cajas de cada clase en pantalla.
# OJO: son colores de interfaz, NO describen la emisión real del Nile Red.
CLASS_COLOR_HEX = {
    "PET": "#e3342f",    # se dibuja rojo
    "PP": "#ff8c00",     # se dibuja naranjo
    "LDPE": "#ffd700",   # se dibuja amarillo
}
```

**Fuente:** Segoe UI (Windows-native)

### Fluorescencia real de cada polímero

No confundir con los colores de interfaz de arriba. Medido sobre el interior de
las cajas del dataset de entrenamiento (media RGB, n=30 por clase):

| Clase | R | G | B | Aspecto |
|---|---|---|---|---|
| PET | 116 | 58 | 65 | rojo–salmón |
| PP | 122 | **125** | 32 | **amarillo verdoso, apagado** |
| LDPE | 181 | 162 | 57 | amarillo franco, brillante |

**PP y LDPE comparten tono y se separan por brillo** (R 122 frente a 181), más el
matiz verdoso del PP. Es la confusión dominante al anotar y explica que el recall
por clase se hunda ahí: PET 0.98 · PP 0.70 · LDPE 0.54.

La documentación decía que PP era «naranjo», lo que no corresponde a la emisión
observada. Corregido en agosto de 2026 a partir de la medición y de la
observación del autor.

---

## 🚀 Instalación y uso

### 🪟 Windows

```batch
SETUP.bat            :: primera vez — crea .venv, detecta GPU, instala todo
iniciar_polyx.bat    :: uso diario
actualizar.bat       :: traer lo nuevo de GitHub
```

### 🍎 macOS

```bash
Lanzar_macOS.command       # instala la 1ª vez, arranca las siguientes
crear_icono_macOS.command  # crea Poly-X.app: abre sin ventana de Terminal
actualizar_macOS.command   # traer lo nuevo de GitHub
construir_app_macOS.command  # empaquetar Poly-X.app (opcional)
```

Clic **derecho** → Abrir la primera vez (Gatekeeper bloquea los scripts
descargados sin firma de Apple).

Instalador y lanzador van en **un solo archivo** a propósito: en macOS cada
`.command` exige su propia aprobación de seguridad la primera vez, y separarlos
obligaría a pasar dos veces por ese aviso — la segunda, con el programa ya
instalado, es cuando más parece que algo se rompió.

Abre **Launcher** → selecciona módulo.

---

## 📊 Módulos principales

### 🔬 Detector (Module 1)
**Propósito:** Análisis en lote de imágenes con modelo entrenado.

**Flujo típico:**
1. **Modelos** → Cargar hasta 3 modelos `.pt` (compara en paralelo)
2. **Imágenes** → Seleccionar carpeta (recursivo)
3. **GT manual** (opcional) → Anotar ground truth manualmente
4. **Parámetros** → conf=0.25, IoU=0.45, um/px=2.5 (calibración)
5. **Ejecutar** → Batch inference con progreso
6. **Resultados** → Tabla (clase, conf, tamaño)
7. **Errores** → Matriz confusión + galería FP/FN
8. **Comparar** → Compara runs (curvas P/R/F1)
9. **Reporte** → HTML self-contained + navegador

**Output:**
- `runs/YYYYMMDD_HHMMSS/images/*.png` — Imágenes anotadas
- `runs/.../centroids.csv` — class, x, y, conf, diam_um
- `runs/.../metrics.json` — TP/FP/FN por clase
- `runs/.../report.html` — Reporte paper-quality (base64)

### 🎯 Entrenador (Module 2)
**Propósito:** Entrenar modelos YOLO v8/v11 desde cero o fine-tuning.

**Flujo:**
1. **Modelo** → YOLO v8/v11, tamaño (nano, small, medium, large, xlarge)
2. **Dataset** → Dataset.yaml (train/val/test split)
3. **Parámetros** → Epochs, batch, imgsz, optimizer (SGD, Adam)
4. **Augmentación** → Flip, rotation, mosaic, HSV, etc.
5. **Entrenar** → Lanzar entrenamiento + loss curves en vivo
6. **Evaluar** → mAP, P, R, F1 por clase (validación)
7. **Comparar** → Compara múltiples runs
8. **Exportar** → PT → ONNX/TensorRT/CoreML
9. **Informe** → Reporte de entrenamiento

**Output:**
- `runs/detect/trainN/weights/best.pt` — Modelo óptimo
- `runs/detect/trainN/results.csv` — Métricas por época
- `runs/detect/trainN/confusion_matrix.png` — Matriz confusión

### 🏷 Etiquetador (Module 3, funcional)
Anotación YOLO manual con pre-anotación automática opcional.

**Verificado con 552 recortes** (agosto 2026). Arranque 0.25 s, interfaz
responde en 0.06 s.

**Seguimiento del avance.** La lista distingue tres estados, y se recupera del
disco al abrir la carpeta, así que el conteo puede repartirse en varias sesiones:

| Marca | Significado |
|---|---|
| `✓ nombre (n)` | revisada, con n partículas |
| `· nombre (0)` | revisada, sin partículas — **es un dato** |
| `○ nombre` | todavía sin revisar |

Un `.txt` **solo** se escribe al marcar la imagen como revisada o al dibujar una
caja. Pasar de largo no crea archivo: registrar como "revisada con cero" una
imagen apenas ojeada falsearía el censo.

**Atajos:** `Espacio` marca revisada y avanza · `Tab` salta a la siguiente sin
revisar · `F` reencuadra · `1-9` clase activa · `←/→` navegar · `Supr` borra la
caja seleccionada · `Ctrl+Z/Y` deshacer/rehacer · rueda zoom · botón medio pan.

**Lado mínimo de caja: 2 px** (`BboxCanvas.LADO_MINIMO_PX`). Estaba en 5 px y
descartaba **en silencio** marcas legítimas: las partículas más pequeñas del
estudio miden ~8 px de lado. Ahora, si una caja se rechaza, se avisa en la barra
de estado.

El zoom se conserva entre imágenes (casilla en el panel derecho), con tolerancia
de tamaño: los recortes de una rejilla difieren en 1 px por redondeo y exigir
igualdad exacta hacía perder el zoom en cada cambio.

### 📐 Visor (Module 4, funcional)
Inspección de una imagen con calibración μm/píxel, detección y exportación.

**Calibración interactiva.** Dos modos: **línea** (2 clics + longitud real) y
**círculo** (3 clics sobre un borde circular + diámetro real). El diálogo del
círculo viene con 100000 μm por defecto, que es la placa Petri del estudio.

> Estuvo inutilizable hasta agosto 2026: `QInputDialog.getDouble()` se llamaba
> con `min=`/`max=`, que PySide6 no acepta como palabras clave, y lanzaba
> `AttributeError` al completar los puntos. La matemática siempre estuvo bien;
> el diálogo reventaba antes de llegar a ella. **Los argumentos deben ir
> posicionales:** `(parent, title, label, value, minValue, maxValue, decimals)`.

**Detección.** `imgsz` configurable (320–8192, por defecto 2080) y GPU si está
disponible. Antes forzaba `device="cpu"` e `imgsz` por defecto 640: con
partículas de ~12 px en fotos de 4096 px, a 640 colapsan a ~2 px y **no se
detecta nada**. Si la GPU se queda sin memoria, se explica cómo bajar `imgsz` en
vez de mostrar la excepción cruda.

**Cargar etiquetas (.txt).** Muestra las anotaciones YOLO que acompañan a la
imagen, con las tallas ya convertidas a μm. Sirve para revisar el conteo manual
sobre la placa completa sin volver al Etiquetador.

**Exportación** a `visor_<imagen>_<ts>/`: PNG anotado, `detecciones.csv`
(una fila por partícula, con diam_px, diam_um y area_um2) y `resumen.json`.

> Lectura y escritura de imágenes con `cv2.imdecode`/`imencode` sobre
> `np.fromfile`, no `imread`/`imwrite`: en Windows estos fallan con rutas que
> llevan acentos y devuelven `None` sin avisar.

---

## 🔑 Clases clave

### `Detection` (polyx/model/yolo_wrap.py)
```python
@dataclass
class Detection:
    class_id: int           # 0=PET, 1=PP, 2=LDPE
    class_name: str         # "PET", "PP", "LDPE"
    conf: float             # 0-1, confianza
    x1, y1, x2, y2: float   # Bbox píxeles (absolutos)
    diam_um: Optional[float]  # Diámetro del círculo de igual área REAL
    area_um2: Optional[float] # Área de la partícula, no de la caja
    largo_um: Optional[float] # Dimensión mayor, siguiendo la curva
    ancho_um: Optional[float]
    aspecto: Optional[float]    # largo/ancho reales
    curvatura: Optional[float]  # 1.0 = recta; ≥1.15 = curva
    morfotipo: Optional[str]    # "fibra" | "fragmento"
```

**La caja no describe la forma.** Medido sobre 7.129 partículas anotadas: la caja
sobreestima el área **1.87×**, y una fibra tumbada en diagonal tiene caja
**cuadrada** — con lo que se reportaba como fragmento.

`polyx/model/morfologia.py` segmenta dentro de cada caja (Otsu sobre el canal de mayor
contraste, componente conexa del centro, recortada a la caja) y mide sobre la
máscara. **El largo sale de dos medidas estándar y se reporta la mayor:**

| Medida | Qué es | Falla en |
|---|---|---|
| **Feret máximo** | mayor distancia entre dos puntos del borde | fibras curvas (da la cuerda: −34.6 % en un arco de 180°) |
| **Diámetro geodésico** | camino más largo *dentro* de la partícula | partículas gruesas (rodea las concavidades) |

El geodésico solo se usa si la partícula cumple **dos** condiciones, y cada una
viene de un fallo observado:

| Condición | Constante | Por qué |
|---|---|---|
| **Delgada**, `largo/grosor ≥ 4` | `DELGADEZ_PARA_GEODESICO` | en una gruesa el camino rodea las concavidades: un grumo real de 44 px recibía 73 |
| **No convexa**, `solidez < 0.90` | `SOLIDEZ_CONVEXA` | en un convexo geodésico y Feret coinciden por definición; una barra girada medía 200 px a 0° y 45° pero **208 a 15°, 30° y 60°** |

Contra formas sintéticas de talla conocida: **error mediano 0.6 %, peor caso
4.7 %**. Fijado en `tests/test_morfologia.py`.

**El rectángulo equivalente `L = (P + √(P²−16A))/4` NO es una talla** y no debe
usarse como tal. Depende del perímetro, así que un borde dentado lo infla
(+22.5 % en una recta con dientes de sierra), y solo está definido si P² ≥ 16A
—un círculo da P² = 4πA y el discriminante sale negativo—. Se conserva en
`largo_rect_eq` como descriptor: comparado con los otros dos delata bordes
irregulares.

> **Corrección de agosto 2026.** Se reportó que el material tenía «22 % de fibras
> y aspecto hasta 21.1». Era un artefacto de usar `rect_eq` como largo. Medido
> con Feret/geodésico sobre 6.638 partículas: **aspecto mediano 1.58, máximo 8.7,
> y 1.1 % de fibras**. El material es mayoritariamente fragmentos compactos.

Coste 5.9 ms por partícula. Solo cv2 y numpy: `scipy` **no** está en
`requirements.txt` —llega de rebote con ultralytics— y apoyarse en él rompería
la instalación el día que ultralytics deje de traerlo.

### `YoloModel` (polyx/model/yolo_wrap.py)
Wrapper lazy-loading de Ultralytics YOLO:
```python
model = YoloModel("bestdetectormedium.pt")
detections = model.predict("image.jpg", conf=0.25, device="0")
```

### `DetectorState` (polyx/gui/detector/state.py)
Estado global del Detector (Qt signals):
```python
state.model_slots: List[ModelSlot]      # Hasta 3 modelos
state.images: List[Path]                # Imágenes a analizar
state.params: InferenceParams           # conf, IoU, um_per_px
state.results: Dict[int, List[ImageResult]]  # Resultados por modelo
```

### `ImageResult` (polyx/gui/detector/state.py)
Resultado de una imagen:
```python
image_path: Path
predictions: List[Detection]
gt: List[Detection]             # Ground truth (si existe)
tp, fp, fn, miscls: int
annotated_png: Optional[bytes]  # PNG anotado en base64
```

### `MatchResult` (polyx/model/metrics.py)
Matching pred vs GT (IoU ≥ 0.5):
```python
tp: int          # True Positives (conf y clase correcta)
fp: int          # False Positives (sin GT)
fn: int          # False Negatives (GT sin pred)
miscls: int      # Bien localizado pero clase incorrecta
```

---

## 🔄 Flujo de datos

### Detector
```
Imágenes (carpeta)
    ↓
YoloModel.predict() × N modelos
    ↓
Detection[] (bbox + clase + conf)
    ↓
read_yolo_txt() → Ground truth (si existe)
    ↓
match_image() → TP/FP/FN (IoU ≥ 0.5)
    ↓
compute_box_size_um() → diam_um (si um_per_px calibrado)
    ↓
ImageResult
    ↓
cv2.rectangle() + anotación → PNG
    ↓
runs/YYYYMMDD_HHMMSS/
```

### Trainer
```
Dataset (images/ + labels/ YOLO)
    ↓
YOLO(model_name).train(epochs=..., batch=...)
    ↓
Ultralytics training loop
    ↓
best.pt (checkpoint óptimo)
    ↓
results.csv + curves
    ↓
runs/detect/trainN/
```

---

## 📝 Convenciones

- **Español:** Todos los comentarios, variables locales, UI en español
- **Inglés:** Docstrings en inglés (excepto clase/método si es muy específico)
- **Qt Signals:** `models_changed`, `run_started`, `run_finished`, etc.
- **Rutas:** Siempre `Path` de pathlib, nunca strings
- **Colores:** Usar `T.ACCENT`, `T.OK`, etc., nunca hardcodes. Como
  **texto**, la variante `_TX` (`T.OK_TX`); como **fondo** de botón
  relleno, la base. En informes HTML, siempre `T.DOC.*`
- **Métodos:** `_` privado (widget, signal callback)

---

## 🔗 Dependencias clave

- **Ultralytics YOLO** → Inference, training, export
- **OpenCV** → `cv2.rectangle()`, anotación, lectura imágenes
- **NumPy** → Cálculos de bbox, IoU
- **Matplotlib** → Gráficas en reportes (en modo Agg, no interactivo)
- **PySide6/Qt** → GUI (signals/slots, QObject)

---

## ⚙️ Parámetros comunes

### Inferencia (DetectorState.params)
```python
conf: float = 0.25          # Umbral confianza
iou_nms: float = 0.45       # IoU para NMS
iou_tp: float = 0.50        # IoU para análisis de errores
imgsz: int = 640            # Tamaño entrada red
device: str = "0"           # "0" (GPU), "cpu"
um_per_px: float = 0.0      # Calibración (0 = sin calibrar)
size_min_um: float = 0.0    # Filtro inferior (0 = sin filtro)
size_max_um: float = 0.0    # Filtro superior
conf_asignacion: float = 0.0  # Confianza mínima para atribuir polímero (0 = off)
```

### Abstención: detectar ≠ identificar

`polyx/model/asignacion.py`. Por debajo de `conf_asignacion` la partícula **sigue
contando** como partícula detectada, pero se reporta como **«no asignable»** en
vez de atribuirle un polímero. El informe declara en Métodos qué porcentaje del
lote quedó sin asignar.

El motivo es químico, no de ingeniería: el Nile Red es **solvatocrómico** y su
emisión responde a la **polaridad** del entorno, no a la identidad del polímero.
El PET —poliéster— se separa limpio (recall 0.98); el PP y el LDPE son las dos
poliolefinas apolares, comparten tono y solo difieren en brillo, que depende de
exposición, foco y espesor. De ahí que su recall se hunda a 0.70 y 0.54.

**No es una clase del modelo, y no debe serlo.** Una clase `desconocido`
entrenada se define por la duda de quien anota y no por lo que se ve: la misma
partícula acabaría LDPE un día y desconocido otro, y ese ruido caería justo
sobre las dos clases ya más débiles.

**Tampoco usa el margen entre las dos clases más probables**, que sería mejor
criterio: tras el NMS, YOLO devuelve una sola confianza y una sola clase por
caja, y las puntuaciones del resto se pierden ahí. Recuperarlas obliga a
interceptar el tensor crudo de la cabeza, frágil entre versiones de ultralytics.

### Entrenamiento
```python
epochs: int = 100
batch: int = 16             # Menor si OOM
imgsz: int = 640            # Menor si OOM
device: str = "0"
optimizer: str = "SGD"      # o "Adam"
lr0: float = 0.01
```

---

## 🐛 Troubleshooting

| Problema | Solución |
|---|---|
| "No module named tkinter" | Reinstalar Python, marcar tcl/tk |
| "CUDA not available" | Normal, solo CPU funciona (más lento) |
| Detector no detecta nada | Bajar conf a 0.10 en Parámetros |
| OOM en entrenamiento | Bajar batch a 4-8, imgsz a 512-576 |
| Reporte no abre | Revisar Manual_PolyX.html manualmente |

---

## 📚 Archivos generados automáticamente

- **Manual_PolyX.html / Manual_PolyX.en.html** (+ sus PDF) — los arma
  `generar_manual.py` desde cero: texto, figuras, índice y hoja de estilo.
- **manual_texto_es.py / manual_texto_en.py** — el texto del manual. Solo
  contenido; el motor está en `generar_manual.py`.
- **manual_screenshots/** — capturas que empotra el manual, una carpeta por
  origen: `es/` y `en/` (interfaz, un idioma cada una), más `web/` y `win/`.

Para regenerar (capturas primero, manual después):
```bash
.venv\Scripts\python.exe capturar_manual.py --salida manual_screenshots/es
set POLYX_IDIOMA=en && .venv\Scripts\python.exe capturar_manual.py --salida manual_screenshots/en
.venv\Scripts\python.exe generar_manual.py --pdf
```

> El generador anterior reemplazaba figuras *por posición* dentro de un HTML ya
> escrito. Se retiró en septiembre de 2026 —queda en el historial de git— porque
> con el manual actual descolocaría las imágenes sin avisar.

---

## 🔮 Próximas mejoras (en construcción)

- ✅ Detector: Funcional, reporte HTML
- ✅ Entrenador: Funcional, curvas en vivo
- ✅ Etiquetador: Funcional, verificado con 552 recortes (ago 2026)
- ✅ Visor: Funcional, calibración reparada y verificada (ago 2026)
- 📋 Validación de dataset (duplicados, outliers)
- 🤖 Recomendaciones automáticas de parámetros

---

## 📞 Contacto

**Cristofher Ferrada**  
Dr (c) en Ciencias mención Química — Pontificia Universidad Católica de Valparaíso  
2026

**Repo:** https://github.com/CrissFerrada/Poly-X-Microplastics
