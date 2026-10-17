# Poly-X — Suite de detección de microplásticos

***Español** · [English](README.en.md)*

> **Detección automatizada y clasificación de microplásticos (PET, PP, LDPE)  
> por fluorescencia Nile Red bajo luz UV (254 nm) con modelos YOLO v8/v11.**

**Autor:** Cristofher Ferrada · Dr (c) en Ciencias mención Química · PUCV · 2026  
**Versión:** 2.0.0 · **Windows 10/11 y macOS** · Python 3.9+

---

## Módulos

| # | Módulo | Estado | Descripción |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Funcional | Menú principal — lanza cada módulo en ventana independiente |
| 2 | 🔬 **Detector** | ✅ Funcional | Análisis en lote con YOLO · reporte HTML paper-quality |
| 3 | 🎯 **Entrenador** | ✅ Funcional | Entrenamiento YOLO v8/v11 con curvas en vivo |
| 4 | 🏷 **Etiquetador** | ✅ Funcional | Anotación YOLO interactiva con pre-anotación automática |
| 5 | 📐 **Visor** | ✅ Funcional | Inspección de imágenes con calibración interactiva μm/píxel |

---

## Características

### 🔬 Detector (análisis en lote)
- Carga hasta **3 modelos .pt** en paralelo para comparación directa
- Inferencia sobre carpetas de imágenes con barra de progreso
- Análisis de errores: **Verdaderos Positivos / Falsos Positivos / Falsos Negativos / Mal Clasificados**, matriz de confusión, galería de falsos positivos/negativos
- **Calibración automática contra la placa Petri**: el borde se encuentra solo y
  su diámetro conocido fija los μm/px **de cada foto**, sin marcar nada. Importa
  porque la distancia de disparo varía entre tomas: en el material del estudio la
  escala real va de 31 a 50 μm/px, un factor 1.6, y un valor único para todo el
  lote daría tallas con hasta un 50 % de error
- **Talla y forma medidas sobre la partícula, no sobre su caja**: largo, ancho,
  área, relación de aspecto y clasificación **fibra / partícula**. La caja de una
  partícula alargada está casi vacía y depende de cómo haya caído — una fibra
  tumbada en diagonal tiene caja cuadrada — así que sobre 7.129 partículas
  anotadas sobreestimaba el área **1,87×**
- **Histograma de distribución de tamaños** por clase (PET/PP/LDPE) y por tramos
  de talla, apilado por polímero
- **Exportar CSV** directo desde la página de resultados
- **Drag & Drop** de modelos `.pt` e imágenes/carpetas
- **Resolución de inferencia** ajustable con perfiles (Rápido 1280 · Equilibrado
  2560 · Máxima detección 4096) y un botón que mide el máximo que aguanta tu GPU
- Reporte HTML autocontenido (imágenes incrustadas en **base64**, calidad paper)
  con **galería comparativa Predicción vs Ground Truth (lado a lado)**. Las
  imágenes de galería se recodifican y se limita su número para que el archivo
  siga siendo abrible en un navegador; las métricas cubren todas las imágenes
- **Exportación a PDF** del reporte (un clic), listo para enviar por correo
- **Secciones del informe elegibles**: trece casillas y tres presets (Completo ·
  Resumen breve · Metodológico). Al desmarcar, las secciones se **renumeran solas**
  y el índice se ajusta; una sección marcada sin datos se omite igualmente
- **Sección de calibración**: de dónde salió la escala de cada foto, su mínimo /
  mediana / máximo, la **media con intervalo de confianza al 95 %** y una figura
  sobre una placa real con el círculo ajustado y su diámetro dibujados encima
- **Talla por carpeta y por foto** (opcional): compara la distribución de tallas
  entre carpetas —cada carpeta como sitio de muestreo, estación o condición— con
  diagramas de caja y una prueba de **Kruskal-Wallis**
- **Perfil en profundidad del testigo** (opcional): cuando las fotos se llaman
  `tramo.testigo`, el tramo deja de ser una carpeta cualquiera y pasa a ser una
  variable **ordenada**. El informe dibuja el perfil con la profundidad en el eje
  vertical —la convención de cualquier testigo de sedimento— y responde si el
  **número de partículas** y la **talla mediana** crecen o decrecen con la
  profundidad, con **Spearman** y un valor de *p* por **permutación**. La
  correlación va sobre los tramos y no sobre las partículas: dos partículas de la
  misma placa no son observaciones independientes de la profundidad
- **Ficha de partículas medidas**: las **6 fibras y las 6 partículas mayores**,
  cada una con su recorte al lado y la medida dibujada encima (Feret en amarillo,
  geodésico en magenta, máscara en verde). El reparto es deliberado: las fibras
  son minoría y son justo donde actúa el método geodésico
- **El informe sale en el idioma de la aplicación**, español o inglés: títulos,
  tablas, pies de figura y la prosa de métodos
- **Alcance del informe elegible**: trabajo completo, solo las fotos que marques, o
  ambos de una vez. Las cifras, los gráficos y la matriz de confusión se recalculan
  sobre lo elegido, así que el informe siempre describe las fotos que muestra
- **Comparación real entre modelos** dentro del informe: tabla foto por foto con las
  detecciones de cada uno y, si hay ground truth, sus TP/FP/FN y el F1 global
- **Troceado automático de fotos grandes**: por encima del umbral la foto se analiza
  en recortes solapados, porque a resolución completa las partículas caen por debajo
  del stride de la red y desaparecen. Las cajas vuelven a coordenadas de la foto
  original y los solapes se fusionan con NMS, de modo que **los resultados y el
  informe se entregan siempre sobre la foto completa, nunca sobre los recortes**
- Aviso previo en la pestaña Ejecutar de si el lote se va a trocear y cuánto tardará

### 🎯 Entrenador
- Soporte para **YOLO v8 y v11**, tamaños nano → xlarge
- Carga de `data.yaml` con auto-detección de splits train/val/test
- **Validación del dataset** con indicadores ✓/✗ (imágenes, labels, clases)
- Curvas de pérdida en vivo (Box loss, mAP, Precision, Recall)
- Augmentación configurable (flip, rotation, mosaic, HSV)
- Exportación a **ONNX / TensorRT / CoreML**
- Diagnóstico explícito de GPU/CPU al iniciar el entrenamiento
- **Entrenar v8 y v11 con la misma configuración** desde una casilla: ambas corridas
  reutilizan idénticos `imgsz`, `batch`, épocas, semilla y augmentación, que es lo que
  permite atribuir la diferencia de métricas a la arquitectura y no a los
  hiperparámetros. Van en secuencia porque comparten GPU, y al terminar el log resume
  la comparación
- **Auditoría del dataset** antes de entrenar: distribución de clases, formato y
  validación, con veredicto de si es apto
- **`best_real.pt`: el peso que sirve para tus fotos.** Ultralytics guarda
  `best.pt` según el mAP de la validación completa, y en un dataset mixto ese
  número lo dominan las placas dopadas de laboratorio (en el dataset del Loa,
  1191 cajas de laboratorio frente a 47 de sedimento real). Al terminar, Poly-X
  evalúa todos los checkpoints contra **solo** el sedimento real y guarda el
  ganador aparte. `best.pt` se conserva intacto

### 🏷 Etiquetador
- Canvas interactivo: **arrastrar** para dibujar cajas YOLO normalizadas
- Clic derecho para asignar clase; teclas `1`–`9` para cambiar clase activa
- **Undo/Redo** (Ctrl+Z / Ctrl+Y) por imagen
- **Seguimiento del avance**, pensado para campañas de cientos de imágenes
  repartidas en varias sesiones. El estado se recupera del disco al abrir la
  carpeta y la lista distingue tres casos:

  | Marca | Significado |
  |---|---|
  | `✓ nombre (n)` | revisada, con n objetos |
  | `· nombre (0)` | revisada, sin objetos — **es un dato** |
  | `○ nombre` | todavía sin revisar |

  Un `.txt` solo se escribe al marcar la imagen como revisada o al dibujar una
  caja: pasar de largo no crea archivo, porque registrar como «revisada con
  cero» una imagen apenas ojeada falsearía un conteo censal.
- `Espacio` marca revisada y avanza · `Tab` salta a la siguiente sin revisar ·
  `F` reencuadra · `←/→` navega · `Supr` borra la caja seleccionada
- **El zoom se conserva** entre imágenes (casilla en el panel derecho)
- **Lado mínimo de caja: 2 px.** Al rechazar una caja se avisa en la barra de
  estado, en vez de descartarla en silencio
- **Pre-anotación automática** con modelo `.pt`, con `conf` e `imgsz` ajustables
  y GPU si está disponible
- **Auto-guardado silencioso** cada 60 segundos; miniaturas generadas en segundo
  plano para que la ventana responda de inmediato
- Guarda `.txt` YOLO junto a la imagen (o en `labels/` si está en `images/`) y
  genera `classes.txt` en la raíz y en cada subcarpeta

### 📐 Visor
- Abre una imagen individual o navega una carpeta con `← →`
- **Calibración interactiva μm/píxel** en 2 modos:
  - 📏 **Línea** (2 clics): marca una referencia conocida, ingresa su tamaño real
  - ⭕ **Círculo** (3 clics): marca 3 puntos del borde, ingresa el diámetro real
    (útil con placas Petri, cuyo diámetro es conocido)
- Barra inferior muestra `📐 0.4880 μm/px (línea)` en tiempo real
- Detección con modelo `.pt` cargado, con **resolución de inferencia
  configurable** (320–8192) y GPU si está disponible. Con objetos diminutos en
  fotos de alta resolución el `imgsz` es determinante: a valores bajos las
  partículas caen por debajo del *stride* de la red y no se detecta nada
- **Cargar etiquetas `.txt`**: muestra anotaciones ya existentes sobre la imagen,
  con sus tallas convertidas a μm. Sirve para revisar un conteo manual sin
  volver al Etiquetador
- **Revisión partícula a partícula**: la tabla lista cada partícula con su
  **número**, clase, tipo (fibra o fragmento), largo, ancho y aspecto. Al
  seleccionar una fila se ve **sobre qué se midió**: el recorte sin marcas a la
  izquierda y, a la derecha, el contorno de la máscara con la medida dibujada
  encima — en amarillo la recta de Feret, en magenta el camino geodésico —, más
  la cuenta completa de píxeles a micrómetros. Una talla que no se puede ver
  medida no se puede verificar
- **Cargar predicciones de una corrida** ya cerrada desde `runs/detect_.../`, sin
  volver a pasar el modelo. Se abre siempre la foto original y nunca el PNG
  anotado, que lleva las cajas pintadas encima
- **Drag & Drop** de imagen o modelo `.pt` directo al canvas
- Exporta: imagen anotada + `detecciones.csv` + `resumen.json`

---

## Cómo se mide la talla de una partícula

El criterio general es **la línea recta más larga que cabe en la partícula**, es
decir la mayor distancia entre dos puntos de su contorno: el *diámetro de Feret
máximo*. No depende de la orientación con que la partícula haya caído y un borde
dentado no la altera.

Esa recta deja de servir cuando la partícula está **contorsionada**: en una fibra
doblada la distancia entre extremos es la cuerda, y en un arco de media
circunferencia se queda un 35 % corta. Para esos casos se mide el *diámetro
geodésico*, el camino más largo que cabe **dentro** de la partícula, que al no
poder salirse de la máscara rodea la curva.

| Forma de la partícula | Qué se reporta como largo |
|---|---|
| Compacta o irregular, pero no doblada | Feret máximo — la recta más larga |
| Alargada y contorsionada (fibra) | Diámetro geodésico — sigue la curva |

El geodésico solo se aplica si la partícula es **delgada** (largo ≥ 4 × grosor) y
**no convexa** (solidez < 0,90). Sin la primera, cualquier concavidad hace que el
camino rodee la partícula en vez de atravesarla; sin la segunda, el largo pasaría
a depender del ángulo de giro, que es justo el defecto que se quería eliminar.

Contra formas sintéticas de talla conocida —rectas, rectas giradas, arcos de 60,
120 y 180°, un círculo, una recta de borde dentado y un grumo con muesca— el
largo así medido da **0,6 % de error mediano y 4,7 % en el peor caso**. Está
fijado en `tests/test_morfologia.py`.

> **El rectángulo equivalente no es una talla.** La fórmula
> *L* = (*P* + √(*P*²−16*A*))/4 da el largo de un rectángulo con la misma área y
> el mismo perímetro, que es otra cosa. Depende del perímetro, así que un borde
> dentado la infla un 22,5 %, y no está definida para partículas compactas, en las
> que *P*² < 16*A*. Se reporta como descriptor porque comparada con las otras dos
> delata bordes irregulares, pero no se usa como talla.

### De píxeles a micrómetros: la escala

Todo lo anterior se mide en **píxeles**. La conversión a micrómetros no es un
factor único para el lote: depende de la distancia de disparo, y en el material de
este estudio la escala real va de 31 a 50 µm/px, un factor 1,6. Por eso **cada
foto se calibra con la suya**, contra el anillo de la placa Petri: se localiza el
centro aproximado con Hough, se muestrea el borde en **720 direcciones** y se
ajusta una circunferencia por mínimos cuadrados con rechazo de atípicos. El radio
de Hough no se usa, porque llega a errar un 12 % y ese error entraría entero en
todos los tamaños.

> **Qué borde son los 100 mm, y hacia dónde puede fallar.** El anillo tiene una
> pared de unos **2 mm** — medido sobre las fotos del estudio: el borde interno
> cae en 0,960 del radio ajustado y el externo en 1,000. El diámetro nominal de
> una placa Petri es ambiguo a ese nivel: puede referirse al **externo** o al
> **útil interior**. Aquí se toma el **externo**, que es el borde al que ajusta el
> círculo. Si el nominal se refiriera al interior, la escala correcta sería un
> **4,2 % mayor** y todas las tallas estarían **subestimadas** en esa cifra. El
> sesgo solo puede ir en ese sentido, porque el externo es el mayor de los dos
> bordes posibles. Queda declarado en el propio informe.

**Partículas en contacto.** Dos partículas que se tocan forman una sola mancha,
y medirlas juntas sumaría sus tallas. Se separan por *watershed* sobre la
transformada de distancia: el centro de cada una queda lejos del fondo y el cuello
que las une queda cerca, de modo que el corte cae por el cuello. Sobre círculos de
talla conocida las separa hasta un **27 % de solapamiento del diámetro**, sin
partir ninguna partícula de una sola pieza.

**La fibra se mide entera.** La máscara no se recorta a la caja del detector
cuando la partícula es alargada, ni se pasa por el separador de partículas
pegadas: las dos cosas cortaban fibras reales. En la que lo destapó se perdía un
53 % del largo — 369 px de componente conexa quedaban en 174 —, y una talla
subestimada no se nota en las cifras, solo en la imagen. Está fijado en
`tests/test_fibra_no_se_trunca.py`.

**Limitaciones declaradas.** Dos partículas solapadas más allá de un 40 % de su
diámetro se siguen midiendo como una sola: a esa altura ya no hay un cuello por el
que cortar. Y en una fibra muy enroscada el camino geodésico ataja por el interior
de cada codo, subestimando hasta un 19 % en el caso más cerrado ensayado.

---

## Polímeros detectados

| ID | Clase | Fluorescencia observada (Nile Red, UV) | Color en la interfaz |
|---|---|---|---|
| 0 | **PET** | Rojo–salmón | 🔴 `#e3342f` |
| 1 | **PP** | **Amarillo verdoso, apagado** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Amarillo franco y **más brillante** | 🟡 `#ffd700` |

Los colores de la derecha son solo los de las cajas en pantalla; no describen la
emisión real. Medido sobre el interior de las cajas del dataset de entrenamiento
(media RGB, n=30 por clase):

| Clase | R | G | B |
|---|---|---|---|
| PET | 116 | 58 | 65 |
| PP | 122 | **125** | 32 |
| LDPE | 181 | 162 | 57 |

**PP y LDPE no se distinguen por tono sino por brillo**: ambos son amarillentos,
pero en PP el verde iguala o supera al rojo y la emisión es bastante más apagada.
Es la confusión más habitual al anotar, y la razón de que el recall por clase caiga
en esas dos.

---

## Idioma

La interfaz está en **español e inglés**. El selector está en la barra superior
del Launcher y la elección se recuerda entre sesiones. La primera vez se toma el
idioma del sistema; `POLYX_IDIOMA=en` lo fuerza sin tocar la interfaz.

Los módulos son procesos aparte y leen el idioma al abrirse, así que el cambio
surte efecto en cuanto abras el siguiente módulo.

**El informe de detección sale en el idioma elegido**, no solo la interfaz:
títulos, tablas, pies de figura, ejes de los gráficos y la prosa de métodos. El
atributo `lang` del HTML se ajusta también, para que el corrector del navegador y
los lectores de pantalla lo traten bien.

Para ver qué falta por traducir:

```bat
.venv\Scripts\python.exe auditar_traduccion.py
```

## Temas

Cuatro temas, en **Preferencias** (botón de la barra superior, o `Ctrl+,`):

| Tema | Para qué |
|---|---|
| **Claro** | El de siempre. Por defecto |
| **Oscuro** | Trabajar de noche, y junto a fotografías de UV |
| **Azul nocturno** | Azul profundo, con el acento más frío |
| **Alto contraste** | Cumple AAA. Para proyector y para vista cansada |

El tema se aplica **al instante en el launcher**, que se reconstruye solo. Los
módulos ya abiertos conservan el suyo —son procesos aparte, con los colores
incrustados en su hoja de estilo— y los que abras después ya salen con el nuevo.
Es el mismo contrato que el idioma.

`POLYX_TEMA=oscuro` lo fuerza sin tocar la interfaz.

> **Los informes HTML no siguen el tema, a propósito.** Un informe se imprime,
> se adjunta a un correo y se archiva: que saliera oscuro porque quien lo generó
> prefiere trabajar de noche gastaría tinta y haría que dos informes del mismo
> análisis no se parecieran.

Todos los pares de texto y fondo de los cuatro temas pasan el **4.5:1** que pide
la WCAG para texto normal.

### Movimiento

El panel del microscopio del launcher se anima: las partículas derivan y un
barrido las va marcando con sus cajas de detección, que es lo que hace el
programa. En **Preferencias → Movimiento** se puede apagar —el panel queda en un
fotograma fijo— si el movimiento molesta o si el equipo va justo.
`POLYX_SIN_ANIMACION=1` hace lo mismo desde el entorno.

---

## Requisitos

| Componente | Windows | macOS |
|---|---|---|
| Sistema | 10 / 11 | 11 Big Sur o posterior |
| Python | **3.11.x** (no 3.12+) | 3.9 o superior |
| RAM | 8 GB mínimo | 8 GB mínimo |
| Aceleración | GPU NVIDIA opcional (20–30× más rápido para entrenar). El instalador elige la versión de CUDA según la tarjeta | **Apple Silicon:** GPU integrada vía MPS. **Intel:** solo CPU |

Dependencias principales: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`, `Matplotlib 3.9`

> **Mac Intel:** PyTorch dejó de publicar versiones para procesadores Intel a partir
> de la 2.3, así que el instalador fija la **2.2.2**, la última con soporte x86_64.
> Funciona correctamente, pero sin aceleración por GPU: cuenta ~1 minuto por foto
> en lotes con troceo.

---

## Instalación

Descarga el proyecto desde GitHub (botón **Code → Download ZIP**, y descomprimir;
o `git clone`) y sigue las instrucciones de tu sistema.

### 🪟 Windows

Doble clic en:

```bat
SETUP.bat
```

El instalador:

1. **Pregunta dónde instalar Poly-X** (pulsa ENTER para instalarlo en la misma carpeta, o escribe otra ruta).
2. Detecta Python 3.11 y crea el entorno `.venv`.
3. Detecta la GPU y descarga PyTorch con **la versión de CUDA que corresponde a esa
   tarjeta**. Las RTX 50xx (Blackwell, `sm_120`) necesitan CUDA 12.8: con las ruedas
   de CUDA 11.8 `torch.cuda.is_available()` devuelve `True` y el fallo aparece recién
   al entrenar, con *"no kernel image is available for execution on the device"*.
4. Instala el resto de dependencias, verifica que todo importa y **comprueba que
   PyTorch traiga kernels para tu GPU concreta**.
5. Registra la versión instalada, para que el launcher pueda avisarte de las nuevas.
6. **Busca instalaciones anteriores de Poly-X** en el equipo. Si encuentra alguna,
   muestra qué datos tiene, los migra a la nueva y manda la vieja a la **papelera**
   (ver más abajo).
7. Te ofrece **crear un acceso directo "Poly-X" en el Escritorio** y **te dice cómo iniciarlo**.

### 🍎 macOS

**Clic derecho** sobre `Lanzar_macOS.command` → **Abrir** → **Abrir**.

Ese archivo hace las dos cosas: la primera vez instala (10–15 min) y después
arranca. No hay un instalador aparte a propósito — en macOS cada `.command`
necesita su propia aprobación de seguridad la primera vez, y con un solo archivo
eso ocurre una sola vez.

El instalador detecta si el Mac es Apple Silicon o Intel, elige el PyTorch que
corresponde, comprueba que todo importe y te ofrece crear un acceso directo en el
Escritorio.

> #### ⚠️ «No se puede abrir porque es de un desarrollador no identificado»
>
> **Es lo esperable y no significa que esté roto.** macOS bloquea por defecto
> cualquier script descargado que no venga firmado con una cuenta de desarrollador
> de Apple (USD 99/año).
>
> **Solución:** clic **derecho** → **Abrir** → confirmar **Abrir**. Solo la primera
> vez; después el doble clic normal funciona. Si se resiste, en Terminal:
> `xattr -d com.apple.quarantine Lanzar_macOS.command`

Detalle completo en **[LEEME_macOS.md](LEEME_macOS.md)**.

---

### Abrir sin consola (Windows)

Doble clic en **`Poly-X.vbs`** (raíz del proyecto): abre el Launcher directamente,
sin ventana negra. Resuelve su propia ubicación, así que funciona desde cualquier
carpeta — basta descargar el repositorio y hacer doble clic, sin editar rutas.

Para tenerlo a mano, clic derecho sobre `Poly-X.vbs` → *Enviar a* → *Escritorio
(crear acceso directo)*, y asígnale el icono `assets\polyx.ico`.

`iniciar_polyx.bat` hace lo mismo pero mostrando la consola, útil para ver
mensajes de error si algo falla al arrancar.

> **Importante:** los modelos entrenados (`*.pt`) **no** se incluyen en la descarga de GitHub
> por su tamaño. Copia tu archivo `.pt` dentro de la carpeta `models\` para usar el Detector y el Visor.

> Requiere **Python 3.11.x** (no 3.12+). Si no lo tienes, descárgalo desde
> [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
> marcando *"Add Python to PATH"*.

## Uso

Doble clic en el acceso directo **Poly-X** del Escritorio, o en `iniciar_polyx.bat`
(Windows) / `Lanzar_macOS.command` (macOS).

Abre el **Launcher** → selecciona el módulo.

O directamente desde terminal:

```bat
REM Windows
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.gui.detector
.venv\Scripts\python.exe -m polyx.gui.trainer
.venv\Scripts\python.exe -m polyx.gui.etiquetador
.venv\Scripts\python.exe -m polyx.gui.visor
```

```bash
# macOS
.venv/bin/python -m polyx.launcher
.venv/bin/python -m polyx.gui.detector
.venv/bin/python -m polyx.gui.trainer
.venv/bin/python -m polyx.gui.etiquetador
.venv/bin/python -m polyx.gui.visor
```

## Actualizar

Para traer la última versión publicada en GitHub **sin reinstalar nada**, doble clic en:

| Sistema | Archivo |
|---|---|
| 🪟 Windows | `actualizar.bat` |
| 🍎 macOS | `actualizar_macOS.command` |

Comprueba si hay un commit nuevo en `main`; si lo hay, descarga y reemplaza solo los
archivos del programa. **Conserva** tu entorno `.venv`, tus modelos `models/*.pt`, tus
`runs/` y cualquier dato local. No necesita tener Git instalado (descarga por HTTPS) —
solo conexión a internet.

Cada actualizador se protege a sí mismo mientras corre, pero **sí actualiza los
archivos de la otra plataforma**: da igual desde cuál actualices, el proyecto queda
completo para ambas.

### Aviso automático

No hace falta acordarse de ejecutarlo: al abrir el launcher se comprueba en segundo
plano si GitHub va por delante y, si es así, aparece un botón con el identificador de
la versión nueva. La comprobación no retrasa el arranque y se calla ante cualquier
fallo — sin internet o con GitHub caído simplemente no aparece el aviso.

### Si el equipo ya tenía Poly-X

`SETUP.bat` busca otras instalaciones al final del proceso. Si encuentra alguna,
muestra qué contiene —modelos, entrenamientos, detecciones, datasets, con su tamaño—
y pide confirmación antes de tocar nada.

El orden es **copiar → verificar → retirar**, nunca al revés:

- Se copian los datos que la instalación nueva no tenga. **No se sobrescribe ningún
  archivo existente.**
- Si algo falla al copiar, la carpeta vieja se conserva intacta y se avisa.
- Solo si todo salió bien, la carpeta antigua va a la **papelera**. No se borra: si
  falta algo, se restaura.

Las carpetas con `.git` se omiten siempre: un repositorio de desarrollo no es una
instalación vieja, y retirarlo se llevaría el trabajo sin versionar.

---

## Estructura del proyecto

```
polyx/
├── launcher.py          # Menú principal
├── core/                # Módulos compartidos (theme, yolo_wrap, metrics, report_html,
│                        #   calibracion, morfologia, procedencia, i18n, plataforma)
├── detector/            # Módulo 2: análisis en lote (9 páginas)
├── trainer/             # Módulo 3: entrenamiento YOLO (9 páginas)
├── etiquetador/         # Módulo 4: anotación interactiva
└── visor/               # Módulo 5: inspección + calibración μm/px
models/                  # Pesos .pt entrenados
runs/                    # Resultados de cada ejecución
data_microplastico/      # Dataset YOLO (images/ + labels/)
tests/                   # Suite de pruebas de medida, calibración y portabilidad

SETUP.bat                # 🪟 Instalador
iniciar_polyx.bat        # 🪟 Lanzador
actualizar.bat           # 🪟 Actualizador
Lanzar_macOS.command     # 🍎 Instalador + lanzador (los dos en uno)
crear_icono_macOS.command    # 🍎 Icono que abre sin ventana de Terminal
actualizar_macOS.command # 🍎 Actualizador
construir_app_macOS.command  # 🍎 Empaquetar Poly-X.app (opcional)
```

Las diferencias entre sistemas —abrir carpetas, lanzar el actualizador, elegir el
dispositivo de cómputo— están concentradas en `polyx/model/plataforma.py`, no
repartidas por el código.

---

## Flujo completo de uso

```
Imágenes de microscopio (UV 254 nm, tinción Nile Red)
        ↓
  🏷 Etiquetador → anota PET/PP/LDPE en formato YOLO
        ↓
  🎯 Entrenador  → entrena modelo YOLO v8/v11
        ↓
  🔬 Detector    → analiza en lote + reporte HTML
        ↓
  📐 Visor       → inspección detallada + calibración μm/px
```

---

## Pruebas

La medida de forma y la calibración tienen suite propia, porque cada cifra que
producen acaba en una tabla del paper y un cambio bienintencionado puede
desplazarlas todas sin que nada avise.

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

43 pruebas sobre **formas sintéticas de talla conocida**, sin depender de ninguna
anotación humana. Cada una fija además el *porqué* de una decisión de diseño, de
modo que si alguien vuelve a intentar una variante ya descartada, la suite se lo
dice.

`pytest` solo hace falta para desarrollar y **no está en `requirements.txt`**: una
instalación de uso no lo necesita.

### Traducción

```bash
.venv\Scripts\python.exe auditar_traduccion.py --listar
```

Recorre el árbol sintáctico de cada módulo buscando llamadas a `tr()` y las
contrasta con el diccionario. Debe decir **0 sin traducir**.

---

## Publicaciones

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. *PLoS ONE* **19**(12): e0314520. https://doi.org/10.1371/journal.pone.0314520

- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).** Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in *Merluccius gayi* fish by Nile Red fluorescent staining and image analysis. *J. Chil. Chem. Soc.* **69**(1): 6082–6085. https://doi.org/10.4067/s0717-97072024000106082

---

## Manual de usuario

`Manual_PolyX.html` (español) y `Manual_PolyX.en.html` (inglés) son el manual
completo: instalación paso a paso desde GitHub con fotografías reales de cada
paso —en Windows y macOS—, las cuatro módulos pantalla por pantalla, cómo se
mide la talla, el informe, los atajos y la solución de problemas. Se abren con
doble clic y sin conexión: las imágenes van incrustadas. Junto a ellos,
`Manual_PolyX.pdf` y `Manual_PolyX.en.pdf` para imprimir o enviar.

Para regenerarlo contra la interfaz actual:

```bat
REM 1. capturas de la interfaz, una tanda por idioma
.venv\Scripts\python.exe capturar_manual.py --salida manual_screenshots/es
set POLYX_IDIOMA=en && .venv\Scripts\python.exe capturar_manual.py --salida manual_screenshots/en

REM 2. el manual, los dos idiomas y sus PDF
.venv\Scripts\python.exe generar_manual.py --pdf
```

El texto está en `manual_texto_es.py` y `manual_texto_en.py`; el motor —plantilla,
hoja de estilo y numeración de figuras— en `generar_manual.py`.

---

## Alcance del repositorio

Este repositorio documenta **el programa**. Todo lo que forma parte de un paper
en preparación —el pipeline de análisis del estudio, sus fotografías y sus
hallazgos— se queda deliberadamente fuera, porque publicarlo aquí adelantaría
resultados que aún no han salido.

---

## Licencia y contacto

**Cristofher Ferrada**  
Laboratorio de Química Ambiental · Pontificia Universidad Católica de Valparaíso  
[cristofher.ferrada@pucv.cl](mailto:cristofher.ferrada@pucv.cl)
