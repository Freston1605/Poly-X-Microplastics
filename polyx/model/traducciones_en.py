"""Diccionario espanol -> ingles de la interfaz. Solo datos, sin logica.

La clave es la cadena en espanol tal como aparece en el codigo, con sus acentos,
emojis y espacios. Si no coincide exactamente, `tr()` devuelve el espanol y la
traduccion no se aplica: `auditar_traduccion.py` detecta esos casos.

No se traducen: nombres de polimero (PET/PP/LDPE), unidades, siglas de metrica
(mAP, IoU, TP/FP/FN), nombres de archivo ni identificadores de Ultralytics.
"""
from __future__ import annotations

# Cadenas que se dejan igual a proposito: ya estan en ingles, son formatos de Qt,
# nombres propios o identificadores. Se listan para que el auditor no las cuente
# eternamente como pendientes.
NO_TRADUCIR = {
    "Poly-X", "analytics", "bestdetectormedium", "MISCLS", "Alias:",
    "%p%  —  %v / %m", "Device (0/cpu):", "IoU NMS:", "Recall:  —",
    "Poly-X · Detector", "AMP (Mixed Precision FP16)", "FP16 (half precision)",
    "Cache:", "Momentum:", "Patience:", "Workers:", "data.yaml:", "Preset:",
    "Run:", "Device:", "device:", "imgsz:", "conf:", "Label smoothing:",
    "Weight decay:", "Batch size:", "Auto-split", "Close mosaic:",
    "Save period:",
}

EN: dict[str, str] = {
    'Curvas': 'Curves',
    'Métricas en vivo': 'Live metrics',
    'Generar reporte HTML': 'Generate HTML report',
    'Guardar como…': 'Save as…',
    'Pre-anotar TODAS': 'Pre-annotate ALL',
    'Pre-anotar imagen actual': 'Pre-annotate current image',
    'Rápido': 'Fast',
    'Equilibrado': 'Balanced',
    'Máxima detección': 'Maximum detection',
    # ══ Preferencias: tema, idioma y movimiento ════════════════════
    'Preferencias': 'Preferences',
    'Tema, idioma y movimiento': 'Theme, language and motion',
    'Tema': 'Theme',
    'Se aplica al instante aquí. Los módulos ya abiertos conservan el suyo.':
        'Applied here straight away. Modules already open keep theirs.',
    'Los módulos que abras a partir de ahora salen en el idioma nuevo.':
        'Modules you open from now on come up in the new language.',
    'Movimiento': 'Motion',
    'Animar el panel del microscopio': 'Animate the microscope panel',
    'Apágalo si el movimiento te molesta o si el equipo va justo. '
    'El panel queda en un fotograma fijo.':
        'Turn it off if the motion bothers you or the machine is struggling. '
        'The panel settles on a still frame.',
    'Cerrar': 'Close',
    'El idioma se aplicará al reabrir Poly-X. Los módulos que '
    'abras desde ahora ya salen en el idioma nuevo.':
        'The language will be applied when you reopen Poly-X. Modules you open '
        'from now on already use the new language.',
    # Nombres de los temas, de theme.NOMBRES_TEMA
    'Claro': 'Light',
    'Oscuro': 'Dark',
    'Azul nocturno': 'Midnight blue',
    'Alto contraste': 'High contrast',
    # ══ Launcher: textos que cambiaron al sustituir los emoji ══════
    'LÉAME': 'READ ME',
    'Manual de usuario': 'User manual',
    'Abriendo…': 'Opening…',
    # ══ Barra lateral de los modulos ═══════════════════════════════
    'GT manual': 'Manual GT',
    'Ejecutar': 'Run',
    'Resultados': 'Results',
    'Errores': 'Errors',
    'Comparar': 'Compare',
    'Reporte': 'Report',
    'Augmentación': 'Augmentation',
    'Entrenar': 'Train',
    'Evaluar': 'Evaluate',
    'Exportar': 'Export',
    'Informe': 'Summary',

    # ══ Detector ═══════════════════════════════════════════════════
    "(esperando imagen)": "(waiting for image)",
    "(sin imagen seleccionada)": "(no image selected)",
    "0 cajas": "0 boxes",
    "0 errores": "0 errors",
    "0 imágenes": "0 images",
    "Ajustar (F)": "Fit (F)",
    "Aún no has ejecutado ninguna detección.\nVe a la pestaña Ejecutar e inicia el análisis primero.":
        "You have not run any detection yet.\nGo to the Run tab and start the analysis first.",
    "Aún no se ha ejecutado ninguna corrida.": "No run has been executed yet.",
    "Carga al menos un modelo .pt en la pestaña Modelos.":
        "Load at least one .pt model in the Models tab.",
    "Carga al menos un modelo en la pestaña Modelos.":
        "Load at least one model in the Models tab.",
    "Carpeta GT (opcional):": "Ground-truth folder (optional):",
    "Clase (para nuevas cajas):": "Class (for new boxes):",
    "Clase:": "Class:",
    "Click y arrastra para dibujar · rueda: zoom · Espacio+arrastre: mover · Supr: borrar caja.":
        "Click and drag to draw · wheel: zoom · Space+drag: pan · Del: delete box.",
    "Confianza mín. (conf):": "Min. confidence (conf):",
    "Configura μm/px en Parámetros para ver la distribución de tamaños.":
        "Set µm/px in Parameters to see the size distribution.",
    "Copia esta imagen y sus cajas corregidas a dataset_correcciones/ para mejorar el modelo con fine-tuning (active learning).":
        "Copies this image and its corrected boxes into dataset_correcciones/ so the "
        "model can be improved by fine-tuning (active learning).",
    "Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> (se crea <code>informe_deteccion.html</code> dentro de la carpeta del run y se abre en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido.":
        "After running at least one detection, press <b>Generate HTML report</b> "
        "(<code>informe_deteccion.html</code> is written inside the run folder and opened "
        "in your browser) or <b>Export to PDF</b> for a file ready to send. Every image "
        "is embedded as base64, so the file is self-contained.",
    "Detenido por el usuario.": "Stopped by the user.",
    "Deteniendo…": "Stopping…",
    "Diseñado por\nCristofher Ferrada\n2026": "Designed by\nCristofher Ferrada\n2026",
    "Ejecuta una detección primero.": "Run a detection first.",
    "Error generando PDF": "Error generating PDF",
    "Error generando reporte": "Error generating report",
    "Error guardando": "Error while saving",
    "Falló el renderizado. Como alternativa, genera el HTML y usa\nCtrl+P → 'Guardar como PDF' en tu navegador.":
        "Rendering failed. As an alternative, generate the HTML and use\n"
        "Ctrl+P → 'Save as PDF' in your browser.",
    "Falló la ejecución": "The run failed",
    "Faltan modelos": "Missing models",
    "Genera el reporte y lo guarda como PDF listo para enviar.":
        "Generates the report and saves it as a PDF ready to send.",
    "Generando PDF… (renderizando con el motor del navegador, unos segundos)":
        "Generating PDF… (rendering with the browser engine, a few seconds)",
    "Generando reporte… (puede tardar unos segundos)":
        "Generating report… (this may take a few seconds)",
    "Incluir galería comparativa (Predicción vs Ground Truth)":
        "Include side-by-side gallery (Prediction vs Ground Truth)",
    "Incluir referencias bibliográficas del autor en el reporte":
        "Include the author's bibliographic references in the report",
    "IoU para emparejar Verdaderos Positivos:": "IoU for matching True Positives:",
    "Limpiar todo": "Clear all",
    "Listo. Click y arrastra para dibujar.": "Ready. Click and drag to draw.",
    "Listo. Pulsa «Revisar en pantalla grande» para inspeccionar y corregir.":
        "Ready. Press “Review full screen” to inspect and correct.",
    "Modelo:": "Model:",
    "No encontrado": "Not found",
    "No encontrado en la raíz del proyecto.": "Not found in the project root.",
    "No hay cajas en esta imagen. Corrige o dibuja antes de enviarla.":
        "There are no boxes in this image. Correct or draw some before sending it.",
    "No hay cajas para guardar.": "There are no boxes to save.",
    "No se pudo generar el PDF": "The PDF could not be generated",
    "No se pudo leer el tamaño de la primera imagen.":
        "Could not read the size of the first image.",
    "No se pudo leer la imagen.": "Could not read the image.",
    "No se pudo medir": "Could not measure",
    "PDF no disponible": "PDF not available",
    "Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\nAlternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\nCtrl+P → 'Guardar como PDF'.":
        "Exporting to PDF needs QtWebEngine (bundled with PySide6-Addons).\n\n"
        "Alternative: press 'Generate HTML report' and, in the browser, use\n"
        "Ctrl+P → 'Save as PDF'.",
    "Perfil:": "Profile:",
    "Personalizado": "Custom",
    "Precisión:  —": "Precision:  —",
    "Revisión de detecciones — Poly-X": "Detection review — Poly-X",
    "Ruta a la carpeta con .txt YOLO (si no, busca junto a la imagen)":
        "Path to the folder with YOLO .txt files (otherwise it looks next to the image)",
    "Ruta al archivo .pt del modelo entrenado…":
        "Path to the trained model's .pt file…",
    "Selecciona imágenes en la pestaña Imágenes (se usa una para medir).":
        "Select images in the Images tab (one of them is used to measure).",
    "Selecciona imágenes en la pestaña Imágenes.": "Select images in the Images tab.",
    "Selecciona imágenes para ver qué se trocearía.":
        "Select images to see what would be tiled.",
    "Selecciona primero una imagen.": "Select an image first.",
    "Si dejas vacío, busca .txt junto a cada imagen y en /labels/ hermana. Si una imagen tiene GT, se incluirá en el análisis de errores (Verdaderos Positivos, Falsos Positivos y Falsos Negativos). Si no, solo se reportan las detecciones del modelo.":
        "If left empty, .txt files are looked up next to each image and in a sibling "
        "/labels/ folder. An image that has ground truth is included in the error "
        "analysis (True Positives, False Positives and False Negatives); otherwise only "
        "the model's detections are reported.",
    "Si tienes el modelo entrenado por el autor, úsalo como Modelo 1 con un solo clic.":
        "If you have the model trained by the author, use it as Model 1 in one click.",
    "Siguiente  →": "Next  →",
    "Sin ejecución todavía.": "No run yet.",
    "Sin resultados": "No results",
    "Tamaño imagen (imgsz):": "Image size (imgsz):",
    "Tamaño máx (μm):": "Max size (µm):",
    "Tamaño mín (μm):": "Min size (µm):",
    "Tipo:": "Type:",
    "Todos": "All",
    "Usar bestdetectormedium.pt": "Use bestdetectormedium.pt",
    "imgsz máximo": "maximum imgsz",
    "Última imagen procesada por cada modelo:": "Last image processed by each model:",
    "μm por píxel:": "µm per pixel:",
    "←  Anterior": "←  Previous",
    "Deshacer": "Undo",
    "Rehacer": "Redo",
    "Probando…": "Testing…",
    "Detener": "Stop",
    "Iniciar detección": "Start detection",
    "✓  Buena": "✓  Good",
    "✕  Limpiar": "✕  Clear",
    "✗  Mala": "✗  Bad",
    "✗ Falló la generación de PDF.": "✗ PDF generation failed.",
    "✗ Falló la generación.": "✗ Generation failed.",
    "✗ No se pudo generar el PDF.": "✗ The PDF could not be generated.",
    "Revisar en pantalla grande": "Review full screen",
    "Guardar GT (.txt YOLO)": "Save ground truth (.txt YOLO)",
    "Guardar correcciones (.txt YOLO)": "Save corrections (.txt YOLO)",
    "Seleccionar carpeta…": "Select folder…",
    "Abrir carpeta de resultados": "Open results folder",
    "Exportar CSV de detecciones": "Export detections CSV",
    "Generar reporte HTML": "Generate HTML report",
    "Exportar a PDF": "Export to PDF",
    "Enviar al dataset de reentrenamiento": "Send to the retraining dataset",
    "Seleccionar imágenes…": "Select images…",
    "Detectar máximo (GPU)": "Detect maximum (GPU)",
    "Borrar selección": "Delete selection",

    # ══ Etiquetador ════════════════════════════════════════════════
    "+ Agregar clase": "+ Add class",
    "Anotaciones (imagen actual):": "Annotations (current image):",
    "Carga un modelo .pt primero.": "Load a .pt model first.",
    "Clases activas:": "Active classes:",
    "Conserva el nivel de acercamiento al cambiar de recorte.\nCon cientos de recortes, reencuadrar cada vez cuesta mucho tiempo.\nF reencuadra manualmente.":
        "Keeps the zoom level when moving between crops.\nWith hundreds of crops, "
        "reframing every time costs a lot of time.\nF reframes manually.",
    "Conteo completo": "Counting complete",
    "Deja constancia de que miraste esta imagen aunque no tenga partículas.\nSin esto, una placa vacía revisada es indistinguible de una sin mirar.":
        "Records that you looked at this image even if it has no particles.\nWithout "
        "this, a reviewed empty plate is indistinguishable from one never looked at.",
    "Error al cargar modelo": "Error loading the model",
    "Error en pre-anotación": "Pre-annotation error",
    "Esta imagen ya tiene anotaciones. ¿Sobrescribir con el modelo?":
        "This image already has annotations. Overwrite them with the model?",
    "Mantener zoom entre imágenes": "Keep zoom between images",
    "No quedan imágenes sin revisar.": "There are no unreviewed images left.",
    "Poly-X · Etiquetador": "Poly-X · Labeler",
    "Pre-anotación automática": "Automatic pre-annotation",
    "Pre-anotando": "Pre-annotating",
    "Preset de clases:": "Class preset:",
    "Resolución de inferencia. Con partículas diminutas en fotos grandes,\nun valor bajo no propone nada.":
        "Inference resolution. With tiny particles in large photographs,\na low value "
        "proposes nothing at all.",
    "Salta a la próxima imagen que aún no has revisado (Tab)":
        "Jumps to the next image you have not reviewed yet (Tab)",
    "Siguiente →": "Next →",
    "Sin anotaciones": "No annotations",
    "Sobrescribir": "Overwrite",
    "← Anterior": "← Previous",
    "⏭  Siguiente sin revisar": "⏭  Next unreviewed",
    "✓  Revisada, siguiente   (Espacio)": "✓  Reviewed, next   (Space)",
    "Guardar (.txt)": "Save (.txt)",
    "Abrir carpeta…": "Open folder…",
    "Cargar modelo…": "Load model…",
    "Pre-anotar TODAS": "Pre-annotate ALL",
    "Pre-anotar imagen actual": "Pre-annotate current image",

    # ══ Entrenador ═════════════════════════════════════════════════
    "%v / %m épocas": "%v / %m epochs",
    "(no se encontró\nla carpeta de train)": "(the train folder\nwas not found)",
    "(sin imágenes\nen train)": "(no images\nin train)",
    "(vacío)": "(empty)",
    "<b>Ninguno</b>: sin transformaciones. &nbsp; <b>Suave</b>: solo flips y jitter HSV. &nbsp; <b>Medio</b> (recomendado): + mosaic, mixup ligero. &nbsp; <b>Fuerte</b>: + copy-paste agresivo.":
        "<b>None</b>: no transformations. &nbsp; <b>Light</b>: flips and HSV jitter only. "
        "&nbsp; <b>Medium</b> (recommended): + mosaic, light mixup. &nbsp; <b>Strong</b>: "
        "+ aggressive copy-paste.",
    "Análisis profundo": "In-depth analysis",
    "Aplica un conjunto de parámetros probados con un clic. Cambiarlo después en 'Parámetros' marca como Personalizado.":
        "Applies a tested set of parameters in one click. Changing it afterwards in "
        "'Parameters' marks it as Custom.",
    "Aún no se han generado runs.": "No runs have been generated yet.",
    "Carga data.yaml en la pestaña Dataset.": "Load data.yaml in the Dataset tab.",
    "Carpeta raíz con images/ y labels/…": "Root folder containing images/ and labels/…",
    "Clases: —": "Classes: —",
    "Cosine LR (curva coseno)": "Cosine LR (cosine schedule)",
    "Estimación de VRAM: —": "VRAM estimate: —",
    "Exportación lista": "Export finished",
    "Falló": "Failed",
    "Falló el entrenamiento": "Training failed",
    "Falta dataset": "Dataset missing",
    "Falta modelo": "Model missing",
    "Familia:": "Family:",
    "Formato:": "Format:",
    "Generando informe…": "Generating report…",
    "Incluir referencias bibliográficas": "Include bibliographic references",
    "Las curvas aparecerán aquí en cuanto el entrenamiento avance la primera época.":
        "The curves appear here as soon as training completes its first epoch.",
    "Learning rate (lr0):": "Learning rate (lr0):",
    "Los porcentajes suman > 100.": "The percentages add up to more than 100.",
    "Nivel:": "Level:",
    "No hay GPU NVIDIA disponible. En CPU recomendamos imgsz ≤ 640.":
        "No NVIDIA GPU available. On CPU we recommend imgsz ≤ 640.",
    "No hay runs disponibles.": "No runs available.",
    "No se encontraron imágenes en images/.": "No images were found in images/.",
    "Nombre del run:": "Run name:",
    "Poly-X · Entrenador": "Poly-X · Trainer",
    "Re-detectar": "Detect again",
    "Ruta a data.yaml…": "Path to data.yaml…",
    "Ruta a un .pt para continuar el entrenamiento…":
        "Path to a .pt file to continue training from…",
    "Selecciona la carpeta raíz primero.": "Select the root folder first.",
    "Selecciona un .pt válido.": "Select a valid .pt file.",
    "Selecciona un data.yaml válido.": "Select a valid data.yaml.",
    "Si tienes un .pt ya entrenado y quieres seguir desde ahí, selecciónalo. Deja vacío para usar el modelo base de la familia/tamaño elegidos.":
        "If you have an already-trained .pt and want to continue from it, select it. "
        "Leave empty to use the base model of the chosen family/size.",
    "Sin GPU no aplica esta sugerencia.": "This suggestion does not apply without a GPU.",
    "Sin data.yaml cargado.": "No data.yaml loaded.",
    "Sin run": "No run",
    "Tamaño:": "Size:",
    "Test %:": "Test %:",
    "Test: —": "Test: —",
    "Toma una carpeta con images/ y labels/ y genera train/val/test automáticamente con un nuevo data.yaml. No mueve archivos: usa listas .txt.":
        "Takes a folder with images/ and labels/ and generates train/val/test "
        "automatically with a new data.yaml. It does not move files: it uses .txt lists.",
    "Train %:": "Train %:",
    "Train: —": "Train: —",
    "Val %:": "Val %:",
    "Val: —": "Val: —",
    "Verifica que el dataset sea correcto antes de entrenar. Se comprueba automáticamente al cargar data.yaml.":
        "Checks that the dataset is sound before training. It runs automatically when "
        "data.yaml is loaded.",
    "Warmup épocas:": "Warmup epochs:",
    "vacío = auto-fecha": "empty = auto date",
    "Épocas:": "Epochs:",
    "• <b>Dataset pequeño (< 500 imgs)</b>: usa Fuerte. Más variedad sintética compensa pocos ejemplos.<br>• <b>Microplásticos PET/PP/LDPE</b>: <b>NO</b> uses HSV-H alto: los colores son la pista principal.<br>• <b>Copy-paste</b> funciona muy bien si el fondo es uniforme (como un filtro).<br>• Si el modelo no converge (mAP no sube), baja el nivel a Suave o Ninguno.":
        "• <b>Small dataset (&lt; 500 imgs)</b>: use Strong. More synthetic variety makes "
        "up for few examples.<br>• <b>PET/PP/LDPE microplastics</b>: do <b>NOT</b> use a "
        "high HSV-H: color is the main cue.<br>• <b>Copy-paste</b> works very well when the "
        "background is uniform (such as a filter).<br>• If the model does not converge "
        "(mAP does not rise), lower the level to Light or None.",
    "• <b>mAP@50</b> debe SUBIR y estabilizarse cerca del 80–95 % para un buen modelo.<br>• <b>Box loss</b> debe BAJAR de forma sostenida. Si oscila o sube, baja el lr0.<br>• Si <b>Sin mejora</b> llega a <b>Patience</b>, el entrenamiento se detendrá solo.":
        "• <b>mAP@50</b> should RISE and settle near 80–95% for a good model.<br>"
        "• <b>Box loss</b> should FALL steadily. If it oscillates or rises, lower lr0.<br>"
        "• If <b>No improvement</b> reaches <b>Patience</b>, training stops on its own.",
    "⌛  Detectando hardware…": "⌛  Detecting hardware…",
    "Iniciar entrenamiento": "Start training",
    "Iniciar validación": "Start validation",
    "✗ Archivo no encontrado.": "✗ File not found.",
    "✗ Error": "✗ Error",
    "¿Qué mirar?": "What to look at",
    "Guardar como…": "Save as…",
    "Abrir carpeta runs_train/": "Open the runs_train/ folder",
    "Exportar": "Export",
    "Otras 6": "6 more",
    "Refrescar lista": "Refresh list",
    "Validar ahora": "Validate now",
    "Auto-split + generar data.yaml": "Auto-split + generate data.yaml",

    # ══ Visor ══════════════════════════════════════════════════════
    "  Sin imagen  —  Sin calibración": "  No image  —  Not calibrated",
    "0 detecciones": "0 detections",
    "Abre una imagen primero.": "Open an image first.",
    "Calibración μm/píxel": "µm/pixel calibration",
    "Confianza mínima:": "Minimum confidence:",
    "Error al guardar": "Error while saving",
    "Error en detección": "Detection error",
    "Filtrar por clase:": "Filter by class:",
    "Guardado": "Saved",
    "Haz clic en 2 puntos sobre\nuna referencia conocida.":
        "Click 2 points on\na known reference.",
    "Haz clic en 3 puntos del borde\nde un objeto circular conocido.":
        "Click 3 points on the edge\nof a circular object of known size.",
    "Haz clic en la imagen para marcar\npuntos de referencia.":
        "Click on the image to place\nreference points.",
    "Imagen": "Image",
    "Lado mayor al que se redimensiona la imagen para inferir.\nMás alto = partículas más grandes para la red, pero más memoria.\nSe redondea al múltiplo de 32 más cercano.":
        "Longest side the image is resized to before inference.\nHigher = bigger particles "
        "for the network, but more memory.\nRounded to the nearest multiple of 32.",
    "Muestra las anotaciones YOLO que acompañan a la imagen.\nSirve para revisar el conteo manual sobre la placa, con las\ntallas ya convertidas a µm por la calibración.":
        "Shows the YOLO annotations that come with the image.\nUseful to review the manual "
        "count over the plate, with sizes\nalready converted to µm by the calibration.",
    "No hay imagen abierta.": "No image is open.",
    "Poly-X · Visor": "Poly-X · Viewer",
    "Resolución de inferencia (px):": "Inference resolution (px):",
    "Sin etiquetas": "No labels",
    "Sin imagen": "No image",
    "Todas las clases": "All classes",
    "Usar GPU si está disponible": "Use the GPU if available",
    "Detectar": "Detect",
    "✕  Cancelar calibración": "✕  Cancel calibration",
    "Círculo": "Circle",
    "Guardar info actual": "Save current info",
    "Carpeta": "Folder",
    "Cargar etiquetas (.txt)": "Load labels (.txt)",
    "Línea": "Line",
    "Sin calibrar": "Not calibrated",
    "Imagen": "Image",

    # ── Añadidas con la comparación de arquitecturas, la selección
    # por dominio, el alcance del informe y el aviso de troceado ──
    'Actualizar': 'Update',
    'Hay una versión nueva disponible en GitHub. Pulsa para descargarla e instalarla.': 'A new version is available on GitHub. Click to download and install it.',
    'Se descargará la versión nueva y se cerrará Poly-X.\n\nTus modelos, resultados y datos no se tocan. ¿Continuar?': 'The new version will be downloaded and Poly-X will close.\n\nYour models, results and data are left untouched. Continue?',
    'No se encontró actualizar.bat en la carpeta de instalación. Descarga la versión nueva manualmente desde GitHub.': 'actualizar.bat was not found in the installation folder. Download the new version manually from GitHub.',
    'Trabajo completo (todas las fotos analizadas)': 'Whole job (every analysed photo)',
    'Solo las fotos que marque abajo': 'Only the photos I tick below',
    'Ambos: un informe completo y otro con las marcadas': 'Both: one complete report and one with the ticked photos',
    'Marcar todas': 'Tick all',
    'Desmarcar todas': 'Untick all',
    'Sin fotos marcadas': 'No photos ticked',
    '⚠ ninguna marcada': '⚠ none ticked',
    '✗ No se pudo generar todo el PDF.': '✗ Could not generate the whole PDF.',
    'Falló el renderizado de: ': 'Rendering failed for: ',
    "Elegiste generar el informe de fotos concretas, pero no hay ninguna marcada.\n\nMarca al menos una en la lista, o cambia el alcance a 'Trabajo completo'.": "You chose a report of specific photos, but none are ticked.\n\nTick at least one in the list, or change the scope to 'Whole job'.",
    'Puedes generar el informe del <b>trabajo completo</b>, solo de las <b>fotos que elijas</b>, o ambos de una vez. Las cifras, los gráficos y la matriz de confusión se recalculan sobre lo que elijas, así que el informe siempre describe exactamente las fotos que muestra.': 'You can generate the report for the <b>whole job</b>, for <b>only the photos you choose</b>, or both at once. Figures, charts and the confusion matrix are recomputed over your choice, so the report always describes exactly the photos it shows.',
    '{} foto(s) en el trabajo': '{} photo(s) in the job',
    '{} de {} marcadas': '{} of {} ticked',
    '{} foto(s) marcada(s)': '{} ticked photo(s)',
    '{} informe(s) generado(s) en {}': '{} report(s) generated in {}',
    '{} PDF generado(s) en {}': '{} PDF(s) generated in {}',
    'Entrenar ambas arquitecturas (v8 y v11) con la misma configuración': 'Train both architectures (v8 and v11) with the same configuration',
    'Elegir el mejor peso según las fotos de terreno, no las de laboratorio': 'Pick the best weight by field photos, not laboratory ones',
    'Se entrenará solo <b>{}</b>. Marca la casilla para entrenar también la otra familia y compararlas en igualdad de condiciones.': 'Only <b>{}</b> will be trained. Tick the box to also train the other family and compare them on equal terms.',
    'Se entrenarán <b>{}</b> uno tras otro, con idénticos imgsz, batch, épocas, semilla y augmentación. Así la diferencia de métricas se puede atribuir a la arquitectura y no a los hiperparámetros.<br>Tarda <b>el doble</b>: van en secuencia porque comparten GPU. Al terminar, la comparación sale en el log y en la pestaña Comparar.': '<b>{}</b> will be trained one after the other, with identical imgsz, batch, epochs, seed and augmentation. That way any difference in metrics can be attributed to the architecture rather than the hyperparameters.<br>It takes <b>twice as long</b>: they run in sequence because they share the GPU. When finished, the comparison appears in the log and in the Compare tab.',
    'Al terminar, se evalúan todos los checkpoints guardados contra <b>solo</b> las imágenes de sedimento real de la validación y se guarda el ganador como <b>best_real.pt</b>.<br>Hace falta porque <code>best.pt</code> lo elige Ultralytics por el mAP global, y en un dataset mixto ese número lo dominan las placas dopadas de laboratorio — el dominio que justamente no transfiere a las fotos de terreno. <code>best.pt</code> se conserva intacto.': 'When training finishes, every saved checkpoint is evaluated against <b>only</b> the real sediment images of the validation split, and the winner is saved as <b>best_real.pt</b>.<br>This is needed because Ultralytics picks <code>best.pt</code> by the global mAP, and in a mixed dataset that number is dominated by the laboratory spiked plates — precisely the domain that does not transfer to field photos. <code>best.pt</code> is left untouched.',
    'Troceado desactivado: cada foto entra completa a imgsz {}. En fotos grandes las partículas pequeñas pueden desaparecer al reescalar.': 'Tiling disabled: each photo enters whole at imgsz {}. In large photos, small particles may vanish when rescaled.',
    'Las fotos ({}×{} px) no llegan al umbral de {} px: se analizan de una pieza.': 'The photos ({}×{} px) do not reach the {} px threshold: they are analysed in one piece.',
    '⚠ Fotos grandes ({}×{} px): se analizarán en {} recortes de {} px con {}% de solape. Va a tardar unas {}× más.<br>El recorte es solo para que el modelo pueda detectar: los resultados y el informe se entregan sobre la foto completa.': '⚠ Large photos ({}×{} px): they will be analysed as {} tiles of {} px with {}% overlap. It will take about {}× longer.<br>Tiling is only so the model can detect: results and the report are delivered on the complete photo.',

    # ── Cadenas que el auditor no veia: viven en asignaciones de
    # clase (PAGE_TITLE = tr(...)) o en llamadas a self.card(tr(...)) ──
    'Poly-X · Suite de microplásticos': 'Poly-X · Microplastics suite',
    'Idioma': 'Language',
    '● PLATAFORMA DE ANÁLISIS': '● ANALYSIS PLATFORM',
    'Plataforma de detección y clasificación de microplásticos': 'Detection and classification platform for microplastics',
    'MÓDULOS': 'MODULES',
    'MÓDULO': 'MODULE',
    'Abrir  →': 'Open  →',
    'LÉAME': 'README',
    'Manual de usuario': 'User manual',
    'Polímeros': 'Polymers',
    'fluorescencia Nile Red': 'Nile Red fluorescence',
    'medición': 'measurement',
    'calibración por píxel': 'per-pixel calibration',
    'Detector': 'Detector',
    'Entrenador': 'Trainer',
    'Etiquetador': 'Labeler',
    'Visor': 'Viewer',
    'Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo.': 'Automated detection of PET, PP and LDPE by Nile Red fluorescence (254 nm) and machine learning. Training, labeling, detection and reporting in a single workflow.',
    'Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Dr (c) en Química, 2026': 'Designed and developed by <b>Cristofher Ferrada</b> · PhD (c) in Chemistry, 2026',
    'Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, CSV con centroides y diámetros, métricas globales e informe HTML de detección.': 'Analyzes images with a trained .pt model. Produces annotated output, a CSV of centroids and diameters, global metrics and an HTML detection report.',
    'Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas de calidad y comparación con runs anteriores.': 'Trains YOLO v8 / v11 models. Live curves, automatic quality recommendations and comparison against earlier runs.',
    'Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo existente y atajos de teclado, ahorra ~80 % del tiempo manual.': 'Annotates images in YOLO format. Supports pre-annotation with an existing model plus keyboard shortcuts, saving ~80% of the manual time.',
    'Inspección de una imagen a la vez con calibración interactiva μm/píxel (línea o círculo) y medición precisa por partícula.': 'Inspects one image at a time with interactive μm/pixel calibration (line or circle) and precise per-particle measurement.',
    'Sin imágenes': 'No images',
    'Imágenes': 'Images',
    'Modelos': 'Models',
    'Modelo': 'Model',
    'Parámetros': 'Parameters',
    'Sin modelo': 'No model',
    'Error': 'Error',
    'Dataset': 'Dataset',
    '● Listo': '● Ready',
    'Carga rápida': 'Quick load',
    'Carga hasta 3 modelos .pt entrenados para detectar microplásticos. Si cargas más de uno, se compararán automáticamente en el reporte final.': 'Load up to 3 trained .pt models for microplastics detection. If you load more than one, they are compared automatically in the final report.',
    'Origen': 'Source',
    'Imágenes cargadas': 'Loaded images',
    "Selecciona imágenes individuales o una carpeta. Si tienes etiquetas verdaderas en formato YOLO (.txt) en la misma carpeta o en una hermana 'labels/', se cargarán y activarán el análisis de errores.": "Select individual images or a folder. If you have ground truth labels in YOLO format (.txt) in the same folder or in a sibling 'labels/' folder, they are loaded and enable error analysis.",
    'Inferencia': 'Inference',
    'Troceado automático (fotos grandes)': 'Automatic tiling (large photos)',
    'Análisis de errores (si hay GT)': 'Error analysis (when GT is available)',
    'Calibración óptica (tamaño de partícula)': 'Optical calibration (particle size)',
    'Filtro por tamaño (opcional)': 'Size filter (optional)',
    'Cuándo trocear:': 'When to tile:',
    'automático': 'automatic',
    'siempre': 'always',
    'nunca': 'never',
    'Umbral (lado mayor, px):': 'Threshold (longest side, px):',
    'Lado del tile (px):': 'Tile side (px):',
    'Solape entre tiles:': 'Overlap between tiles:',
    'Configura confianza, IoU, calibración óptica y filtros. La calibración μm/px es opcional pero recomendada para reportes científicos.': 'Set confidence, IoU, optical calibration and filters. The μm/px calibration is optional but recommended for scientific reports.',
    'Elige un perfil y listo. «Máxima detección» es el recomendado para microplásticos pequeños en fotos de microscopía de alta resolución.': 'Pick a profile and you are done. «Maximum detection» is recommended for small microplastics in high-resolution microscopy photos.',
    '0.05–0.95. Más alta = menos detecciones (con falsos positivos).': '0.05–0.95. Higher = fewer detections (and fewer false positives).',
    '0.30–0.70. Suprime cajas superpuestas.': '0.30–0.70. Suppresses overlapping boxes.',
    'Estándar COCO: 0.50. Más estricto: 0.75.': 'COCO standard: 0.50. Stricter: 0.75.',
    "'0' = primera GPU. 'cpu' = CPU. '0,1' = GPU 0 y 1.": "'0' = first GPU. 'cpu' = CPU. '0,1' = GPU 0 and 1.",
    '0 = sin filtro. Aplica sobre el diámetro equivalente.': '0 = no filter. Applies to the equivalent diameter.',
    '«auto» decide sola mirando el lado mayor de cada foto.': '«auto» decides on its own from the longest side of each photo.',
    'Más grande = detecta partículas más pequeñas (clave para microplásticos), pero más lento y usa más memoria GPU. Para fotos de alta resolución sube a 4096+. El botón prueba el máximo que aguanta tu GPU con el modelo cargado.': 'Larger = detects smaller particles (key for microplastics), but slower and heavier on GPU memory. For high-resolution photos go up to 4096+. The button probes the maximum your GPU can take with the loaded model.',
    'Una foto de placa completa entra a la red reescalada a imgsz: a 4096 px reducidos a 2080, cada partícula encoge a la mitad y desaparece bajo el stride de la red. Troceando, cada tile entra a resolución nativa y las cajas vuelven a coordenadas de la foto completa, fusionadas con NMS global — no hay que cortar nada a mano ni sumar los conteos después.': 'A full-plate photo enters the network rescaled to imgsz: at 4096 px reduced to 2080, every particle shrinks by half and disappears below the network stride. With tiling, each tile enters at native resolution and the boxes return to full-photo coordinates, merged with a global NMS — nothing has to be cut by hand nor the counts added up afterwards.',
    'Por encima de esto se trocea. 2000 deja pasar enteros los recortes del estudio (1630 px) y trocea las placas completas (~4096 px).': 'Above this, tiling kicks in. 2000 lets the study crops (1630 px) through whole and tiles the full plates (~4096 px).',
    '0 = automático, min(umbral, imgsz). Conviene que coincida con el tamaño de los recortes con que se entrenó el modelo: si el modelo vio partículas a una escala y el tile las presenta a otra, el recall cae.': '0 = automatic, min(threshold, imgsz). It should match the size of the crops the model was trained on: if the model saw particles at one scale and the tile presents them at another, recall drops.',
    '0.25 = cada tile pisa un cuarto del vecino. El solape es lo que evita perder o duplicar la partícula que cae justo en la costura.': '0.25 = each tile overlaps a quarter of its neighbour. The overlap is what avoids losing or duplicating a particle that falls right on the seam.',
    'Verdaderos Positivos: clase correcta + IoU ≥ umbral.  Falsos Positivos: predicción sin GT cercano.  Falsos Negativos: GT no detectado.  Mal Clasificados: IoU alto pero clase distinta.': 'True Positives: correct class + IoU ≥ threshold.  False Positives: prediction with no nearby GT.  False Negatives: GT not detected.  Misclassified: high IoU but wrong class.',
    '0 = sin medición. Ej: objetivo 40× y CMOS calibrado → ~0.244 μm/px.': '0 = no measurement. E.g. 40× objective with a calibrated CMOS → ~0.244 μm/px.',
    'El detector calcula área y diámetro equivalente (de círculo con misma área) para cada partícula detectada. Si hay calibración, también en μm y μm².': 'The detector computes the area and equivalent diameter (of a circle with the same area) for every detected particle. With calibration, also in μm and μm².',
    'Progreso': 'Progress',
    'Preview en vivo': 'Live preview',
    'Inicia la inferencia. Verás progreso en vivo, previews de imágenes anotadas y podrás cancelar en cualquier momento.': 'Starts inference. You will see live progress, previews of annotated images, and you can cancel at any time.',
    'Métricas generales': 'General metrics',
    'Métricas detalladas (modelo principal)': 'Detailed metrics (main model)',
    'Distribución de tamaños (μm) — solo con calibración activa': 'Size distribution (μm) — only with calibration active',
    'Exportar datos': 'Export data',
    'Por imagen': 'Per image',
    'Resumen cuantitativo del análisis. Para gráficos completos y galería por imagen, genera el reporte HTML en la pestaña Reporte.': 'Quantitative summary of the analysis. For full charts and a per-image gallery, generate the HTML report in the Report tab.',
    'Filtros': 'Filters',
    'Cajas con error': 'Boxes with errors',
    'Lista de cajas problemáticas (solo si hay Ground Truth). Falsos Positivos: detección sin GT cercano. Falsos Negativos: GT no detectado. Mal Clasificados: bien localizado, mala clase.': 'List of problematic boxes (only with Ground Truth). False Positives: detection with no nearby GT. False Negatives: GT not detected. Misclassified: located correctly, wrong class.',
    'Resumen por modelo': 'Per-model summary',
    'Detecciones por imagen': 'Detections per image',
    'Si cargaste más de un modelo, aquí ves tabla resumen con todas las métricas lado a lado y cuántas detecciones hizo cada uno por imagen.': 'If you loaded more than one model, here is a summary table with every metric side by side and how many detections each one made per image.',
    'Informe de detección': 'Detection report',
    'Generar informe': 'Generate report',
    'Contenido del reporte': 'Report contents',
    'Alcance del informe': 'Report scope',
    'trabajo completo': 'whole job',
    'Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, galería comparativa y análisis de errores. Puedes exportarlo directamente a PDF para enviarlo.': 'Generates a self-contained HTML report (all images embedded in base64, so they do not break when you send it to someone else) with methods, metrics, charts, comparison gallery and error analysis. You can export it straight to PDF.',
    'Anotación': 'Annotation',
    'Anotador completo de Ground Truth. Click izquierdo para dibujar, click sobre una caja para seleccionar, arrastra los handles para redimensionar, Espacio o botón medio para pan, rueda para zoom. Teclas 1..9: clase activa · Supr: borrar · Ctrl+Z/Y: undo/redo · ←/→: imagen anterior/siguiente (autoguarda).': 'Full Ground Truth annotator. Left click to draw, click a box to select it, drag the handles to resize, Space or middle button to pan, wheel to zoom. Keys 1..9: active class · Del: delete · Ctrl+Z/Y: undo/redo · ←/→: previous/next image (autosaves).',
    'Seleccionar modelo': 'Select model',
    'Preset rápido': 'Quick preset',
    'Familia y tamaño': 'Family and size',
    'Pesos personalizados (opcional)': 'Custom weights (optional)',
    "Elige el preset según el caso, o personaliza familia y tamaño. El preset 'Balanceado' es el recomendado para la mayoría de proyectos.": "Pick the preset that fits your case, or customize family and size. The 'Balanced' preset is recommended for most projects.",
    'data.yaml': 'data.yaml',
    'Conteos por split': 'Counts per split',
    'Vista previa (6 imágenes random con cajas)': 'Preview (6 random images with boxes)',
    'Auto-split (si tu dataset no está dividido)': 'Auto-split (if your dataset is not divided)',
    'Validación del dataset': 'Dataset validation',
    'Carga tu data.yaml. Validamos la estructura, contamos imágenes y mostramos una vista previa con cajas. Si tu dataset no está dividido, usa Auto-split.': 'Load your data.yaml. We validate the structure, count images and show a preview with boxes. If your dataset is not divided, use Auto-split.',
    'Parámetros de entrenamiento': 'Training parameters',
    'Configuración básica': 'Basic configuration',
    'Optimizador y learning rate': 'Optimizer and learning rate',
    'Early stopping y checkpoints': 'Early stopping and checkpoints',
    'Hardware e I/O': 'Hardware and I/O',
    'Configuración': 'Configuration',
    'Maximizar imgsz para mi GPU': 'Maximize imgsz for my GPU',
    'Detectando GPU…': 'Detecting GPU…',
    'Detectar GPU / VRAM': 'Detect GPU / VRAM',
    'Sugerir imgsz MÁXIMO': 'Suggest MAXIMUM imgsz',
    'Sugerir batch para mi GPU': 'Suggest batch for my GPU',
    'Optimizar todo (imgsz → batch → velocidad)': 'Optimize everything (imgsz → batch → speed)',
    'Sin GPU': 'No GPU',
    'Sin GPU utilizable': 'No usable GPU',
    'Configuración optimizada': 'Optimized configuration',
    "Cada parámetro tiene un hint con su explicación. Los valores por defecto son sensatos para microplásticos. Si tocas algo, el preset cambia a 'Personalizado'.": "Every parameter has a hint explaining it. The defaults are sensible for microplastics. If you change anything, the preset switches to 'Custom'.",
    'Fija imgsz al máximo que aguanta la tarjeta (sin pasar de la resolución nativa del dataset), después sube el batch con lo que sobre, y al final ajusta AMP, workers y cache. En ese orden.': "Sets imgsz to the maximum the card can take (without exceeding the dataset's native resolution), then raises the batch with what is left, and finally tunes AMP, workers and cache. In that order.",
    'MÁS ALTO = mejor para partículas pequeñas. Recomendado: 1280+.': 'HIGHER = better for small particles. Recommended: 1280+.',
    'Bajar si hay OOM. Subir si la GPU tiene RAM de sobra.': 'Lower it on OOM. Raise it if the GPU has RAM to spare.',
    '150–300 típico. Early stopping detiene si no mejora (ver Patience).': '150–300 is typical. Early stopping halts if it stops improving (see Patience).',
    'Épocas sin mejora antes de detener. 50 = sensato.': 'Epochs without improvement before stopping. 50 is sensible.',
    'Guardar checkpoint cada N épocas.': 'Save a checkpoint every N epochs.',
    "'0' = primera GPU. 'cpu' = CPU (muy lento).": "'0' = first GPU. 'cpu' = CPU (very slow).",
    "'ram' es lo más rápido si cabe; 'disk' es seguro; 'False' para datasets gigantes.": "'ram' is fastest when it fits; 'disk' is safe; 'False' for huge datasets.",
    'Augmentación de datos': 'Data augmentation',
    'Nivel de augmentación': 'Augmentation level',
    'Sliders manuales (avanzado)': 'Manual sliders (advanced)',
    'Recomendaciones': 'Recommendations',
    'Aumenta artificialmente la variedad del dataset. Útil para datasets pequeños (< 500 imágenes). Demasiado aug = el modelo no converge; muy poco = sobreajuste.': 'Artificially increases the variety of the dataset. Useful for small datasets (< 500 images). Too much augmentation and the model will not converge; too little and it overfits.',
    'Control': 'Control',
    'Inicia el entrenamiento del modelo con los parámetros configurados. Verás curvas en vivo, métricas y log. Puedes detenerlo en cualquier momento.': 'Starts training the model with the configured parameters. You will see live curves, metrics and the log. You can stop it at any time.',
    'Valida cualquier modelo .pt sobre un dataset YOLO (data.yaml). Útil para comprobar la calidad de un best.pt sobre un test set distinto al de entrenamiento.': 'Validates any .pt model against a YOLO dataset (data.yaml). Useful for checking the quality of a best.pt on a test set other than the training one.',
    'Comparar runs': 'Compare runs',
    'Runs disponibles': 'Available runs',
    'Tabla con todos los entrenamientos anteriores en runs_train/. Útil para comparar configuraciones y elegir el mejor modelo.': 'A table of every previous training run in runs_train/. Useful for comparing configurations and picking the best model.',
    'Exportar modelo': 'Export model',
    'Modelo a exportar': 'Model to export',
    'Formato y opciones': 'Format and options',
    'Convierte el modelo entrenado a otros formatos para producción: ONNX (portable), TensorRT (NVIDIA rápido), TFLite (móvil), CoreML (Apple), etc.': 'Converts the trained model to other production formats: ONNX (portable), TensorRT (fast on NVIDIA), TFLite (mobile), CoreML (Apple), and so on.',
    'Informe del entrenamiento': 'Training report',
    'Selección de run': 'Run selection',
    'Generar': 'Generate',
    'Genera un informe HTML autocontenido con las curvas y métricas del run elegido. Listo para convertir a PDF (Ctrl+P en el navegador).': 'Generates a self-contained HTML report with the curves and metrics of the chosen run. Ready to convert to PDF (Ctrl+P in the browser).',

    # ── Descripciones de tamano del modelo (SIZE_DESCRIPTIONS) ──
    'El más liviano y rápido. Útil para probar.': 'The lightest and fastest. Useful for testing.',
    'Buen balance velocidad/precisión.': 'A good speed/accuracy balance.',
    'RECOMENDADO. Precisión sólida en GPU media.': 'RECOMMENDED. Solid accuracy on a mid-range GPU.',
    'Más preciso, requiere GPU potente.': 'More accurate, needs a powerful GPU.',
    'El más preciso, muy lento sin GPU buena.': 'The most accurate, very slow without a good GPU.',
    # ── Anadidas en agosto de 2026: calibracion contra la placa, medida de
    # forma de particula, selector de secciones del informe y revision
    # particula a particula en el Visor. ──
    '   ⚠ Vuelve a Ejecutar la detección: el informe usa el GT leído en la última corrida.':
        '   ⚠ Run the detection again: the report uses the ground truth read during the last run.',
    'No hay ninguna caja dibujada. ¿Guardar esta imagen como revisada con cero partículas? Se sobrescribirá la anotación anterior, si la hubiera.':
        'No boxes have been drawn. Save this image as reviewed with zero particles? Any previous annotation will be overwritten.',
    'Medir la placa en cada foto y usar su escala (recomendado)':
        'Measure the dish in every photo and use its scale (recommended)',
    'Respaldo: se usa solo si no se puede medir la placa. 0 = sin medición.':
        'Fallback: used only when the dish cannot be measured. 0 = no calibration.',
    'Diámetro real de la placa (mm):':
        'Actual dish diameter (mm):',
    'Diámetro externo nominal. La placa Petri del estudio mide 100 mm.':
        'Nominal outer diameter. The Petri dish used in this study is 100 mm.',
    'Altura de la placa (mm):':
        'Dish height (mm):',
    '0 = no corregir. Una Petri de 100 mm suele medir 15 mm de alto.':
        '0 = no correction. A 100 mm Petri dish is usually 15 mm tall.',
    'Distancia cámara → base (mm):':
        'Camera-to-base distance (mm):',
    '0 = no corregir. Medir hasta el FONDO de la placa, no hasta el borde.':
        '0 = no correction. Measure to the BOTTOM of the dish, not to its rim.',
    'No hay que marcar nada: el borde de la placa se encuentra solo. El radio se obtiene ajustando un círculo al anillo muestreado en 720 direcciones, con rechazo de atípicos; Hough solo aporta el centro aproximado porque su radio llega a errar un 12 %. En los recortes la placa no aparece, así que ahí la escala se hereda del índice de calibración que dejó el recorte.':
        'Nothing has to be marked: the dish rim is found automatically. The radius comes from fitting a circle to the rim sampled along 720 directions, with outlier rejection; the Hough transform only supplies the approximate centre, because its radius can be off by as much as 12 %. In crops the dish is not visible, so there the scale is inherited from the calibration index left behind when the crop was made.',
    'El borde de la placa está más cerca de la cámara que el fondo donde reposan las partículas, de modo que se proyecta más grande y la escala sale pequeña: sin corregir, TODAS las tallas quedan subestimadas. Con la placa a 15 mm y la cámara a 100 mm son casi 16 puntos porcentuales; a 500 mm, un 3 %. Rellena los dos campos de arriba para corregirlo.':
        'The dish rim sits closer to the camera than the base where the particles lie, so it projects larger and the resulting scale is too small: left uncorrected, ALL sizes are underestimated. With a 15 mm dish and the camera at 100 mm that is almost 16 percentage points; at 500 mm, about 3 %. Fill in both fields above to correct it.',
    'Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, galería comparativa, conteo por muestra y análisis de errores. Puedes exportarlo directamente a PDF para enviarlo, o guardar aparte cada foto con sus etiquetas dibujadas.':
        'Generates a self-contained HTML report (every image embedded as base64, so nothing breaks when you send it to someone else) with methods, metrics, charts, a comparison gallery, per-sample counts and error analysis. You can export it straight to PDF to send it, or save each photo separately with its labels drawn on.',
    'Generar informe HTML':
        'Generate HTML report',
    'Guardar las fotos con las etiquetas':
        'Save the photos with their labels',
    'Guarda cada foto analizada con sus cajas dibujadas, en su resolución original. Eliges una carpeta y dentro se crea una subcarpeta por cada opción que marques (<code>conteo_manual</code>, <code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), así no se mezclan. Respeta el <b>alcance</b> elegido arriba: si marcaste fotos concretas, solo se guardan esas.':
        'Saves every analysed photo with its boxes drawn on, at its original resolution. You pick a folder and a subfolder is created inside it for each option you tick (<code>conteo_manual</code>, <code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), so they do not get mixed up. It honours the <b>scope</b> chosen above: if you ticked specific photos, only those are saved.',
    'Etiquetas manuales (Ground Truth)':
        'Manual labels (ground truth)',
    'Detecciones del modelo':
        'Model detections',
    'Las dos superpuestas en la misma foto':
        'Both overlaid on the same photo',
    'Guardar fotos etiquetadas':
        'Save labelled photos',
    'Carpeta donde guardar las fotos etiquetadas':
        'Folder to save the labelled photos in',
    'Qué incluir en el informe':
        'What to include in the report',
    'El informe completo es largo. Desmarca lo que no necesites: las secciones se renumeran solas y el índice se ajusta. Una sección marcada que no tenga datos —errores sin ground truth, por ejemplo— se omite igualmente.':
        'The full report is long. Untick whatever you do not need: sections renumber themselves and the table of contents adjusts. A ticked section with no data — error analysis without ground truth, for instance — is left out anyway.',
    'Galería por imagen  (es lo que más pesa)':
        'Per-image gallery  (this is what makes the file big)',
    'Nada que guardar':
        'Nothing to save',
    'Marca al menos una de las tres opciones: etiquetas manuales, detecciones del modelo, o las dos superpuestas.':
        'Tick at least one of the three options: manual labels, model detections, or both overlaid.',
    "El alcance está en 'solo las fotos que marque' y no hay ninguna marcada.":
        "The scope is set to 'only the photos I tick' and none are ticked.",
    "No se encontró ninguna imagen anotada en disco para lo que pediste.\n\nSi marcaste 'etiquetas manuales', recuerda que solo existen para las fotos que tienen Ground Truth.":
        "No annotated image was found on disk for what you asked for.\n\nIf you ticked 'manual labels', remember that these only exist for photos that have ground truth.",
    'Completo':
        'Full',
    'Resumen breve':
        'Short summary',
    'Metodológico':
        'Methodological',
    'Guardando {} imagen(es)…':
        'Saving {} image(s)…',
    'Algunas no se pudieron guardar':
        'Some could not be saved',
    '{} imagen(es) en {} carpeta(s) dentro de {}':
        '{} image(s) in {} folder(s) inside {}',
    '{} fallaron':
        '{} failed',
    'Métricas detalladas (suma de todos los modelos)':
        'Detailed metrics (summed over all models)',
    'Ground truth':
        'Ground truth',
    'No pide ninguna carpeta: relee los .txt que están junto a cada foto, los mismos que ves en GT manual. Hace falta porque la corrida guarda el ground truth que leyó, así que corregir una anotación después no cambia por sí solo estas métricas ni el informe. Recalcula sin pasar el modelo otra vez y solo redibuja las imágenes cuyo .txt cambió.':
        'It asks for no folder: it re-reads the .txt files sitting next to each photo, the same ones you see under Manual GT. This is needed because a run freezes the ground truth it read, so fixing an annotation afterwards does not by itself change these metrics or the report. It recalculates without running the model again, and only redraws the images whose .txt actually changed.',
    'Releer los .txt del disco y recalcular':
        'Re-read the .txt files from disk and recalculate',
    'Revisar partícula a partícula en el Visor':
        'Review particle by particle in the Viewer',
    'Abre la imagen seleccionada en el Visor con estas mismas detecciones, numeradas. Allí cada partícula muestra sobre qué se midió y si es fibra o fragmento — sin volver a pasar el modelo.':
        'Opens the selected image in the Viewer with these very detections, numbered. There each particle shows what it was measured on and whether it is a fibre or a fragment — without running the model again.',
    'Releyendo ground truth…':
        'Re-reading ground truth…',
    'Cancelar':
        'Cancel',
    'Recargar GT':
        'Reload GT',
    'Revisar':
        'Review',
    'Selecciona primero una fila de la tabla de abajo.':
        'Select a row from the table below first.',
    'Esa imagen no tiene detecciones que revisar.':
        'That image has no detections to review.',
    'Guardar también el mejor peso para el conjunto real':
        'Also save the best weights for the real dataset',
    'Al terminar quedan tres pesos en la carpeta del run: <b>best_sintetico.pt</b>, <b>best_real.pt</b> y <b>last.pt</b> (este último sirve para reanudar el entrenamiento).':
        'When it finishes there are three sets of weights in the run folder: <b>best_sintetico.pt</b>, <b>best_real.pt</b> and <b>last.pt</b> (this last one is what you use to resume training).',
    'Cargar predicciones de una corrida':
        'Load predictions from a run',
    'Lee las predicciones que el Detector dejó en runs/detect_.../ para esta misma foto. Sirve para revisar una corrida ya cerrada, sin volver a pasar el modelo.':
        'Reads the predictions the Detector left in runs/detect_.../ for this same photo. Use it to review a run you have already closed, without running the model again.',
    'Selecciona una partícula para ver su medición':
        'Select a particle to see how it was measured',
    'Carpeta de la corrida (runs/detect_...)':
        'Run folder (runs/detect_...)',
    'Abre primero la foto original.':
        'Open the original photo first.',
    'Sin predicciones':
        'No predictions',
    'Varios modelos':
        'Several models',
    'Esta foto se analizó con varios modelos. ¿Cuál revisar?':
        'This photo was analysed with several models. Which one do you want to review?',
    'El archivo está vacío: esa foto no tuvo detecciones.':
        'The file is empty: that photo had no detections.',
    'No se pudo aislar esta partícula del fondo':
        'This particle could not be separated from the background',
    '● {n} predicciones cargadas de {c}':
        '● {n} predictions loaded from {c}',
    "En esa corrida no hay predicciones para {n}.\n\nSe buscó en las subcarpetas 'labels'.":
        "That run has no predictions for {n}.\n\nThe 'labels' subfolders were searched.",
    # ── Cadenas que estaban fuera de tr() y se detectaron arrancando los
    # tres modulos en ingles y buscando texto con aspecto de espanol. ──
    'Detecciones': 'Detections',
    'Confianza media': 'Mean confidence',
    'Tamaño medio (μm)': 'Mean size (μm)',
    'Época actual': 'Current epoch',
    'Sin mejora': 'No improvement',
    'Split train con imágenes': 'Train split with images',
    'Split val con imágenes': 'Val split with images',
    'Resumen': 'Summary',
    'Métodos': 'Methods',
    'Calibración de escala': 'Scale calibration',
    'Forma y talla de las partículas': 'Particle shape and size',
    'Ficha de partículas medidas (muestra)': 'Measured-particle cards (sample)',
    'Resultados generales': 'Overall results',
    'Análisis de errores': 'Error analysis',
    'Comparación entre modelos': 'Model comparison',
    'Galería por imagen': 'Per-image gallery',
    'Conteo por muestra y tipo de plástico': 'Counts by sample and polymer type',
    'Referencias bibliográficas': 'References',
    # El indice del informe y el aviso de deteccion de GPU: las dos ultimas
    # cadenas visibles que quedaban sin traducir.
    '<ol><li><b>Resumen</b> con detecciones, confianza media y tamaño medio.</li><li><b>Métodos</b> — modelo, parámetros, calibración y dispositivo.</li><li><b>Calibración de escala</b> — de dónde salió el µm/px de cada foto.</li><li><b>Forma y talla</b> — reparto fibra/fragmento, distribución de tallas, la mayor y la menor, recuento por imagen y cómo se mide el largo.</li><li><b>Fichas</b> — una muestra de partículas con su medición dibujada.</li><li><b>Resultados generales</b> — clases, confianza y tamaños.</li><li><b>Resumen por modelo</b> (tabla comparativa)</li><li><b>Análisis de errores</b> (solo si hay ground truth) — matriz de confusión y P/R/F1 por clase.</li><li><b>Comparación entre modelos</b></li><li><b>Galería por imagen</b> — predicción y ground truth lado a lado.</li><li><b>Conteo por muestra y tipo de plástico</b> — por imagen, por tramo y por estación.</li><li><b>Referencias bibliográficas</b></li></ol><p>Todas las secciones son opcionales: se eligen arriba.</p>':
        "<ol><li><b>Summary</b> with detections, mean confidence and mean size.</li><li><b>Methods</b> — model, parameters, calibration and device.</li><li><b>Scale calibration</b> — where each photo's µm/px came from.</li><li><b>Shape and size</b> — fibre/fragment split, size distribution, the largest and the smallest, per-image counts and how length is measured.</li><li><b>Particle cards</b> — a sample of particles with their measurement drawn on.</li><li><b>Overall results</b> — classes, confidence and sizes.</li><li><b>Per-model summary</b> (comparison table)</li><li><b>Error analysis</b> (only when ground truth is available) — confusion matrix and per-class P/R/F1.</li><li><b>Model comparison</b></li><li><b>Per-image gallery</b> — prediction and ground truth side by side.</li><li><b>Counts by sample and polymer type</b> — per image, per depth interval and per station.</li><li><b>References</b></li></ol><p>Every section is optional: you choose them above.</p>",
    '⌛  Detectando GPU en background (puede tardar 10–30 s la primera vez)…':
        '⌛  Detecting GPU in the background (may take 10–30 s the first time)…',
    # ── Ultimo barrido: alias por defecto, estimacion de VRAM y
    # descripciones de los formatos de exportacion. ──
    'Modelo .pt:': 'Model .pt:',
    'Pesos .pt:': 'Weights .pt:',
    'Modelo {n}': 'Model {n}',
    'Estimación de VRAM: ~{gb} (modelo {size}, imgsz {imgsz}, batch {batch}, AMP={amp})': 'VRAM estimate: ~{gb} (model {size}, imgsz {imgsz}, batch {batch}, AMP={amp})',
    'Más portable. Para CPU o GPU vía onnxruntime.': 'Most portable. For CPU or GPU via onnxruntime.',
    'Modelo nativo de PyTorch.': 'Native PyTorch model.',
    'Solo NVIDIA. Inferencia más rápida en producción.': 'NVIDIA only. Fastest inference in production.',
    'Intel CPU/iGPU.': 'Intel CPU/iGPU.',
    'Mobile/Edge devices.': 'Mobile/edge devices.',
    'Apple Silicon (M1/M2/M3) y iOS.': 'Apple Silicon (M1/M2/M3) and iOS.',

    # ── Soporte macOS y secciones nuevas del informe (agosto 2026) ──
    "'mps' = GPU del Mac (Apple Silicon). 'cpu' = procesador.":
        "'mps' = the Mac's built-in GPU (Apple Silicon). 'cpu' = processor.",
    "Este Mac es Intel: solo 'cpu'. MPS necesita Apple Silicon.":
        "This is an Intel Mac: 'cpu' only. MPS requires Apple Silicon.",
    'No se encontró {} en la carpeta de instalación. Descarga la versión nueva manualmente desde GitHub.':
        'Could not find {} in the installation folder. Download the new version manually from GitHub.',
    'No se pudo lanzar {}. Ejecútalo a mano desde la carpeta de instalación.':
        'Could not launch {}. Run it by hand from the installation folder.',
    '<ol><li><b>Resumen</b> con detecciones, confianza media y tamaño medio.</li><li><b>Métodos</b> — modelo, parámetros, calibración y dispositivo.</li><li><b>Calibración de escala</b> — de dónde salió el µm/px de cada foto.</li><li><b>Forma y talla</b> — reparto fibra/fragmento, distribución de tallas, la mayor y la menor, recuento por imagen y cómo se mide el largo.</li><li><b>Talla por carpeta y por foto</b> (opcional, desmarcada de fábrica) — compara el tamaño entre las carpetas del lote y, dentro de cada una, entre sus fotos, con una prueba de significancia. Sirve cuando cada carpeta es un sitio de muestreo distinto.</li><li><b>Fichas</b> — una muestra de partículas con su medición dibujada.</li><li><b>Resultados generales</b> — clases, confianza y tamaños.</li><li><b>Resumen por modelo</b> (tabla comparativa)</li><li><b>Análisis de errores</b> (solo si hay ground truth) — matriz de confusión y P/R/F1 por clase.</li><li><b>Comparación entre modelos</b></li><li><b>Galería por imagen</b> — predicción y ground truth lado a lado.</li><li><b>Conteo por muestra y tipo de plástico</b> — por imagen, por tramo y por estación.</li><li><b>Referencias bibliográficas</b></li></ol><p>Todas las secciones son opcionales: se eligen arriba.</p>':
        '<ol><li><b>Summary</b> with detections, mean confidence and mean size.</li><li><b>Methods</b> — model, parameters, calibration and device.</li><li><b>Scale calibration</b> — where the µm/px of each photo came from.</li><li><b>Shape and size</b> — fibre/particle split, size distribution, the largest and the smallest, per-image counts and how length is measured.</li><li><b>Size by folder and by photo</b> (optional, off by default) — compares size across the folders of the batch and, within each one, across its photos, with a significance test. Useful when each folder is a different sampling site.</li><li><b>Cards</b> — a sample of particles with their measurement drawn on them.</li><li><b>General results</b> — classes, confidence and sizes.</li><li><b>Per-model summary</b> (comparison table)</li><li><b>Error analysis</b> (only with ground truth) — confusion matrix and per-class P/R/F1.</li><li><b>Model comparison</b></li><li><b>Per-image gallery</b> — prediction and ground truth side by side.</li><li><b>Counts by sample and polymer type</b> — per image, per depth section and per station.</li><li><b>References</b></li></ol><p>Every section is optional: you pick them above.</p>',

    # ── Informe de deteccion: tablas y rotulos ──
    '<th>Parámetro</th><th>Valor</th>':
        '<th>Parameter</th><th>Value</th>',
    '<th>Componente</th><th>Detalle</th>':
        '<th>Component</th><th>Detail</th>',
    '<th>Procedencia de la escala</th><th>Imágenes</th>':
        '<th>Source of the scale</th><th>Images</th>',
    '<th>µm por píxel</th><th>Valor</th>':
        '<th>µm per pixel</th><th>Value</th>',
    '<th>Morfotipo</th><th>Cuántas</th><th>%</th>':
        '<th>Morphotype</th><th>How many</th><th>%</th>',
    '<th>Estadístico</th><th>Largo (µm)</th>':
        '<th>Statistic</th><th>Length (µm)</th>',
    '<th>Medida</th><th>µm</th>':
        '<th>Measure</th><th>µm</th>',
    '<th>Forma de la partícula</th><th>Qué se reporta como largo</th>':
        '<th>Particle shape</th><th>What is reported as length</th>',
    '<td>mínimo</td>':
        '<td>minimum</td>',
    '<td>mediana</td>':
        '<td>median</td>',
    '<td>máximo</td>':
        '<td>maximum</td>',
    '<td>percentil 10</td>':
        '<td>10th percentile</td>',
    '<td>percentil 90</td>':
        '<td>90th percentile</td>',
    '<td>media ± IC 95%</td>':
        '<td>mean ± 95% CI</td>',
    '<td>Fibra (relación de aspecto ≥ 3)</td>':
        '<td>Fibre (aspect ratio ≥ 3)</td>',
    '<td>Partícula (no fibrosa)</td>':
        '<td>Particle (non-fibrous)</td>',
    '<td>Compacta o irregular, pero no doblada</td><td>Feret máximo (la recta más larga)</td>':
        '<td>Compact or irregular, but not bent</td><td>Maximum Feret (the longest straight line)</td>',
    '<td>Alargada y contorsionada (fibra)</td><td>Diámetro geodésico (sigue la curva)</td>':
        '<td>Elongated and contorted (fibre)</td><td>Geodesic diameter (follows the curve)</td>',
    'Sistema operativo':
        'Operating system',
    'Procesador':
        'Processor',
    'Memoria RAM':
        'RAM',
    'no disponible':
        'not available',

    # ── Informe: galeria ──
    'Sin Ground Truth para esta imagen':
        'No ground truth for this image',

    # ── Informe: portada, pie y titulos de seccion ──
    'Informe de detección de microplásticos':
        'Microplastics detection report',
    'por fluorescencia Nile Red':
        'by Nile Red fluorescence',
    'Autor:':
        'Author:',
    'Modelos:':
        'Models:',
    'Índice':
        'Contents',
    'Generado por Poly-X':
        'Generated by Poly-X',
    'Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).':
        'Microplastics detection suite by Nile Red fluorescence (254 nm) and AI (YOLO v8/v11).',
    'Talla por carpeta y por foto (comparación entre sitios)':
        'Size by folder and by photo (comparison between sites)',

    # ── Informe: prosa de metodo y calibracion ──
    'Cómo se mide el largo':
        'How length is measured',
    'Un ejemplo: la partícula mayor, medida':
        'An example: the largest particle, measured',
    'Un ejemplo: la partícula menor, medida':
        'An example: the smallest particle, measured',
    'Partícula mayor y menor':
        'Largest and smallest particle',
    'Distribución de tallas':
        'Size distribution',
    'Talla por tipo de plástico':
        'Size by polymer type',
    'Recuento por imagen':
        'Per-image counts',
    'Talla por carpeta':
        'Size by folder',
    'Talla por foto, dentro de cada carpeta':
        'Size by photo, within each folder',
    'Cómo se obtiene la escala, sobre una placa real':
        'How the scale is obtained, on a real dish',
    'Equipo de cómputo':
        'Computing hardware',
    'Entrenamiento de cada modelo':
        'Training of each model',
    'Se reporta la mayor de las dos primeras':
        'The larger of the first two is reported',
    'El patrón de longitud es el diámetro externo nominal de la placa Petri. El radio en píxeles se obtiene ajustando un círculo por mínimos cuadrados al borde del anillo muestreado en 720 direcciones, con rechazo de atípicos; la transformada de Hough solo aporta el centro aproximado, porque su radio llega a errar un 12&nbsp;% y ese error entraría entero en todos los tamaños reportados.':
        'The length standard is the nominal outer diameter of the Petri dish. The radius in pixels comes from least-squares fitting a circle to the rim edge sampled in 720 directions, with outlier rejection; the Hough transform only supplies the approximate centre, because its radius can be off by 12&nbsp;% and that error would carry straight into every reported size.',
    'En verde, la circunferencia que el ajuste encontró para el borde externo del anillo; en amarillo, su diámetro. Ese diámetro es la magnitud conocida':
        'In green, the circle the fit found for the outer edge of the rim; in yellow, its diameter. That diameter is the known quantity',
    'Conviene mirar si el círculo verde sigue el borde del anillo y no otra cosa —la sombra de la placa, un reflejo, o el borde del filtro de dentro—. Si cayera mal, <strong>todas</strong> las tallas de esa foto saldrían escaladas por el mismo factor equivocado, y ninguna otra cifra del informe lo delataría.':
        "It is worth checking that the green circle follows the rim edge and not something else —the dish's shadow, a reflection, or the inner edge of the filter—. If it landed wrong, <strong>every</strong> size in that photo would be scaled by the same wrong factor, and no other figure in the report would reveal it.",

    # ── Informe: tabla de particula mayor y menor ──
    '<th></th><th>Clase</th><th>Largo</th><th>Ancho</th><th>Área</th><th>Aspecto</th><th>Curvatura</th><th>Morfotipo</th><th>Imagen</th>':
        '<th></th><th>Class</th><th>Length</th><th>Width</th><th>Area</th><th>Aspect</th><th>Curvature</th><th>Morphotype</th><th>Image</th>',
    'Mayor':
        'Largest',
    'Menor':
        'Smallest',

    # ── Informe: incertidumbre de la escala (lleva {pared} y {sesgo}) ──
    '<strong>Incertidumbre de la escala, y en qué dirección.</strong> El anillo de la placa tiene una pared de unos <strong>{pared:g}&nbsp;mm</strong> (medido sobre las fotos de este estudio: el borde interno cae en 0,960 del radio ajustado y el externo en 1,000). El diámetro nominal de una placa Petri es ambiguo a ese nivel: puede referirse al diámetro <em>externo</em> o al <em>útil interior</em>. Aquí se toma el <strong>externo</strong>, que es el borde al que ajusta el círculo. Si el nominal se refiriera al interior, la escala correcta sería un <strong>{sesgo:g}&nbsp;% mayor</strong> y todas las tallas de este informe estarían <strong>subestimadas</strong> en esa cifra. El sesgo solo puede ir en ese sentido, porque el externo es el mayor de los dos bordes posibles.':
        '<strong>Uncertainty of the scale, and in which direction.</strong> The rim of the dish has a wall about <strong>{pared:g}&nbsp;mm</strong> thick (measured on the photographs of this study: the inner edge falls at 0.960 of the fitted radius and the outer edge at 1.000). The nominal diameter of a Petri dish is ambiguous at that level: it may refer to the <em>outer</em> diameter or to the <em>usable inner</em> one. Here the <strong>outer</strong> is taken, since that is the edge the circle is fitted to. If the nominal figure referred to the inner edge, the correct scale would be <strong>{sesgo:g}&nbsp;% larger</strong> and every size in this report would be <strong>underestimated</strong> by that amount. The bias can only run in that direction, because the outer edge is the larger of the two possible ones.',

    # -- Informe: metodo de medida del largo --
    '<p>Las magnitudes se miden sobre la <strong>máscara de cada partícula</strong>, no sobre la caja del detector. La caja de una partícula alargada está casi vacía y depende de cómo haya caído: una fibra tumbada en diagonal tiene caja cuadrada, de modo que medir sobre la caja la reportaría como si no fuera fibra, y con una talla equivocada. La máscara se obtiene umbralizando cada recorte a medio camino entre el nivel del fondo —la mediana del anillo que rodea la caja— y el de la partícula —el percentil 90 dentro de ella—.</p>':
        "<p>Every magnitude is measured on <strong>each particle's mask</strong>, not on the detector's box. The box of an elongated particle is almost empty and depends on how it happened to land: a fibre lying diagonally has a square box, so measuring on the box would report it as if it were not a fibre, and with the wrong size. The mask is obtained by thresholding each crop halfway between the background level —the median of the ring surrounding the box— and the particle's own —the 90th percentile inside it—.</p>",
    '<p><strong>El criterio general es la línea recta más larga que cabe en la partícula</strong>, es decir la mayor distancia entre dos puntos de su contorno: el <em>diámetro de Feret máximo</em>. Para una partícula de forma irregular pero no doblada, esa recta es su talla, y tiene dos propiedades que la hacen fiable: no depende de la orientación con que la partícula haya caído, y un borde dentado no la altera.</p>':
        '<p><strong>The general criterion is the longest straight line that fits inside the particle</strong>, that is the greatest distance between two points of its outline: the <em>maximum Feret diameter</em>. For a particle of irregular but unbent shape, that line is its size, and it has two properties that make it reliable: it does not depend on the orientation the particle happened to land in, and a jagged edge does not disturb it.</p>',
    '<p><strong>Esa recta deja de servir cuando la partícula está contorsionada.</strong> En una fibra doblada la distancia entre sus extremos es la cuerda, no su longitud: en un arco de media circunferencia se queda un 35&nbsp;% corta. Para esos casos se mide el <em>diámetro geodésico</em>, que es el camino más largo que cabe <strong>dentro</strong> de la partícula. Al no poder salirse de la máscara, ese camino rodea la curva y devuelve la longitud recorrida.</p>':
        '<p><strong>That straight line stops working when the particle is contorted.</strong> In a bent fibre the distance between its ends is the chord, not its length: on a half-circle arc it falls 35&nbsp;% short. For those cases the <em>geodesic diameter</em> is measured, which is the longest path that fits <strong>inside</strong> the particle. Since it cannot leave the mask, that path follows the curve and returns the distance travelled.</p>',
    '<p>La distinción se hace por el <strong>grosor</strong>: solo se usa el geodésico si el largo supera al menos cuatro veces el grosor máximo inscrito de la partícula. La razón es que en una partícula gruesa cualquier concavidad obliga al camino geodésico a bordearla en vez de atravesarla, y entonces el número se infla; en una fibra delgada eso no puede ocurrir.</p>':
        "<p>The distinction is made by <strong>thickness</strong>: the geodesic is used only if the length exceeds at least four times the particle's maximum inscribed thickness. The reason is that in a thick particle any concavity forces the geodesic path to skirt around it instead of crossing it, and the number is then inflated; in a thin fibre that cannot happen.</p>",
    '<p>Contrastado con formas de talla conocida —rectas, rectas giradas 45&nbsp;°, arcos de 60, 120 y 180&nbsp;°, un círculo y una recta de borde dentado— el largo así medido da un <strong>error mediano del 0,8&nbsp;% y del 4,7&nbsp;% en el peor caso</strong>.</p>':
        '<p>Checked against shapes of known size —straight bars, bars rotated 45&nbsp;°, arcs of 60, 120 and 180&nbsp;°, a circle and a bar with a jagged edge— the length measured this way gives a <strong>median error of 0.8&nbsp;% and 4.7&nbsp;% in the worst case</strong>.</p>',
    '<p><em>Partículas en contacto.</em> Dos partículas que se tocan forman una sola mancha, y medirlas juntas sumaría sus tallas. Se separan por <em>watershed</em> sobre la transformada de distancia: el centro de cada una queda lejos del fondo y el cuello que las une queda cerca, de modo que el corte cae por el cuello. Sobre círculos de talla conocida las separa hasta un 27&nbsp;% de solapamiento del diámetro, sin partir ninguna partícula de una sola pieza.</p>':
        '<p><em>Touching particles.</em> Two particles in contact form a single blob, and measuring them together would add their sizes up. They are separated by <em>watershed</em> on the distance transform: the centre of each one sits far from the background while the neck joining them sits close, so the cut falls on the neck. Against circles of known size it separates them up to 27&nbsp;% overlap of the diameter, without splitting any particle that is a single piece.</p>',
    '<p><em>Limitaciones conocidas.</em> Dos partículas solapadas más allá de un 40&nbsp;% de su diámetro se siguen midiendo como una sola: a esa altura ya no hay un cuello por el que cortar. Y en una fibra muy enroscada el camino geodésico ataja por el interior de cada codo, subestimando la longitud hasta un 19&nbsp;% en el caso más cerrado que se ensayó.</p>':
        '<p><em>Known limitations.</em> Two particles overlapping by more than 40&nbsp;% of their diameter are still measured as one: past that point there is no neck left to cut along. And in a tightly coiled fibre the geodesic path cuts the corner at each bend, underestimating the length by up to 19&nbsp;% in the tightest case tested.</p>',
    '<p><strong>{n} partículas ({pct} %) están contorsionadas</strong>, entendiendo por tal que su largo supera en más de un 15&nbsp;% su extensión en línea recta.</p>':
        '<p><strong>{n} particles ({pct} %) are contorted</strong>, meaning that their length exceeds their straight-line extent by more than 15&nbsp;%.</p>',

    # -- Informe: rotulos de figuras, tablas y secciones --
    'Umbral de confianza':
        'Confidence threshold',
    'F1 (con clase)':
        'F1 (with class)',
    'Valor':
        'Value',
    'F1 por clase':
        'F1 per class',
    'Distribución por clase':
        'Detections per class',
    'Confianza':
        'Confidence',
    'Histograma de confianza':
        'Confidence histogram',
    'Diámetro equivalente (μm)':
        'Equivalent diameter (μm)',
    'Distribución de tamaños':
        'Size distribution',
    'talla (µm, dimensión mayor siguiendo la curva)':
        'size (µm, longest dimension following the curve)',
    'partículas':
        'particles',
    'talla (µm, escala log)':
        'size (µm, log scale)',
    'Predicción':
        'Prediction',
    'Matriz de confusión':
        'Confusion matrix',
    'Modelos cargados':
        'Models loaded',
    'Confianza mínima':
        'Minimum confidence',
    'IoU para emparejar Verdaderos Positivos':
        'IoU for matching true positives',
    'Tamaño de imagen (imgsz)':
        'Image size (imgsz)',
    'Dispositivo':
        'Device',
    'μm por píxel':
        'μm per pixel',
    'Filtro tamaño (μm)':
        'Size filter (μm)',
    'sin filtro':
        'no filter',
    'Imágenes procesadas':
        'Images processed',
    'Total de detecciones':
        'Total detections',
    'medida sobre la placa Petri de esta foto':
        "measured on this photograph's Petri dish",
    'heredada del recorte (índice de calibración)':
        'inherited from the crop (calibration index)',
    'introducida a mano en Parámetros':
        'entered by hand in Parameters',
    'Partícula':
        'Particle',
    'Estación':
        'Station',
    'la talla puede estar inflada':
        'the size may be inflated',
    'sin diferencia significativa':
        'no significant difference',

    # -- Informe: prosa de conteo, comparacion y fichas --
    '<tr><th>Polímero</th><th>Conteo manual</th><th>Detectadas por el modelo</th><th>Diferencia</th></tr>':
        '<tr><th>Polymer</th><th>Manual count</th><th>Detected by the model</th><th>Difference</th></tr>',
    '<tr><th>Polímero</th><th>Detectadas por el modelo</th></tr>':
        '<tr><th>Polymer</th><th>Detected by the model</th></tr>',
    '<p>Configuración y métricas de validación con que se entrenó cada peso, leídas del propio archivo <code>.pt</code>.</p>':
        '<p>Validation settings and metrics each weight was trained with, read from the <code>.pt</code> file itself.</p>',
    '<tr><th>Modelo</th><th>Arquitectura base</th><th>imgsz</th><th>batch</th><th>épocas</th><th>optimizador</th><th>Precisión</th><th>Recall</th><th>mAP@50</th><th>mAP@50-95</th></tr>':
        '<tr><th>Model</th><th>Base architecture</th><th>imgsz</th><th>batch</th><th>epochs</th><th>optimiser</th><th>Precision</th><th>Recall</th><th>mAP@50</th><th>mAP@50-95</th></tr>',
    'Kruskal-Wallis entre los {n} grupos: {frase} (H={h}, gl={gl}, p={p}). No dice CUÁL grupo difiere de cuál, solo que no todos comparten la misma distribución de talla.':
        'Kruskal-Wallis across the {n} groups: {frase} (H={h}, df={gl}, p={p}). It does not say WHICH group differs from which, only that not all of them share the same size distribution.',
    '<strong>diferencia significativa</strong>':
        '<strong>significant difference</strong>',
    ' (las {n} con más partículas)':
        ' (the {n} with the most particles)',
    ', en este caso por <strong>{metodo}</strong>':
        ', here by <strong>{metodo}</strong>',
    '<p>A la izquierda la partícula; a la derecha, en verde, el contorno que se midió sobre ella.</p>':
        '<p>On the left the particle; on the right, in green, the outline that was measured on it.</p>',
    'imágenes':
        'images',
    'imagen':
        'image',
    '<p>Las placas del mismo tramo se <strong>suman</strong>: son submuestras de la misma masa de sedimento, no repeticiones fotográficas. El tramo es la unidad de análisis.</p>':
        '<p>Dishes from the same depth interval are <strong>added together</strong>: they are subsamples of the same mass of sediment, not repeated photographs. The interval is the unit of analysis.</p>',
    'Por tramo de profundidad':
        'By depth interval',
    'Tramo':
        'Interval',
    'Por estación':
        'By station',
    '{n} imagen(es) no siguen la nomenclatura <code>tramo.testigo</code> y quedan fuera de las tablas agrupadas; sí están en la tabla por imagen.':
        '{n} image(s) do not follow the <code>interval.core</code> naming and are left out of the grouped tables; they are still in the per-image table.',
    '<p>Partículas contadas en cada muestra, desglosadas por polímero. La columna <em>manual</em> es la anotación humana (Ground Truth) y la columna <em>modelo</em> son todas las detecciones de <strong>{alias}</strong>{coletilla}</p>':
        '<p>Particles counted in each sample, broken down by polymer. The <em>manual</em> column is the human annotation (ground truth) and the <em>model</em> column is every detection made by <strong>{alias}</strong>{coletilla}</p>',
    ', el primer modelo activo.':
        ', the first active model.',
    '<p><strong>Léase como conteo, no como evaluación.</strong> La columna del modelo no descuenta falsos positivos ni empareja caja a caja: es cuántas partículas de cada polímero reportó. Coincidir en el total no implica haber acertado partícula por partícula; para eso está el análisis de errores.</p>':
        "<p><strong>Read it as a count, not as an evaluation.</strong> The model's column does not subtract false positives nor match box to box: it is how many particles of each polymer it reported. Agreeing on the total does not imply getting each particle right; that is what the error analysis is for.</p>",

    # -- Informe: errores, comparacion entre modelos y galeria --
    'Sin fila para {clases}: no aparece ni en la anotación manual ni entre las predicciones de este lote, de modo que no hay métrica que informar. El modelo sí está entrenado para esa clase.':
        "No row for {clases}: it appears neither in the manual annotation nor among this batch's predictions, so there is no metric to report. The model is trained for that class all the same.",
    'Figura. Matriz de confusión (modelo principal: {modelo}, IoU = {iou}).':
        'Figure. Confusion matrix (main model: {modelo}, IoU = {iou}).',
    'Precisión / Recall / F1 por clase':
        'Precision / recall / F1 per class',
    '<th>Clase</th>':
        '<th>Class</th>',
    '<th>Precisión</th><th>Recall</th><th>F1</th>':
        '<th>Precision</th><th>Recall</th><th>F1</th>',
    '<p><strong>Los dos F1 miden cosas distintas.</strong> <em>Localización</em> responde si el detector encuentra la partícula (P {pl} · R {rl} · <strong>F1 {fl}</strong>). <em>Con clase</em> exige además acertar el polímero, contando cada caja mal clasificada como falso positivo de la clase predicha y falso negativo de la real (P {pc} · R {rc} · <strong>F1 {fc}</strong>).</p><p>La diferencia corresponde a <strong>{mc}</strong> partícula(s) bien localizada(s) pero asignada(s) a la clase incorrecta. Es la cifra que concilia esta tabla con la de precisión por clase de la sección de errores.</p>':
        '<p><strong>The two F1 figures measure different things.</strong> <em>Localisation</em> answers whether the detector finds the particle (P {pl} · R {rl} · <strong>F1 {fl}</strong>). <em>With class</em> also requires getting the polymer right, counting each misclassified box as a false positive of the predicted class and a false negative of the real one (P {pc} · R {rc} · <strong>F1 {fc}</strong>).</p><p>The difference corresponds to <strong>{mc}</strong> particle(s) located correctly but assigned to the wrong class. It is the figure that reconciles this table with the per-class precision in the error section.</p>',
    '<p><strong>Mejor desempeño: {alias}</strong>, con F1 {f1} al umbral {umbral} (P {p} · R {r}).</p>':
        '<p><strong>Best performance: {alias}</strong>, with F1 {f1} at threshold {umbral} (P {p} · R {r}).</p>',
    '<p>La diferencia con {otro} es de <strong>{d}</strong> de F1. Con un solo entrenamiento por arquitectura, una diferencia pequeña no distingue el diseño de la red del azar de inicialización: haría falta repetir con distintas semillas para afirmar que una es superior.</p>':
        '<p>The gap against {otro} is <strong>{d}</strong> of F1. With a single training run per architecture, a small difference does not tell the network design apart from initialisation luck: it would take repeating with different seeds to claim one is better.</p>',
    '<p><b>Mejor F1 de localización:</b> {alias} ({f1}) — encontrar la partícula, sin exigir que acierte el polímero. El veredicto con clase está más arriba.</p>':
        '<p><b>Best localisation F1:</b> {alias} ({f1}) — finding the particle, without requiring the polymer to be right. The with-class verdict is further up.</p>',
    'Sin ground truth no se puede declarar un ganador: un modelo con más detecciones puede estar acertando o inventando. Carga anotaciones para obtener F1 por modelo.':
        'Without ground truth no winner can be declared: a model with more detections may be getting them right or making them up. Load annotations to obtain F1 per model.',
    'Figura. Precisión, Recall y F1 de cada modelo al umbral {umbral}, en los dos criterios. La distancia entre paneles es la confusión entre polímeros.':
        'Figure. Precision, recall and F1 of each model at threshold {umbral}, under both criteria. The gap between panels is the confusion between polymers.',
    'Figura. F1 por clase. Un modelo puede ganar en el agregado y perder en el polímero que interesa.':
        'Figure. F1 per class. A model can win overall and lose on the polymer that matters.',
    'Figura. Detecciones por imagen de un modelo frente al otro; la diagonal es el acuerdo exacto. Escala simétrica-logarítmica, porque casi todas las placas tienen pocas partículas y unas pocas cientos.':
        'Figure. Detections per image of one model against the other; the diagonal is exact agreement. Symmetric-log scale, because almost every dish has few particles and a handful have hundreds.',
    'Figura. F1 (con clase) frente al umbral de confianza; la estrella marca el máximo de cada modelo. La curva arranca en {umbral}, el umbral con que se ejecutó: por debajo las detecciones no se calcularon.':
        "Figure. F1 (with class) against the confidence threshold; the star marks each model's maximum. The curve starts at {umbral}, the threshold it was run with: below that the detections were never computed.",
    '<p>Detecciones de cada modelo imagen por imagen. El total puede estar dominado por unas pocas fotos densas, así que conviene mirar el detalle antes de elegir modelo.</p>':
        "<p>Each model's detections image by image. The total can be dominated by a few dense photographs, so it is worth looking at the detail before choosing a model.</p>",
    'Solo se ejecutó un modelo, así que no hay comparación. Carga un segundo modelo en la pestaña Modelos para compararlos sobre las mismas imágenes.':
        'Only one model was run, so there is no comparison. Load a second model in the Models tab to compare them over the same images.',

    # -- Informe: calibracion, talla por carpeta y avisos --
    '<p><strong>La escala no es común a todo el lote:</strong> varía {var}× entre la foto más cercana y la más lejana. Por eso cada imagen se convierte con su propio factor; usar uno solo para todas daría tamaños con hasta un {pct}% de error.</p>':
        '<p><strong>The scale is not shared by the whole batch:</strong> it varies {var}× between the nearest and the farthest photograph. That is why each image is converted with its own factor; using a single one for all of them would give sizes off by as much as {pct}%.</p>',
    'Incidencias durante la calibración:':
        'Issues during calibration:',
    '<p>Escala única para todo el lote, introducida a mano: <strong>{um} µm/píxel</strong>. No se midió ninguna placa, de modo que este factor no está trazado a un patrón de longitud y cualquier variación en la distancia de disparo entre fotos queda sin corregir.</p>':
        '<p>A single scale for the whole batch, entered by hand: <strong>{um} µm/pixel</strong>. No dish was measured, so this factor is not traced to a length standard and any variation in shooting distance between photographs goes uncorrected.</p>',
    '<strong>{n} de {total} partículas ({pct}&nbsp;%) tienen una talla que pide comprobación</strong>: miden bastante más que la diagonal de su caja, lo que casi siempre significa varias partículas en contacto medidas como una sola. La talla de esas queda <em>sobreestimada</em>. Van marcadas con ⚠ en sus fichas.':
        "<strong>{n} of {total} particles ({pct}&nbsp;%) have a size that needs checking</strong>: they measure considerably more than their box's diagonal, which almost always means several touching particles measured as one. Those sizes are <em>overestimated</em>. They are marked with ⚠ on their cards.",
    'carpeta de origen':
        'source folder',
    '<p>Compara la talla entre las carpetas del lote analizado: cada carpeta como un grupo distinto. Útil cuando cada carpeta es un sitio de muestreo, una estación o una condición.</p>':
        '<p>Compares size across the folders of the analysed batch, each folder as a separate group. Useful when each folder is a sampling site, a station or a condition.</p>',
    'Talla por foto — {carpeta}':
        'Size per photograph — {carpeta}',
    'foto':
        'photograph',
    '({n} fotos)':
        '({n} photographs)',
    ' Se muestran las {n} carpetas con más partículas de las {total} que tienen más de una foto.':
        ' The {n} folders with the most particles are shown, out of the {total} that have more than one photograph.',
    '<p>Compara la talla entre las fotos individuales de una misma carpeta: por ejemplo, si el tamaño cambia con la profundidad o el momento de muestreo dentro de un mismo sitio.{nota}</p>':
        '<p>Compares size across the individual photographs of one folder: for instance, whether size changes with depth or with the sampling moment within the same site.{nota}</p>',

    # -- Informe: galeria comparativa --
    'Se muestran las primeras {n} imágenes, cada una con todos los modelos; {fuera} quedaron fuera de la galería para que el archivo siga siendo manejable. Las métricas de las secciones anteriores sí incluyen todas.':
        'The first {n} images are shown, each with every model; {fuera} were left out of the gallery to keep the file manageable. The metrics in the earlier sections do cover all of them.',
    '<p>Cada bloque muestra, a la izquierda, las detecciones del modelo (<em>bounding boxes</em> dibujadas por YOLO con su clase y confianza) y, a la derecha, las etiquetas reales de control (<em>Ground Truth</em>). Esta vista lado a lado permite evaluar visualmente dónde acertó o falló el modelo.</p>':
        "<p>Each block shows, on the left, the model's detections (<em>bounding boxes</em> drawn by YOLO with their class and confidence) and, on the right, the real control labels (<em>ground truth</em>). This side-by-side view lets you judge visually where the model got it right or wrong.</p>",
    'Frecuencia':
        'Frequency',
    'Feret máximo (cuerda)':
        'Maximum Feret (chord)',
    'Geodésico (sigue la curva)':
        'Geodesic (follows the curve)',
    'Sin datos suficientes para una prueba estadística (hace falta más de una observación por grupo).':
        'Not enough data for a statistical test (more than one observation per group is required).',

    # -- Informe: resumen, fichas y recuento por imagen --
    'Localización':
        'Localisation',
    'Con clase':
        'With class',
    ' y de ahí sale todo lo demás.':
        ' and everything else follows from it.',
    '{n} detección(es) por YOLO':
        '{n} detection(s) by YOLO',
    'Figura. Distribución de detecciones por clase.':
        'Figure. Distribution of detections per class.',
    'Figura. Histograma de confianza.':
        'Figure. Confidence histogram.',
    'Figura. Distribución de tamaños (diámetro equivalente en μm).':
        'Figure. Size distribution (equivalent diameter in μm).',
    '(una sola foto calibrada, sin IC)':
        '(a single calibrated photograph, no CI)',
    'La detección automatizada se realizó con el modelo YOLO «{modelos}» (Ultralytics {ul}) a una resolución de entrada de {imgsz} px, umbral de confianza {conf} y supresión de no-máximos con IoU {iou}. ':
        'Automated detection was carried out with the YOLO model «{modelos}» (Ultralytics {ul}) at an input resolution of {imgsz} px, confidence threshold {conf} and non-maximum suppression at IoU {iou}. ',
    'Resumen de la configuración empleada:':
        'Summary of the settings used:',
    'Las métricas de error se calcularon contra anotación manual independiente, emparejando predicciones y etiquetas con IoU ≥ {iou}. ':
        'Error metrics were computed against independent manual annotation, matching predictions and labels at IoU ≥ {iou}. ',
    'La calibración óptica fue de {um} μm/píxel. ':
        'Optical calibration was {um} μm/pixel. ',
    'Se procesaron {imgs} imágenes con un total de {dets} detecciones.':
        '{imgs} images were processed with a total of {dets} detections.',
    '<th>Imagen</th><th>Total</th><th>Fibras</th><th>Partículas</th>':
        '<th>Image</th><th>Total</th><th>Fibres</th><th>Particles</th>',
    '<th>Largo mediano<br>(µm)</th><th>Mayor<br>(µm)</th>':
        '<th>Median length<br>(µm)</th><th>Largest<br>(µm)</th>',
    '<strong>{n} de {total} imágenes no tienen escala</strong> ({cuales})':
        '<strong>{n} of {total} images have no scale</strong> ({cuales})',
    '<strong>Ninguna imagen tiene escala</strong>':
        '<strong>No image has a scale</strong>',
    ', y por eso su largo aparece como «—». Las partículas <em>sí</em> se midieron —el conteo y la forma de la tabla son correctos—, pero una medida en píxeles no se puede pasar a micrómetros sin saber cuántos µm mide cada píxel.</p>':
        ', which is why their length shows as «—». The particles <em>were</em> measured —the counts and the shape columns are correct— but a measurement in pixels cannot be turned into micrometres without knowing how many µm each pixel is.</p>',
    '<p>Para obtenerlo, en <em>Parámetros</em>: activa <strong>«Medir la placa Petri»</strong> —cada foto obtiene su propia escala del anillo de la placa, que es lo más fiable— o escribe un valor de <strong>µm/píxel</strong>':
        "<p>To get one, in <em>Parameters</em>: tick <strong>«Measure the Petri dish»</strong> —each photograph then gets its own scale from the dish's rim, which is the most reliable route— or type a <strong>µm/pixel</strong> value",
    '. Si ya está activo, en esas fotos concretas no se encontró el anillo de la placa: revisa que se vea entero en el encuadre.':
        ". If it is already on, then in those particular photographs the dish's rim was not found: check that it is fully inside the frame.",
    '<p>En {n} partículas no se pudo separar la partícula del fondo; su talla proviene de la caja y no es comparable con el resto.</p>':
        '<p>In {n} particles the particle could not be separated from the background; their size comes from the box and is not comparable with the rest.</p>',
    'medido por {metodo}':
        'measured by {metodo}',
    'Se muestran las <strong>{fibras} fibra(s) y las {otras} partículas más grandes</strong>':
        'The <strong>{fibras} fibre(s) and the {otras} largest particles</strong> are shown',
    'Se muestran las <strong>{otras} partículas más grandes</strong>. En este lote no se detectó ninguna fibra, así que la muestra no incluye ninguna':
        'The <strong>{otras} largest particles</strong> are shown. No fibre was detected in this batch, so the sample includes none',
    '<p>Cada partícula con el número que lleva en la imagen anotada, su recorte y la medida dibujada <strong>sobre ella</strong>: en amarillo la recta de Feret, en magenta el camino geodésico, según cuál de las dos haya decidido su talla. El contorno verde es la máscara que se midió. A la izquierda va la partícula sin marcas, para poder juzgar si el contorno la sigue.</p>':
        '<p>Each particle with the number it carries in the annotated image, its crop and the measurement drawn <strong>on it</strong>: in yellow the Feret line, in magenta the geodesic path, whichever of the two decided its size. The green outline is the mask that was measured. On the left the particle without marks, so you can judge whether the outline follows it.</p>',
    '. El reparto es deliberado: las fibras son el caso donde actúa el método geodésico —y son minoría, de modo que una muestra tomada al azar podría no enseñar ninguna—, y la mayor de cada tipo es la que sostiene cualquier afirmación sobre talla máxima. <strong>No es una muestra representativa del lote</strong>, sino la selección que permite comprobar el método donde más puede fallar.</p>':
        '. The split is deliberate: fibres are the case where the geodesic method comes into play —and they are a minority, so a sample taken at random might show none— and the largest of each type is what any claim about maximum size rests on. <strong>It is not a representative sample of the batch</strong>, but the selection that lets the method be checked where it can fail most.</p>',
    'Se incluyó análisis de errores con Ground Truth (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Mal Clasificados).':
        'Error analysis against ground truth was included (true positives, false positives, false negatives and misclassified).',
    'No se aportó Ground Truth, por lo que no se reportan métricas de error.':
        'No ground truth was supplied, so no error metrics are reported.',
    'Conf. media':
        'Mean conf.',
    'Se analizaron <strong>{imgs} imágenes</strong> con <strong>{n} {modelos}</strong> YOLO entrenado para detectar microplásticos de PET, PP y LDPE bajo fluorescencia Nile Red (254 nm). El total de detecciones fue <strong>{dets}</strong> con una confianza media de <strong>{conf}</strong>.':
        '<strong>{imgs} images</strong> were analysed with <strong>{n} YOLO {modelos}</strong> trained to detect PET, PP and LDPE microplastics under Nile Red fluorescence (254 nm). Detections totalled <strong>{dets}</strong> with a mean confidence of <strong>{conf}</strong>.',
    'modelo':
        'model',
    'modelos':
        'models',
    '<th>Modelo</th><th>Imágenes</th><th>Detecciones</th><th>Conf. media</th>':
        '<th>Model</th><th>Images</th><th>Detections</th><th>Mean conf.</th>',
    'localización':
        'localisation',
    'con clase':
        'with class',

    # -- Informe: morfotipos --
    'Fibra':
        'Fibre',

    # -- Informe: dos rotulos que quedaban --
    'Precisión':
        'Precision',
    'Predicción del modelo':
        'Model prediction',

    # -- Informe: nombre del metodo de medida y puntos del borde --
    '{n} puntos de borde':
        '{n} edge points',
    'Feret maximo':
        'maximum Feret',
    'geodesico':
        'geodesic',

    # -- Informe: detecciones sobre el anillo de la placa --
    '<strong>{n} detección(es) caen sobre el anillo de la placa</strong> —más allá del {pct} % del radio ajustado— y quedan fuera de esta sección: el aro es una banda brillante que el detector confunde con material, y como sale enorme desplazaría a las partículas de verdad del extremo grande. Siguen contadas en la sección de conteo.':
        "<strong>{n} detection(s) fall on the dish's rim</strong> —beyond {pct} % of the fitted radius— and are left out of this section: the rim is a bright band the detector mistakes for material, and being huge it would displace the real particles at the large end. They are still counted in the counting section.",

    # -- Informe: la recta de Feret en particulas concavas --
    '{pct} % de la recta fuera del contorno':
        '{pct} % of the line outside the outline',
    '<p>En una partícula <strong>cóncava</strong> la recta amarilla cruza por fuera del contorno, y eso es correcto: Feret es la separación de dos mordazas de calibre, no un camino por dentro de la partícula. Para que se vea, el tramo que cae fuera va <strong>a trazos</strong>. El camino que sí va por dentro es el geodésico, en magenta, y se usa cuando la partícula es delgada y está doblada.</p>':
        '<p>On a <strong>concave</strong> particle the yellow line crosses outside the outline, and that is correct: Feret is the separation of two caliper jaws, not a path through the particle. So that this is visible, the stretch that falls outside is <strong>dashed</strong>. The path that does stay inside is the geodesic, in magenta, used when the particle is thin and bent.</p>',

    # -- Informe: perfil en profundidad del testigo --
    'Perfil en profundidad del testigo':
        'Depth profile of the core',
    'crece':
        'increases',
    'decrece':
        'decreases',
    'el número de partículas':
        'the number of particles',
    'la talla mediana':
        'the median size',
    '<strong>{que} {sentido} con la profundidad</strong>':
        '<strong>{que} {sentido} with depth</strong>',
    'No se detecta tendencia en {que} con la profundidad':
        'No trend in {que} with depth is detected',
    '{veredicto} (Spearman ρ = {rho}, p {p}, n = {n} tramos).':
        '{veredicto} (Spearman ρ = {rho}, p = {p}, n = {n} intervals).',
    'partículas por tramo':
        'particles per interval',
    'talla mediana (µm)  ·  banda = rango intercuartílico':
        'median size (µm)  ·  band = interquartile range',
    'tramo (mayor = más profundo)':
        'interval (higher = deeper)',
    '<p>El tramo es la <strong>profundidad</strong> a la que se tomó el sedimento, y a diferencia de una carpeta cualquiera está <em>ordenado</em>. Eso permite preguntar por una tendencia —si hay más partículas abajo que arriba, o si las de abajo son más grandes— y no solo si los grupos difieren entre sí.</p>':
        '<p>The interval is the <strong>depth</strong> the sediment was taken from and, unlike an arbitrary folder, it is <em>ordered</em>. That makes it possible to ask about a trend —whether there are more particles deeper down, or whether the deeper ones are larger— and not merely whether the groups differ.</p>',
    "<p class='caption' style='text-align:left'>La correlación se calcula sobre los <strong>tramos</strong>, no sobre las partículas: dos partículas de la misma placa no son observaciones independientes de la profundidad, y contarlas como tales inflaría cualquier significación. El valor de p sale de barajar la serie 20 000 veces, no de una aproximación que con tan pocos tramos no valdría.</p>":
        "<p class='caption' style='text-align:left'>The correlation is computed over the <strong>intervals</strong>, not over the particles: two particles from the same dish are not independent observations of depth, and counting them as such would inflate any significance. The p value comes from shuffling the series 20,000 times, not from an approximation that would not hold with so few intervals.</p>",
    '<th>Estación</th><th>Tramo</th><th>Placas</th><th>Partículas</th><th>Con talla</th><th>Mediana (µm)</th><th>Mayor (µm)</th>':
        '<th>Station</th><th>Interval</th><th>Dishes</th><th>Particles</th><th>With size</th><th>Median (µm)</th><th>Largest (µm)</th>',

    # -- Informe: aviso de mascara recortada --
    '<strong>{n} de {total} partículas ({pct}&nbsp;%) tienen una talla que pide comprobación</strong>: al segmentarlas, la máscara se salía bastante más allá de la caja del detector, casi siempre porque el umbral enganchó una partícula vecina o una zona brillante del fondo. En esos casos la máscara se <em>recorta a la caja</em>, de modo que la talla reportada no está inflada; lo que puede es quedarse <em>corta</em>, si la partícula seguía de verdad más allá de la caja. Van marcadas con ⚠ en sus fichas.':
        "<strong>{n} of {total} particles ({pct}&nbsp;%) have a size that needs checking</strong>: when segmenting them, the mask ran well beyond the detector's box, almost always because the threshold caught a neighbouring particle or a bright patch of background. In those cases the mask is <em>clipped to the box</em>, so the reported size is not inflated; what it may be is <em>short</em>, if the particle really did continue beyond the box. They are marked with ⚠ on their cards.",

    # -- Informe: aviso corto de la ficha --
    'la máscara se salía de la caja y se recortó a ella':
        'the mask ran outside the box and was clipped to it',
    # -- Abstencion: hasta donde se sostiene el polimero --
    'no asignable':
        'unassignable',
    'Asignación de polímero (opcional)':
        'Polymer assignment (optional)',
    'Confianza mínima para asignar:':
        'Minimum confidence to assign:',
    'Confianza mínima para asignar polímero':
        'Minimum confidence to assign a polymer',
    'Partículas sin polímero asignable':
        'Particles with no assignable polymer',
    '0 = desactivado. Por debajo de este valor la partícula se cuenta igual, pero se reporta como «no asignable» en vez de atribuirle un polímero. El Nile Red responde a la polaridad del entorno, no a la identidad química: separa bien el PET —poliéster— del resto, mientras que PP y LDPE son las dos poliolefinas y se distinguen solo por brillo, que depende de la exposición y del foco. Abstenerse es más honesto que inventar la clase, y el informe declara qué porcentaje quedó sin asignar.':
        '0 = off. Below this value the particle is still counted, but reported as '
        '"unassignable" instead of being given a polymer. Nile Red responds to the '
        'polarity of its environment, not to chemical identity: it separates PET '
        '—a polyester— cleanly from the rest, while PP and LDPE are both polyolefins '
        'and differ only in brightness, which depends on exposure and focus. '
        'Abstaining is more honest than inventing the class, and the report states '
        'what percentage was left unassigned.',
}
