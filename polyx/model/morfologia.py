"""Mide la particula de verdad, no su caja.

La caja de un detector son cuatro numeros y no describe la forma. Para una
particula alargada tumbada en diagonal la caja esta casi vacia: una fibra de
2000x20 um a 45 grados tiene una caja de 1414x1414, cuya area es 50 veces la de
la fibra. Y lo peor no es la magnitud sino que **la misma particula rotada da
otro numero**, con lo que la medida no es una medida.

Aqui se segmenta dentro de la caja -- que ya dice donde mirar, asi que no hace
falta anotar mascaras ni reentrenar nada -- y las magnitudes salen de la mascara.

COMO SE MIDE EL LARGO
---------------------
Se calculan dos cotas y se reporta la mayor:

  * FERET MAXIMO. La mayor distancia entre dos puntos del borde, sobre la
    envolvente convexa. Es la medida correcta del largo aparente y de una fibra
    RECTA, y no la afecta un borde dentado. Pero en una fibra curva da la
    cuerda: en un arco de 180 grados se queda un 35% corto.

  * DIAMETRO GEODESICO. El camino mas largo que cabe DENTRO de la particula,
    por doble propagacion. Al no poder salirse de la mascara, rodea la curva y
    devuelve su longitud. En una particula compacta el camino mas largo es el
    recto y coincide con Feret.

El geodesico solo se usa si la particula es DELGADA (largo/grosor >= 4) y NO
CONVEXA (solidez < 0.90). Las dos condiciones salen de fallos observados:

  * En una particula gruesa, cualquier concavidad hace que el camino la RODEE
    en vez de atravesarla. Un grumo real de 44 px de extension con una muesca
    recibia 73 px.
  * En una forma convexa el camino mas largo por dentro es el recto, asi que
    geodesico y Feret coinciden por definicion. Si el geodesico sale mayor es
    error de discretizacion del chamfer, que no es isotropo: una barra recta
    girada media 200 px a 0 y 45 grados pero 208 a 15, 30 y 60 -- justo la
    dependencia de la orientacion que se querIa eliminar.

Con las dos condiciones el error mediano contra formas de talla conocida baja a
0.6% y el peor caso queda en 4.7%.

QUE SE DESCARTO, Y POR QUE
--------------------------
El rectangulo de igual area y perimetro, L = (P + sqrt(P^2-16A))/4, se uso como
talla y NO sirve para eso, aunque se conserva como descriptor:

  * Depende del perimetro, y un borde dentado lo infla. Medido sobre una recta
    con dientes de sierra de talla conocida: +22.5%.
  * Solo existe si P^2 >= 16A. Un circulo da P^2 = 4*pi*A < 16A y no tiene
    rectangulo equivalente.
  * No es "el largo de la particula" sino el de un rectangulo con su misma area
    y su mismo perimetro, que es otra cosa.

Se conserva en ``largo_rect_eq`` porque comparado con Feret y con el geodesico
informa de lo irregular que es la particula: si es mucho mayor que los otros
dos, el borde esta dentado.

El modelo de rectangulo daba 1.2% de error mediano pero 22.5% en el peor caso,
y aqui importa mas no fallar feo que afinar en el caso bueno.

Todo esto esta fijado en tests/test_morfologia.py contra formas de talla
conocida, incluida la barra girada que destapo lo de la convexidad.

PARTICULAS EN CONTACTO
----------------------
Dos particulas que se tocan forman una sola componente conexa. Se separan por
watershed sobre la transformada de distancia en ``separar_pegadas()``, hasta un
27% de solapamiento del diametro.

LIMITES CONOCIDOS
-----------------
  * Por encima de un 40% de solapamiento ya no hay cuello por el que cortar y
    las dos particulas se siguen midiendo juntas.
  * En una fibra muy enroscada el camino geodesico corta por dentro en cada codo
    y subestima: en una fibra en ese de curvas cerradas se queda un 19% corto.
    ``largo_rect_eq`` acierta mas en ese caso concreto (-5%), asi que si las dos
    cifras discrepan mucho la particula pide un vistazo.

Solo se usan cv2 y numpy. Ni scipy -- que no es dependencia declarada y solo
llega de rebote con ultralytics -- ni skimage ni opencv-contrib: anadir
dependencias es justo lo que rompe la instalacion en otra maquina.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Morfotipos. Umbral en 3 sobre el aspecto REAL (no el de la caja): es el corte
# que se usa habitualmente para separar fibra de fragmento.
FIBRA = "fibra"
FRAGMENTO = "fragmento"
INDETERMINADO = "indeterminado"

ASPECTO_FIBRA = 3.0

# El diametro geodesico solo se usa si la particula es al menos asi de delgada,
# medida como largo/grosor. En una particula gruesa cualquier concavidad hace
# que el camino geodesico la RODEE en vez de atravesarla, y entonces infla: se
# vio en una particula real de 44 px de extension y 22 de grosor a la que el
# geodesico le daba 73 px por bordear una muesca.
#
# El umbral no es delicado: sobre formas de talla conocida, los grumos quedan en
# 1.05 y 1.64 de delgadez y la fibra mas gorda en 6.7, asi que cualquier valor
# entre 3 y 6 separa igual. Con el, el error mediano del largo baja de 3.1% a
# 0.8% -- un circulo pasa de 104 a 100 exacto -- sin tocar ninguna fibra.
DELGADEZ_PARA_GEODESICO = 4.0

# Y ademas la particula tiene que ser NO CONVEXA para que el geodesico aporte
# algo. En una forma convexa el camino mas largo por dentro es el recto, asi que
# geodesico y Feret coinciden POR DEFINICION; si el geodesico sale mayor es solo
# error de discretizacion del chamfer, que no es isotropo y se desvia mas en
# angulos oblicuos. Se vio en una barra recta girada: a 0 y 45 grados medIa
# 200 px, pero a 15, 30 y 60 daba 208, un 4% que aparecia y desaparecia con el
# giro -- justo el defecto que se querIa eliminar.
#
# La solidez (area / area de la envolvente convexa) los separa sin ambiguedad
# sobre formas de talla conocida: barras y circulos quedan por encima de 1.00,
# los arcos entre 0.27 y 0.71.
SOLIDEZ_CONVEXA = 0.90

# Fraccion de la distancia maxima al fondo por encima de la cual un pixel se
# considera NUCLEO de una particula. Dos particulas que se tocan tienen dos
# nucleos aunque formen una sola componente conexa, y el watershed las separa
# por el cuello que las une.
#
# 0.80 sale de barrer el umbral sobre formas de talla conocida. Separa dos
# circulos de 60 px con solapes de 2, 6, 10 y 16 px -- hasta un 27% del
# diametro -- y no parte NINGUNA particula sana: ni un circulo, ni una barra, ni
# una fibra curva, ni una irregular de cuatro lobulos. Subirlo a 0.90 hace
# estallar la fibra curva en 27 trozos, asi que el margen es amplio por abajo y
# estrecho por arriba.
UMBRAL_NUCLEO = 0.80

# Por encima de esto la particula se considera curva y el largo de la cuerda
# deja de servir como talla. 1.15 = el contorno recorre un 15% mas que la recta.
CURVA_DESDE = 1.15


@dataclass
class Morfologia:
    """Forma y talla de una particula, medidas sobre su mascara."""
    ok: bool = False
    # px
    area_px: float = 0.0
    perimetro_px: float = 0.0
    largo_px: float = 0.0        # la talla que se reporta: max(feret, geodesico)
    ancho_px: float = 0.0
    cuerda_px: float = 0.0       # extension recta (minAreaRect)
    feret_px: float = 0.0        # mayor distancia entre dos puntos del borde
    geodesico_px: float = 0.0    # camino mas largo DENTRO de la particula
    # Largo del rectangulo de igual area y perimetro. NO es "el largo de la
    # particula": es el de un rectangulo equivalente, y un borde dentado infla
    # el perimetro y con el esta cifra. Se conserva porque comparado con feret y
    # geodesico informa de lo irregular o filamentosa que es la particula.
    largo_rect_eq_px: float = 0.0
    # um, si hay calibracion
    area_um2: Optional[float] = None
    largo_um: Optional[float] = None
    ancho_um: Optional[float] = None
    feret_um: Optional[float] = None
    geodesico_um: Optional[float] = None
    largo_rect_eq_um: Optional[float] = None
    # adimensionales
    aspecto: float = 0.0
    circularidad: float = 0.0
    curvatura: float = 1.0       # largo / cuerda; 1.0 = recta
    llenado: float = 0.0         # area de la particula / area de su caja
    solidez: float = 1.0         # area / area de su envolvente convexa
    morfotipo: str = INDETERMINADO
    metodo: str = ""             # de donde salio el largo
    # Que fraccion de la recta de Feret cae FUERA de la particula. Solo lo
    # rellena dibujar_medicion(), que es quien la dibuja. En una particula
    # concava es distinto de cero y no es un error: Feret mide la separacion de
    # dos mordazas, no un camino por dentro.
    feret_fuera: float = 0.0
    aviso: str = ""
    # La medida sale, pero pide un vistazo humano. Se marca en vez de descartar:
    # descartar en silencio pierde particulas reales del conteo, y dar el numero
    # sin avisar mete en la tabla tallas que no se sostienen.
    revisar: bool = False

    @property
    def curva(self) -> bool:
        return self.curvatura >= CURVA_DESDE


def margen_de_caja(ancho_caja: float, alto_caja: float) -> int:
    """Cuanto fondo se recorta alrededor de la caja, en pixeles.

    Proporcional al lado menor, con minimo. El 0.50 no es arbitrario: con 0.35
    una particula vecina que apenas rozaba a la nuestra caIa casi entera FUERA
    del recorte, y una vecina truncada pierde su nucleo, de modo que el
    watershed no llegaba a separarlas. Medido sobre dos circulos de 60 px, el
    error pasa de +16.3% a -2.6% con solo ampliar el margen; de 0.50 en adelante
    el resultado ya no cambia.

    Se expone como funcion para que las pruebas usen el mismo numero que el
    codigo en vez de repetirlo, que es como se desincronizan. Por el mismo
    motivo la usa ``dibujar_medicion``: si cada uno calculara su margen, la
    mascara y la foto de debajo se dibujarian desde origenes distintos.

    SE MIDE SOBRE EL LADO MAYOR, no el menor. Con el menor, el recorte de una
    fibra se calculaba a partir de su GROSOR: para una caja de 72x18 daba 9 px
    de margen, y entonces el recorte de trabajo ni siquiera contenia la fibra
    entera -- se perdia antes de segmentar nada. En una particula compacta los
    dos lados son parecidos y el numero no cambia.
    """
    return int(max(3, round(0.50 * max(max(1.0, ancho_caja),
                                       max(1.0, alto_caja)))))


def _recorte(bgr: np.ndarray, x1, y1, x2, y2, margen: int):
    """Recorta la caja con margen, y devuelve tambien el desplazamiento.

    El margen no es cosmetico: para separar la particula del fondo por umbral
    hace falta que en el recorte haya fondo. Una caja ajustada al borde de la
    particula es casi toda particula, y Otsu partiria la particula en dos en vez
    de separarla del fondo.
    """
    H, W = bgr.shape[:2]
    ax = max(0, int(x1) - margen)
    ay = max(0, int(y1) - margen)
    bx = min(W, int(np.ceil(x2)) + margen)
    by = min(H, int(np.ceil(y2)) + margen)
    if bx - ax < 3 or by - ay < 3:
        return None, 0, 0
    return bgr[ay:by, ax:bx], ax, ay


# A partir de esta elongacion -- largo/ancho del rectangulo minimo que la
# contiene -- una mascara se trata como alargada, y eso cambia dos decisiones:
# no se parte por watershed, y se le permite salirse mas de la caja.
#
# 3.0 no es un numero elegido de la nada: es el mismo corte con que
# ASPECTO_FIBRA separa fibra de particula, y ademas deja fuera el caso que
# motivo el recorte a la caja. Dos circulos de 20 px de radio unidos por un
# puente forman un conjunto de unos 90x40 px -> elongacion 2.25, por debajo del
# umbral, asi que ese caso se sigue recortando y separando como antes.
ELONGACION_ALARGADA = 3.0

# Cuanto se le permite a la mascara salirse de la caja del detector, como
# fraccion de un lado de la caja. Son dos numeros porque el error del detector
# es distinto en cada caso.
#
#   COMPACTA: fraccion del lado MENOR. La caja de una particula rechoncha es
#   buena, y la holgura solo cubre que el detector suele ajustar un pelo por
#   dentro del borde real.
#
#   ALARGADA: fraccion del lado MAYOR. En una fibra el lado menor es su grosor,
#   asi que una fraccion de el no alcanza nunca a lo largo por mucho que se
#   amplie: sobre una fibra de 36 px encajonada en 29 px, el 10% del grosor da
#   2 px y sigue cortandola. Con el lado mayor, el 25% da 7 px y la recupera.
#
# QUE EL SEGUNDO CASO EXISTA ES EL ARREGLO. Antes una mascara alargada NO se
# recortaba en absoluto, y ese "en absoluto" era el fallo: una fuga -- la
# particula mas una cresta brillante que sale de la caja -- tambien es larga y
# delgada, de modo que puntuaba como alargada y desactivaba la unica proteccion
# que la habria parado. En el informe salieron tres fichas de seis midiendo
# entre 1.9 y 2.1 veces la diagonal de su caja, una de ellas 6.4 mm: mas que el
# limite de 5 mm con que se define un microplastico.
#
# La elongacion no puede distinguir una fibra de una fuga -- las dos son largas
# y delgadas --, asi que no se le pide que lo haga. Se le pide a la CAJA, que es
# lo unico que sabe de que particula hablamos. El 0.25 sale de que las 98
# particulas anotadas a mano no pasan de 1.41 veces la diagonal de su caja.
# Diferencia minima de brillo entre la caja y el anillo que la rodea para
# admitir que ahi hay una particula. Por debajo se devuelve None en vez de
# medir: el percentil 90 de la caja contra la mediana del anillo, en niveles
# de 0-255.
CONTRASTE_MINIMO = 12

HOLGURA_COMPACTA = 0.10
HOLGURA_ALARGADA = 0.25

# A partir de cuanto se considera que la mascara se fugo. Se compara el largo
# de la mascara SIN RECORTAR contra la diagonal de la caja. 1.5 sale de los
# datos: sobre 98 particulas anotadas a mano, ninguna pasa de 1.41.
FUGA_X_DIAGONAL = 1.5


def _elongacion(mascara: np.ndarray) -> float:
    """Cuantas veces mas larga que gruesa es la mascara. 1.0 si no se puede.

    Se calcula como ``area / grosor**2``, con el grosor tomado del mayor
    circulo inscrito (transformada de distancia). Para una forma de largo L y
    grosor w el area vale del orden de L*w, asi que el cociente da L/w.

    NO se usa el rectangulo minimo, que seria lo evidente, porque **falla justo
    en las fibras curvas**: el rectangulo que envuelve un arco es casi cuadrado
    y da elongacion cercana a 1. Con esa medida, la fibra curva de la placa
    10.3 -- la que se veia cortada en el informe -- quedaba clasificada como
    compacta y se seguia recortando.

    Al depender solo del area y del grosor, esta version no cambia con la
    curvatura ni con la orientacion.
    """
    if mascara is None or not mascara.any():
        return 1.0
    area = float(np.count_nonzero(mascara))
    dt = cv2.distanceTransform((mascara > 0).astype(np.uint8), cv2.DIST_L2, 5)
    grosor = 2.0 * float(dt.max())
    if grosor <= 0:
        return 1.0
    return area / (grosor * grosor)


def separar_pegadas(mascara: np.ndarray, semilla) -> np.ndarray:
    """Si la mascara une varias particulas, devuelve solo la de la semilla.

    Dos particulas en contacto forman una sola componente conexa, y medirlas
    juntas suma sus tallas: se observo en placas reales y afecta sobre todo al
    extremo grande de la distribucion, que es justo el que se reporta como
    "la particula mayor".

    Se separan por watershed sobre la transformada de distancia, que es el
    metodo habitual para objetos en contacto en microscopia. La idea es que el
    centro de cada particula esta lejos del fondo y el cuello que las une esta
    cerca, de modo que umbralizando la distancia aparecen dos nucleos separados
    aunque la mascara sea una sola pieza; el watershed crece desde ellos y corta
    justo por el cuello.

    Si solo hay un nucleo no se toca nada: la inmensa mayoria de las particulas
    estan aisladas y no tiene sentido arriesgar un corte donde no hay nada que
    cortar.
    """
    if mascara is None or not mascara.any():
        return mascara

    # Una mascara claramente alargada no se parte. Una fibra de grosor
    # irregular tiene VARIAS zonas gruesas, y cada una pasa el umbral de nucleo
    # y se convierte en un centro de watershed: entonces la fibra se corta por
    # su propio adelgazamiento, como si fueran dos particulas en contacto. Se
    # midio sobre una fibra real de la placa 10.3: de 369 px de area quedaban
    # 254, un 31% menos, y el largo caia con ella.
    #
    # El umbral 0.80 se calibro sobre CIRCULOS pegados, que es el caso para el
    # que existe esta funcion. Una fibra no es ese caso.
    #
    # El riesgo de no partir -- dos fibras que de verdad se tocan quedan como
    # una -- ya lo cubre medir_deteccion(), que marca para revisar cualquier
    # particula mas larga que 1.5 veces la diagonal de su caja. Entre subestimar
    # en silencio y sobreestimar avisando, avisar es lo correcto.
    if _elongacion(mascara) >= ELONGACION_ALARGADA:
        return mascara

    dt = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    if dt.max() < 2:
        return mascara

    _, nucleos = cv2.threshold(dt, UMBRAL_NUCLEO * dt.max(), 255,
                               cv2.THRESH_BINARY)
    nucleos = nucleos.astype(np.uint8)
    n, marcas = cv2.connectedComponents(nucleos)
    if n <= 2:                       # fondo + un nucleo: particula aislada
        return mascara

    # cv2.watershed reserva el 0 para "desconocido" y el 1 para el fondo.
    marcas = marcas + 1
    marcas[mascara == 0] = 1
    marcas[(mascara > 0) & (nucleos == 0)] = 0
    try:
        cv2.watershed(cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR), marcas)
    except cv2.error:
        return mascara

    sx, sy = int(semilla[0]), int(semilla[1])
    sx = int(np.clip(sx, 0, marcas.shape[1] - 1))
    sy = int(np.clip(sy, 0, marcas.shape[0] - 1))
    etiqueta = marcas[sy, sx]
    if etiqueta <= 1:
        # El centro cayo en la linea divisoria o en el fondo. Se queda la region
        # mayor de las que tocan el centro, antes que devolver el conjunto.
        vecinas = marcas[max(0, sy - 2):sy + 3, max(0, sx - 2):sx + 3]
        candidatas = [v for v in np.unique(vecinas) if v > 1]
        if not candidatas:
            return mascara
        etiqueta = max(candidatas,
                       key=lambda v: int((marcas == v).sum()))

    salida = ((marcas == etiqueta) & (mascara > 0)).astype(np.uint8) * 255
    return salida if salida.any() else mascara


def segmentar(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              margen: Optional[int] = None,
              registro: Optional[dict] = None) -> Optional[np.ndarray]:
    """Mascara de la particula dentro de la caja. None si no se pudo aislar.

    Si se pasa ``registro``, se anota en el cuanto se salia la mascara antes de
    recortarla a la caja. Es lo unico que distingue una fuga de una particula
    grande, y solo se sabe aqui: despues del recorte la evidencia ya no esta.

    Bajo Nile Red la particula emite y el fondo del filtro queda oscuro, asi que
    se umbraliza el canal de mayor contraste con Otsu. Se toma la componente
    conexa que cubre el centro de la caja: dentro del recorte pueden caer trozos
    de particulas vecinas, y la que interesa es la que el detector senalo.
    """
    if bgr is None or bgr.size == 0:
        return None
    # Una caja que no corta la imagen es un dato erroneo, no un caso limite. Sin
    # esta guarda, el margen alcanzaba una esquina de la foto y se devolvia una
    # medida de fondo como si fuera una particula.
    alto_img, ancho_img = bgr.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= ancho_img or y1 >= alto_img:
        return None
    ancho_caja = max(1.0, x2 - x1)
    alto_caja = max(1.0, y2 - y1)
    if margen is None:
        margen = margen_de_caja(ancho_caja, alto_caja)

    sub, ox, oy = _recorte(bgr, x1, y1, x2, y2, margen)
    if sub is None:
        return None

    # El canal mas contrastado, no siempre el gris: PP y LDPE emiten en amarillo
    # (R y G altos, B bajo) y PET en rojo, de modo que el maximo por pixel separa
    # mejor del fondo que la media ponderada que usa BGR2GRAY.
    canal = sub.max(axis=2)
    canal = cv2.GaussianBlur(canal, (3, 3), 0)

    # El umbral sale del CONTRASTE ENTRE EL ANILLO Y LA CAJA, no de Otsu sobre
    # todo el recorte. Otsu parte el histograma en dos clases sin saber donde
    # esta la particula, y cuando el recorte es casi todo particula acaba
    # metiendo fondo oscuro dentro de la mascara: medido sobre 2863 particulas,
    # la mediana de la mascara que era fondo oscuro era del 5.4%, y en un 7.4%
    # de ellas pasaba del 20%. Eso inflaba el area de forma sistematica.
    #
    # Aqui se usa lo que la caja del detector ya sabe: fuera de ella hay fondo,
    # dentro hay particula. El nivel de fondo es la mediana del anillo, el de la
    # particula el percentil 90 de la caja, y se corta a medio camino -- la
    # misma tecnica con que calibracion.py encuentra el borde de la placa.
    bx1 = int(np.clip(int(x1) - ox, 0, canal.shape[1]))
    by1 = int(np.clip(int(y1) - oy, 0, canal.shape[0]))
    bx2 = int(np.clip(int(np.ceil(x2)) - ox, 0, canal.shape[1]))
    by2 = int(np.clip(int(np.ceil(y2)) - oy, 0, canal.shape[0]))
    anillo = np.ones(canal.shape, dtype=bool)
    anillo[by1:by2, bx1:bx2] = False
    dentro_caja = canal[by1:by2, bx1:bx2]

    umbral = None
    if anillo.sum() >= 20 and dentro_caja.size >= 9:
        fondo = float(np.median(canal[anillo]))
        pico = float(np.percentile(dentro_caja, 90))
        if pico - fondo >= CONTRASTE_MINIMO:
            umbral = fondo + 0.5 * (pico - fondo)
        else:
            # SIN CONTRASTE NO SE INVENTA UNA TALLA. Aqui se caia a Otsu, que
            # sin particula que separar parte el histograma del RUIDO y
            # devuelve una mascara fragmentada -- con una talla de aspecto
            # normal, indistinguible de un dato bueno salvo mirando la imagen.
            #
            # Medido sobre 100 cajas anotadas a mano: 7 caen por aqui, y las 7
            # son de las dos placas anotadas en el marco de pixeles equivocado,
            # donde la caja señala un sitio sin particula. El corte es nitido:
            # esas 7 tienen contraste 2-7 y la siguiente tiene 34.
            #
            # Devolver None hace que el informe diga "no se pudo separar la
            # particula del fondo" y la cuente aparte, que es lo que de verdad
            # paso.
            return None

    if umbral is None:
        _, mascara = cv2.threshold(canal, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mascara = (canal > umbral).astype(np.uint8) * 255

    # NO se filtra por color, aunque separaria dos particulas de polimero
    # distinto que se tocan. Se probo y se midio, y no funciona en este material:
    #
    #   nucleo -> borde de la MISMA particula   mediana 0.197 de cromaticidad
    #   nucleo -> vecino de OTRA clase          mediana 0.281, p25 0.173
    #
    # Las dos distribuciones se solapan, asi que no hay umbral: con 0.10 se corta
    # el 93.7% de los bordes propios y con 0.26 solo se separa el 54% de los
    # vecinos ajenos. La causa es que PP y LDPE se distinguen por BRILLO y no por
    # tono, y que el borde tenue de cualquier particula deriva su color hacia el
    # fondo. El caso se resuelve por geometria en separar_pegadas(), que no
    # necesita distinguir de que polimero es cada una.

    # Quita el ruido de sal sin cerrar el hueco de una particula con forma de U.
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8), iterations=1)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return None
    # Centro de la caja en coordenadas del recorte.
    cx = int(round((x1 + x2) / 2 - ox))
    cy = int(round((y1 + y2) / 2 - oy))
    cx = int(np.clip(cx, 0, etiquetas.shape[1] - 1))
    cy = int(np.clip(cy, 0, etiquetas.shape[0] - 1))
    idx = etiquetas[cy, cx]
    if idx == 0:
        # El centro cayo en fondo -- particula anular o caja algo descentrada.
        # Se usa la componente mayor que toque la caja.
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 4:
        return None
    mascara = (etiquetas == idx).astype(np.uint8) * 255

    # Separar dos particulas en contacto ANTES de recortar a la caja, no
    # despues. El recorte trunca a la vecina, y una vecina truncada pierde su
    # nucleo: sobre dos circulos de 60 px pegados, sin recortar se detectaban
    # los dos nucleos con solapes de hasta 16 px, y ya recortados no se detectaba
    # ninguno. Es decir que con el orden inverso la separacion no llegaba a
    # actuar nunca.
    mascara = separar_pegadas(mascara, (cx, cy))

    # Recortar a la caja. El margen existe para que el umbral tenga fondo con
    # que comparar, no para medir en el: sin este recorte, una particula pegada
    # a otra forma con ella una sola componente conexa y se mide el conjunto. Se
    # observo en placas reales una mascara 1.71 veces mas ancha que su caja, y
    # de ahi salian tallas de 8.9 mm que ninguna caja de 97 px puede contener.
    #
    # Se deja una holgura: el detector suele ajustar la caja algo por dentro del
    # borde real, y cortar a ras amputaria la punta de la particula.
    #
    # LA HOLGURA DEPENDE DE LO ALARGADA QUE SEA LA MASCARA, pero SIEMPRE SE
    # RECORTA. Las dos mitades de esa frase costaron un fallo cada una:
    #
    #   * Con la holgura fija del 10% del lado menor, las FIBRAS salian cortadas
    #     por un borde recto. En una fibra el lado menor es su grosor, asi que
    #     esa fraccion no alcanza nunca a lo largo. Sobre dos fibras reales de
    #     la placa 10.3 el recorte se llevaba el 31% y el 34% del area.
    #
    #   * El arreglo de entonces fue no recortar NADA si la mascara era
    #     alargada, y eso abrio el fallo contrario. La elongacion no distingue
    #     una fibra de una fuga: una particula con una cresta brillante pegada
    #     tambien es larga y delgada, asi que puntuaba como alargada y apagaba
    #     la unica proteccion que la habria parado. En un informe salieron tres
    #     fichas de seis midiendo entre 1.9 y 2.1 veces la diagonal de su caja,
    #     una de ellas de 6.4 mm -- por encima del limite de 5 mm con que se
    #     define un microplastico.
    #
    # De ahi la forma actual: a una mascara alargada se le da MAS holgura, no
    # holgura infinita, y medida sobre el lado MAYOR para que alcance a lo
    # largo. La caja es lo unico que sabe de que particula estamos hablando; la
    # forma de la mascara no puede decirlo, porque una fibra y una fuga tienen
    # la misma forma.
    #
    # La proteccion sigue en pie donde importa: dos circulos de 20 px unidos por
    # un puente dan elongacion 1.7 -- rechonchos -- y se recortan como siempre.
    # Es el caso que motivo este recorte y del que salian tallas de 8.9 mm.
    if registro is not None:
        registro["largo_sin_recortar_px"] = feret_maximo(_contorno_mayor(mascara))
    if _elongacion(mascara) < ELONGACION_ALARGADA:
        holgura = max(2, int(round(HOLGURA_COMPACTA * min(ancho_caja, alto_caja))))
    else:
        holgura = max(2, int(round(HOLGURA_ALARGADA * max(ancho_caja, alto_caja))))
    hx1 = max(0, int(x1) - ox - holgura)
    hy1 = max(0, int(y1) - oy - holgura)
    hx2 = min(mascara.shape[1], int(np.ceil(x2)) - ox + holgura)
    hy2 = min(mascara.shape[0], int(np.ceil(y2)) - oy + holgura)
    fuera = np.ones_like(mascara, dtype=bool)
    fuera[hy1:hy2, hx1:hx2] = False
    mascara[fuera] = 0
    return mascara if mascara.any() else None


def _contorno_mayor(mascara: np.ndarray) -> Optional[np.ndarray]:
    """El contorno externo de mayor area, o None si la mascara esta vacia."""
    if mascara is None or not mascara.any():
        return None
    cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cont, key=cv2.contourArea) if cont else None


def feret_maximo(contorno: np.ndarray) -> float:
    """Diametro de Feret maximo: la mayor distancia entre dos puntos del borde.

    Se calcula sobre la envolvente convexa, no sobre el contorno entero: el par
    mas separado siempre esta en la envolvente, y esta suele tener decenas de
    puntos frente a los miles del contorno, asi que la busqueda exhaustiva es
    barata y exacta.

    Es la medida correcta para el "largo maximo aparente" y para una fibra
    RECTA. Para una fibra curva da la cuerda, que subestima su longitud.
    """
    if contorno is None or len(contorno) < 2:
        return 0.0
    env = cv2.convexHull(contorno).reshape(-1, 2).astype(float)
    if len(env) < 2:
        return 0.0
    d = env[:, None, :] - env[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def _propagar_geodesico(dentro: np.ndarray, semilla: tuple) -> np.ndarray:
    """Geodesic distance from a seed, staying inside the mask.

    chamfer 3x3 successive-minima sweep over the 8 neighbours, weight 1 on
    straight and sqrt(2)-ish on diagonal. Vectorized: each iteration is eight
    shifted adds, and as many are needed as the longest path has pixels.

    Hot-loop buffer design (S02 rewrite): the loop allocates NOTHING on the
    steady state. All scratch is preallocated once (candidate buffer ``p``,
    update buffer ``e``, cached ``fuera`` mask, three slab temps for the
    shifted adds), every shifted add writes into a preallocated ``out=`` view,
    and the convergence update happens in-place into ``e`` followed by a
    reference swap instead of a full copy. Allocation churn was the dominant
    avoidable cost at ~5.9 ms/particle measured on a 10-plate batch.

    Bit-identical contract, pinned by tests/test_geodesia.py:
      * float32 weights np.float32(0.95509) / np.float32(1.36930), UNCHANGED;
      * the 8 shifted minimums accumulate into ``p`` in the SAME arange order
        as the pre-rewrite body (rectos then diagonals, bottom-up then top-down)
        — reordering is not free: float minimum is exact but the accumulator
        order is part of the frozen behavior;
      * ``tope = int(dentro.sum()) + 2`` iteration budget, unchanged;
      * convergence breaks on ``np.array_equal`` between successive sweeps,
        bit-exactly as before.

    Done this way rather than with scipy.sparse.csgraph because scipy is not
    a declared project dependency: it arrives transitively with ultralytics,
    and relying on it would break the install the day ultralytics stops
    shipping it.
    """
    INF = np.inf
    H, W = dentro.shape
    d = np.full(dentro.shape, INF, dtype=np.float32)
    d[semilla] = 0.0
    # Pesos optimos del chamfer 3x3, no 1 y raiz de 2. Con los pesos ingenuos el
    # camino en escalera sobreestima la distancia euclidea hasta un 8% -- medido
    # sobre un circulo de diametro conocido, daba 108 en vez de 100 --. Estos
    # dos minimizan ese error maximo y lo dejan en torno al 1.4%.
    UNO = np.float32(0.95509)
    R2 = np.float32(1.36930)
    # Cota de seguridad: ningun camino simple puede tener mas pasos que pixeles.
    tope = int(dentro.sum()) + 2
    # Buffers preasignados una vez: p (candidatos), e (actualizacion), el
    # inverso de ``dentro`` cacheado y tres losas para los sumandos desplazados
    # (fila, columna y diagonal). Nada de esto se asigna dentro del bucle.
    p = np.full_like(d, INF)
    e = np.empty_like(d)
    fuera = ~dentro  # cacheada: ~=dentro es constante durante todo el barrido
    t_fila = np.empty((max(H - 1, 0), W), dtype=np.float32)
    t_col = np.empty((H, max(W - 1, 0)), dtype=np.float32)
    t_diag = np.empty((max(H - 1, 0), max(W - 1, 0)), dtype=np.float32)
    for _ in range(tope):
        p.fill(INF)
        # Rectos
        np.add(d[:-1, :], UNO, out=t_fila[: H - 1])
        np.minimum(p[1:, :], t_fila[: H - 1], out=p[1:, :])
        np.add(d[1:, :], UNO, out=t_fila[: H - 1])
        np.minimum(p[:-1, :], t_fila[: H - 1], out=p[:-1, :])
        np.add(d[:, :-1], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, 1:], t_col[:, : W - 1], out=p[:, 1:])
        np.add(d[:, 1:], UNO, out=t_col[:, : W - 1])
        np.minimum(p[:, :-1], t_col[:, : W - 1], out=p[:, :-1])
        # Diagonales
        np.add(d[:-1, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, 1:], t_diag[: H - 1, : W - 1], out=p[1:, 1:])
        np.add(d[:-1, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[1:, :-1], t_diag[: H - 1, : W - 1], out=p[1:, :-1])
        np.add(d[1:, :-1], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, 1:], t_diag[: H - 1, : W - 1], out=p[:-1, 1:])
        np.add(d[1:, 1:], R2, out=t_diag[: H - 1, : W - 1])
        np.minimum(p[:-1, :-1], t_diag[: H - 1, : W - 1], out=p[:-1, :-1])
        p[fuera] = INF
        # Actualizacion en sitio con swap de referencias, sin copiar el array.
        np.minimum(d, p, out=e)
        # Basta comparar con el paso anterior: la propagacion es monotona
        # decreciente, asi que si nada bajo, ya converge.
        if np.array_equal(e, d):
            break
        d, e = e, d
    return d


def largo_geodesico(mascara: np.ndarray) -> float:
    """Camino mas largo que cabe DENTRO de la particula, siguiendo su forma.

    Es el diametro geodesico. Hace lo correcto en los dos casos que importan y
    sin cambiar de formula:

      * Fibra curva: el camino no puede salirse de la particula, asi que rodea
        la curva y devuelve su longitud, no la cuerda entre los extremos.
      * Particula compacta: el camino mas largo es el recto, y coincide con el
        diametro de Feret maximo.

    Se obtiene con la tecnica de las dos propagaciones, la misma que da el
    diametro de un arbol: desde un punto cualquiera se busca el mas lejano, y
    desde ese se vuelve a propagar; el maximo de esa segunda propagacion es el
    diametro. No depende del perimetro, de modo que un borde dentado no lo
    infla como si hace el modelo de rectangulo.
    """
    idx = np.argwhere(mascara > 0)
    if len(idx) < 2:
        return 0.0
    # Recortar al minimo rectangulo que contiene la particula. El coste de la
    # propagacion va con el area del lienzo, y una particula de 30 px dentro de
    # una placa de 4096 costaria lo mismo que la placa entera.
    y0, x0 = idx.min(axis=0)
    y1_, x1_ = idx.max(axis=0) + 1
    dentro = (mascara[y0:y1_, x0:x1_] > 0)
    idx = np.argwhere(dentro)

    d1 = _propagar_geodesico(dentro, tuple(idx[0]))
    finitos = np.isfinite(d1)
    if not finitos.any():
        return 0.0
    a = np.unravel_index(np.argmax(np.where(finitos, d1, -1)), d1.shape)

    d2 = _propagar_geodesico(dentro, a)
    finitos2 = np.isfinite(d2)
    return float(d2[finitos2].max()) if finitos2.any() else 0.0


def camino_geodesico(mascara: np.ndarray):
    """Los puntos del camino geodesico mas largo, para poder dibujarlo.

    Misma doble propagacion que ``largo_geodesico``, pero devolviendo el
    recorrido en vez de solo su longitud. Se reconstruye bajando siempre al
    vecino de menor distancia desde el extremo mas lejano.

    Existe para que el informe pueda ENSENAR sobre que recorrido se midio cada
    fibra. Una talla que no se puede ver medida no se puede verificar.
    """
    idx = np.argwhere(mascara > 0)
    if len(idx) < 2:
        return []
    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0) + 1
    dentro = (mascara[y0:y1, x0:x1] > 0)
    ii = np.argwhere(dentro)
    d1 = _propagar_geodesico(dentro, tuple(ii[0]))
    if not np.isfinite(d1).any():
        return []
    a = np.unravel_index(np.argmax(np.where(np.isfinite(d1), d1, -1)), d1.shape)
    d2 = _propagar_geodesico(dentro, a)
    if not np.isfinite(d2).any():
        return []
    b = np.unravel_index(np.argmax(np.where(np.isfinite(d2), d2, -1)), d2.shape)

    camino, p = [b], b
    for _ in range(int(dentro.sum()) + 2):
        if p == a:
            break
        mejor, menor = None, d2[p]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (p[0] + dy, p[1] + dx)
                if (0 <= q[0] < d2.shape[0] and 0 <= q[1] < d2.shape[1]
                        and d2[q] < menor):
                    menor, mejor = d2[q], q
        if mejor is None:
            break
        p = mejor
        camino.append(p)
    return [(int(c[1] + x0), int(c[0] + y0)) for c in camino]


def extremos_feret(contorno: np.ndarray):
    """Los dos puntos del contorno mas separados entre si, para dibujarlos."""
    if contorno is None or len(contorno) < 2:
        return None
    env = cv2.convexHull(contorno).reshape(-1, 2).astype(float)
    if len(env) < 2:
        return None
    d = env[:, None, :] - env[None, :, :]
    D = np.sqrt((d ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return tuple(env[i].astype(int)), tuple(env[j].astype(int))


def _dibujar_feret(vis: np.ndarray, mascara: np.ndarray, p, q, color) -> float:
    """Dibuja la recta de Feret: continua dentro de la particula, a trazos fuera.

    Devuelve que fraccion del segmento cae FUERA de la mascara, para poder
    decirlo con un numero en vez de dejarlo al ojo.

    Se dibuja asi porque Feret es la separacion de dos mordazas, no un camino:
    en una particula concava el segmento pasa por fuera y eso es correcto. Con
    la linea continua parecia un error de medida.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    largo = float(np.hypot(*(q - p)))
    if largo < 1:
        return 0.0
    n = int(largo) + 1
    ts = np.linspace(0.0, 1.0, n)
    pts = p[None, :] + ts[:, None] * (q - p)[None, :]
    xs = np.clip(pts[:, 0].round().astype(int), 0, mascara.shape[1] - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, mascara.shape[0] - 1)
    dentro = mascara[ys, xs] > 0
    for k in range(n - 1):
        # Fuera de la particula, un pixel de cada tres: el trazo discontinuo es
        # lo que distingue "aqui la mordaza no toca" de "aqui se midio".
        if not dentro[k] and (k % 3):
            continue
        cv2.line(vis, (xs[k], ys[k]), (xs[k + 1], ys[k + 1]), color, 1)
    return float((~dentro).mean())


def dibujar_medicion(bgr: np.ndarray, det, zoom: int = 4):
    """Recorte de la particula con la medida que se le aplico, dibujada encima.

    Devuelve (imagen, Morfologia) o (None, None). La imagen lleva el contorno de
    la mascara y, encima, la recta de Feret o el camino geodesico segun cual de
    los dos haya decidido la talla, para que se vea EXACTAMENTE sobre que se
    midio esa particula y no haya que creerselo.
    """
    try:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        mascara = segmentar(bgr, x1, y1, x2, y2)
        if mascara is None:
            return None, None
        m = medir(mascara)
        # Mismo margen que uso segmentar() para recortar -- tiene que ser
        # EXACTAMENTE margen_de_caja(), no un numero aparte. Aqui habia un 0.35
        # copiado a mano que no coincidia con el 0.50 real de segmentar(): la
        # mascara se calculaba sobre un recorte y el fondo mostrado se leia
        # desde otro origen, asi que el contorno y la foto de abajo quedaban
        # desalineados unos pocos px (visibles a zoom alto). Se vio a ojo en un
        # informe: la mascara "quedaba corrida" respecto de la particula.
        margen = margen_de_caja(x2 - x1, y2 - y1)
        ox, oy = max(0, int(x1) - margen), max(0, int(y1) - margen)
        sub = bgr[oy:oy + mascara.shape[0], ox:ox + mascara.shape[1]].copy()
        if sub.size == 0:
            return None, None

        vis = sub.copy()
        cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cont, -1, (0, 220, 0), 1)
        if m.ok:
            if m.metodo.startswith("geodesico"):
                cam = camino_geodesico(mascara)
                for k in range(1, len(cam)):
                    cv2.line(vis, cam[k - 1], cam[k], (255, 80, 255), 1)
            elif cont:
                ext = extremos_feret(max(cont, key=cv2.contourArea))
                if ext:
                    fuera = _dibujar_feret(vis, mascara, ext[0], ext[1],
                                           (0, 210, 255))
                    for q in ext:
                        cv2.circle(vis, q, 2, (0, 210, 255), -1)
                    m.feret_fuera = fuera
        par = np.hstack([sub, vis])
        if zoom > 1:
            par = cv2.resize(par, None, fx=zoom, fy=zoom,
                             interpolation=cv2.INTER_NEAREST)
        return par, m
    except Exception:
        return None, None


def medir(mascara: np.ndarray, um_por_px: Optional[float] = None) -> Morfologia:
    """Talla y forma a partir de la mascara de una particula."""
    m = Morfologia()
    if mascara is None or not mascara.any():
        m.aviso = "sin mascara"
        return m

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contornos:
        m.aviso = "sin contorno"
        return m
    c = max(contornos, key=cv2.contourArea)

    # El area se cuenta sobre la mascara y no con contourArea: la formula del
    # poligono subestima en particulas de pocos pixeles, que son la mayoria aqui.
    area = float(np.count_nonzero(mascara))
    # El contorno se recorre pixel a pixel, SIN suavizar. Se probo suavizarlo
    # con approxPolyDP para quitar la escalera de digitalizacion, contra formas
    # sinteticas de talla conocida, y empeora: con epsilon 1 px el error mediano
    # sube de 1.0% a 2.4% y el peor de 3.1% a 7.8%, porque recorta los recodos
    # reales de la particula junto con la escalera. Tambien se probo estimar el
    # largo como area/grosor con la transformada de distancia, inmune al
    # perimetro, y es peor todavia: 26.6% de error mediano.
    perim = float(cv2.arcLength(c, True))
    if area < 4 or perim <= 0:
        m.aviso = "particula demasiado pequena para medir su forma"
        return m

    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    cuerda = float(max(rw, rh))
    grosor_recto = float(min(rw, rh))

    # Modelo de rectangulo: valido solo si existe un rectangulo con esa area y
    # ese perimetro. En una particula compacta el discriminante es negativo.
    # Rectangulo de igual area y perimetro. Descriptor, NO la talla: solo existe
    # si P^2 >= 16A, que es tanto como decir que la particula es lo bastante
    # alargada -- un circulo da P^2 = 4*pi*A < 16A y no tiene rectangulo
    # equivalente. Comparado con feret y geodesico dice si la particula es
    # irregular: si es mucho mayor que ellos, el borde esta dentado.
    disc = perim * perim - 16.0 * area
    largo_rect_eq = (perim + float(np.sqrt(disc))) / 4.0 if disc > 0 else 0.0

    # ── El largo que se reporta ──
    # Feret maximo y diametro geodesico, y se toma el mayor. Para una forma
    # conexa el geodesico SIEMPRE es mayor o igual que Feret -- en una convexa
    # coinciden --, asi que cuando Feret sale mayor es solo error de
    # discretizacion del chamfer, y quedarse con el mayor toma la cota mas
    # ajustada de las dos.
    #
    # Se prefiere esto al modelo de rectangulo, medido contra formas de talla
    # conocida, porque el modelo depende del perimetro y un borde dentado lo
    # infla: en una recta con dientes de sierra daba +22.5%. max(Feret,
    # geodesico) tiene un error mediano algo mayor (3.6% frente a 1.2%) pero su
    # peor caso es 4.7% en vez de 22.5%, y aqui importa mas no fallar feo.
    feret = feret_maximo(c)
    geo = largo_geodesico(mascara)

    # El grosor decide cual de los dos vale. Solo en una particula delgada el
    # camino geodesico sigue su forma; en una gruesa rodea las concavidades y
    # devuelve un numero mayor que su extension real.
    dt = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    grosor = 2.0 * float(dt.max())
    delgadez = geo / grosor if grosor > 0 else 0.0

    # Solidez: cuanto se aparta de su propia envolvente convexa. Una fibra
    # doblada deja mucho hueco dentro de la envolvente; una barra recta, ninguno.
    envolvente = cv2.contourArea(cv2.convexHull(c))
    solidez = area / envolvente if envolvente > 0 else 1.0

    if (delgadez >= DELGADEZ_PARA_GEODESICO
            and solidez < SOLIDEZ_CONVEXA and geo >= feret):
        largo, metodo = geo, "geodesico"
    else:
        largo, metodo = feret, "Feret maximo"

    # El ancho es ese mismo grosor maximo inscrito: el diametro del mayor
    # circulo que cabe dentro de la particula.
    #
    # No se usa area/largo, que seria el ancho medio de una cinta, porque en una
    # particula compacta no da su diametro: un disco tiene A/L = pi*d/4 = 0.785d,
    # con lo que un circulo perfecto salia con aspecto 1.27 en vez de 1.0 y la
    # clasificacion fibra/fragmento arrancaba sesgada.
    ancho = grosor

    # Cuanto se aparta de una recta. Con el geodesico esto ya no depende del
    # perimetro: es cuanto mas largo es el camino por dentro que la cuerda.
    curvatura = largo / cuerda if largo > 0 and cuerda > 0 else 1.0

    m.ok = True
    m.area_px = area
    m.perimetro_px = perim
    m.largo_px = largo
    m.ancho_px = max(ancho, 0.0)
    m.cuerda_px = cuerda
    m.feret_px = feret
    m.geodesico_px = geo
    m.largo_rect_eq_px = largo_rect_eq
    m.aspecto = largo / ancho if ancho > 0 else 0.0
    m.circularidad = float(4.0 * np.pi * area / (perim * perim))
    m.curvatura = curvatura
    m.llenado = area / (rw * rh) if rw * rh > 0 else 0.0
    m.solidez = solidez
    m.metodo = metodo
    m.morfotipo = FIBRA if m.aspecto >= ASPECTO_FIBRA else FRAGMENTO

    if um_por_px and um_por_px > 0:
        m.area_um2 = area * um_por_px * um_por_px
        m.largo_um = largo * um_por_px
        m.ancho_um = m.ancho_px * um_por_px
        m.feret_um = feret * um_por_px
        m.geodesico_um = geo * um_por_px
        m.largo_rect_eq_um = largo_rect_eq * um_por_px if largo_rect_eq else None
    return m


def medir_deteccion(bgr: np.ndarray, det, um_por_px: Optional[float] = None,
                    margen: Optional[int] = None) -> Morfologia:
    """Segmenta la caja de una ``Detection`` y la mide. Nunca lanza.

    Un fallo de segmentacion en una particula no puede tumbar el lote: se
    devuelve una Morfologia con ``ok`` en False y quien llama decide si cae al
    tamano de la caja.
    """
    try:
        registro: dict = {}
        mascara = segmentar(bgr, det.x1, det.y1, det.x2, det.y2, margen,
                            registro=registro)
        if mascara is None:
            m = Morfologia()
            m.aviso = "no se pudo separar la particula del fondo"
            return m
        m = medir(mascara, um_por_px)
        diagonal = float(np.hypot(det.x2 - det.x1, det.y2 - det.y1))
        # El aviso mira la mascara ANTES del recorte, no despues.
        #
        # Despues del recorte la medida ya esta acotada por construccion, asi
        # que comprobarla ahi no puede delatar nada: seria un aviso que no se
        # dispara nunca. Lo que dice si esta particula es de fiar es cuanto se
        # SALIA -- si la mascara se iba al doble de su caja, el umbral engancho
        # algo que no era esta particula, y lo que se reporta es el trozo que
        # quedo dentro de la caja, no la particula entera.
        previo = float(registro.get("largo_sin_recortar_px", 0.0) or 0.0)
        if m.ok and diagonal > 0 and previo > FUGA_X_DIAGONAL * diagonal:
            m.revisar = True
            m.aviso = (m.aviso + " | " if m.aviso else "") + (
                f"la mascara se salia {previo / diagonal:.1f}x la diagonal de su "
                f"caja y se recorto a ella: puede ser una vecina pegada, y la "
                f"talla se queda corta si la particula seguia de verdad")
        return m
    except cv2.error as e:
        m = Morfologia()
        m.aviso = f"error de OpenCV al segmentar: {e}"
        return m


def aplicar_a_deteccion(det, bgr: np.ndarray,
                        um_por_px: Optional[float] = None) -> bool:
    """Mide la particula y vuelca el resultado en la ``Detection``.

    Devuelve si se pudo medir. Cuando no, los campos de forma quedan en None y
    quien llama debe caer al tamano de la caja: es peor perder la particula del
    conteo que reportarla con una talla aproximada y decirlo.

    ``diam_um`` pasa a ser el diametro del circulo de igual area **real**, no el
    de la caja. Es la misma magnitud de siempre pero bien calculada, de modo que
    los filtros por tamano y el histograma del informe siguen funcionando sin
    tocarlos, ahora sobre el numero correcto.
    """
    m = medir_deteccion(bgr, det, um_por_px)
    if not m.ok:
        return False
    det.aspecto = m.aspecto
    det.curvatura = m.curvatura
    det.morfotipo = m.morfotipo
    det.metodo_largo = m.metodo
    # Se copia el aviso a la Detection para que el informe pueda mostrarlo. Sin
    # esto, una talla inflada por dos particulas pegadas llegaba al documento
    # como un numero limpio, indistinguible de una medida buena.
    det.revisar = m.revisar
    det.aviso_forma = m.aviso or None
    if um_por_px and um_por_px > 0:
        det.area_um2 = m.area_um2
        det.largo_um = m.largo_um
        det.ancho_um = m.ancho_um
        det.feret_um = m.feret_um
        det.geodesico_um = m.geodesico_um
        det.largo_rect_eq_um = m.largo_rect_eq_um
        det.diam_um = 2.0 * float(np.sqrt(m.area_um2 / np.pi)) if m.area_um2 else None
    return True
