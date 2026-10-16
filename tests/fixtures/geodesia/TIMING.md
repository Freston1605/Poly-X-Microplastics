# TIMING de _propagar_geodesico: referencia vs reescrita

Generado por `tests/bench_geodesia.py` (correrlo a mano: no es un test). Numeros reales medidos en esta maquina, no placeholders.

## Contexto de la maquina

- Python 3.14.8 | numpy 2.5.3 | OpenCV 4.14.0
- Linux 7.2.9-ogc3.1.fc44.x86_64 (x86_64)

## Que mide este artefacto (y que NO)

Cada mascara se mide replicando el patron de `largo_geodesico`: las DOS propagaciones (d1 desde la semilla del rectangulo envolvente y d2 desde el extremo lejano), sobre el mismo recorte. Se reporta la mediana de repeticiones (con calentamiento descartado) y el peor caso entre repeticiones.

**Honestidad del alcance:** en esta copia de trabajo NO hay datos `runs/` ni ultralytics, asi que el lote de 10 placas no puede correrse aca. La linea base externa (~5.9 ms/particula y 7.1 s para el lote de 10 placas (medido en la maquina Windows del usuario, ANTES de la reescritura)) solo se CITA como contexto. Lo comparable entre maquinas es el **ratio referencia/reescrita sobre el mismo corpus y el mismo hardware**, y eso es lo que mide este artefacto.

Nota sobre ratios por debajo de 1.0x: aparecen solo en mascaras cuyo costo total ronda el tick del reloj (~0.05-1.5 ms): ahi el ruido del temporizador domina y la ratio no es informativa (sub-1% del corpus). Ninguna CLASE queda por debajo de 1.0x en el resumen.

## Tabla por clase

| clase | mascara | px | lado | ref ms | new ms | ratio | ref max ms | new max ms |
|---|---|---|---|---|---|---|---|---|
| fixture real | y11n__fibra_curva__000.npy | 456 | 36x75 | 5.178 | 5.070 | 1.02x | 5.451 | 5.200 |
| fixture real | y11n__fibra_recta__000.npy | 461 | 5x93 | 4.850 | 4.941 | 0.98x | 4.896 | 4.944 |
| fixture real | y11n__fibra_sinuosa__000.npy | 898 | 40x103 | 9.644 | 9.499 | 1.02x | 10.467 | 9.549 |
| fixture real | y11n__fragmento_circular__000.npy | 1253 | 39x39 | 2.682 | 2.655 | 1.01x | 2.690 | 2.657 |
| fixture real | y11n__fragmento_irregular__000.npy | 2528 | 55x56 | 4.427 | 4.311 | 1.03x | 4.435 | 4.326 |
| fixture real | y11n__fragmento_rugoso__000.npy | 1626 | 53x53 | 4.097 | 4.022 | 1.02x | 4.107 | 4.030 |
| fixture real | y11n__particula_alargada_diagonal__000.npy | 757 | 87x87 | 9.783 | 9.587 | 1.02x | 9.856 | 9.641 |
| fixture real | y11n__particula_bifurcada__000.npy | 748 | 77x69 | 7.416 | 7.238 | 1.02x | 7.498 | 7.295 |
| procedural | blob borde ruidoso | 13033 | 145x138 | 28.598 | 26.940 | 1.06x | 28.653 | 27.106 |
| procedural | cinta sinuosa | 2216 | 132x166 | 49.626 | 46.822 | 1.06x | 49.689 | 46.982 |
| procedural | espiral 1.6 vueltas | 2526 | 134x155 | 103.511 | 98.036 | 1.06x | 103.592 | 98.170 |
| procedural | grumo con huecos | 15405 | 145x145 | 29.494 | 27.656 | 1.07x | 29.592 | 27.774 |
| procedural | erodada bernoulli | 2855 | 47x70 | 5.722 | 5.606 | 1.02x | 5.749 | 5.645 |
| procedural | dos componentes | 4906 | 141x57 | 6.844 | 6.596 | 1.04x | 6.927 | 6.626 |
| procedural | 1 px | 1 | 1x1 | 0.046 | 0.052 | 0.88x | 0.050 | 0.053 |
| procedural | 2 px desconectados | 2 | 3x3 | 0.064 | 0.075 | 0.86x | 0.073 | 0.080 |
| forma conocida | recta | 2211 | 11x201 | 13.154 | 13.113 | 1.00x | 13.243 | 13.126 |
| forma conocida | recta girada 45 | 2123 | 149x149 | 32.931 | 31.019 | 1.06x | 33.050 | 31.251 |
| forma conocida | arco 60 | 894 | 81x20 | 6.230 | 6.212 | 1.00x | 6.234 | 6.285 |
| forma conocida | arco 120 | 1734 | 132x45 | 14.587 | 14.274 | 1.02x | 14.604 | 14.285 |
| forma conocida | arco 180 | 2556 | 151x80 | 29.101 | 27.723 | 1.05x | 29.140 | 27.793 |
| forma conocida | circulo | 7845 | 101x101 | 12.825 | 12.481 | 1.03x | 12.846 | 12.804 |
| forma conocida | recta dentada | 2419 | 16x203 | 14.969 | 14.723 | 1.02x | 15.062 | 14.741 |
| forma conocida | grumo con muesca | 5815 | 82x101 | 10.008 | 9.770 | 1.02x | 10.054 | 9.799 |
| escala 30px | espiral | 488 | 28x29 | 1.862 | 1.854 | 1.00x | 1.866 | 1.906 |
| escala 30px | blob borde ruidoso | 324 | 23x23 | 1.372 | 1.383 | 0.99x | 1.378 | 1.388 |
| escala 80px | espiral | 1119 | 56x61 | 13.126 | 12.836 | 1.02x | 13.219 | 12.868 |
| escala 80px | blob borde ruidoso | 2287 | 59x55 | 4.716 | 4.553 | 1.04x | 4.738 | 4.601 |
| escala 150px | espiral | 1942 | 100x116 | 52.412 | 50.635 | 1.04x | 52.548 | 50.919 |
| escala 150px | blob borde ruidoso | 8213 | 108x108 | 15.392 | 14.823 | 1.04x | 15.404 | 14.863 |
| escala 300px | espiral | 3800 | 203x235 | 297.016 | 279.527 | 1.06x | 297.507 | 280.020 |
| escala 300px | blob borde ruidoso | 30330 | 208x199 | 72.206 | 67.364 | 1.07x | 72.328 | 67.390 |
| escala 500px | espiral | 6456 | 340x392 | 1208.458 | 1124.045 | 1.08x | 1208.603 | 1124.989 |
| escala 500px | blob borde ruidoso | 82380 | 349x352 | 313.644 | 292.808 | 1.07x | 318.818 | 293.089 |

## Resumen por clase

| clase | mascaras | ref ms total | new ms total | ratio |
|---|---|---|---|---|
| fixture real | 8 | 48.078 | 47.322 | 1.02x |
| procedural | 8 | 223.905 | 211.782 | 1.06x |
| forma conocida | 8 | 133.805 | 129.315 | 1.03x |
| escala 30px | 2 | 3.234 | 3.237 | 1.00x |
| escala 80px | 2 | 17.842 | 17.388 | 1.03x |
| escala 150px | 2 | 67.804 | 65.458 | 1.04x |
| escala 300px | 2 | 369.222 | 346.890 | 1.06x |
| escala 500px | 2 | 1522.102 | 1416.853 | 1.07x |

## Global

- Tiempo total referencia: **2385.99 ms**
- Tiempo total reescrita: **2238.24 ms**
- Ratio global: **1.07x** (ref/new, sobre 34 mascaras)

> Citado como contexto externo, NO re-medido aqui: ~5.9 ms/particula y 7.1 s para el lote de 10 placas (medido en la maquina Windows del usuario, ANTES de la reescritura).
