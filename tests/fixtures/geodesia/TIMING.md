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
| procedural | blob borde ruidoso | 13033 | 145x138 | 29.816 | 27.966 | 1.07x | 29.860 | 28.008 |
| procedural | cinta sinuosa | 2216 | 132x166 | 52.668 | 48.595 | 1.08x | 52.834 | 48.691 |
| procedural | espiral 1.6 vueltas | 2526 | 134x155 | 108.249 | 102.066 | 1.06x | 108.357 | 102.666 |
| procedural | grumo con huecos | 15405 | 145x145 | 30.732 | 28.821 | 1.07x | 30.913 | 29.000 |
| procedural | erodada bernoulli | 2855 | 47x70 | 6.013 | 5.789 | 1.04x | 6.029 | 5.818 |
| procedural | dos componentes | 4906 | 141x57 | 7.130 | 6.733 | 1.06x | 7.166 | 6.748 |
| procedural | 1 px | 1 | 1x1 | 0.047 | 0.052 | 0.91x | 0.049 | 0.053 |
| procedural | 2 px desconectados | 2 | 3x3 | 0.066 | 0.074 | 0.89x | 0.075 | 0.076 |
| forma conocida | recta | 2211 | 11x201 | 13.612 | 13.287 | 1.02x | 13.769 | 13.326 |
| forma conocida | recta girada 45 | 2123 | 149x149 | 34.557 | 31.804 | 1.09x | 34.685 | 31.864 |
| forma conocida | arco 60 | 894 | 81x20 | 6.422 | 6.364 | 1.01x | 6.497 | 6.375 |
| forma conocida | arco 120 | 1734 | 132x45 | 15.218 | 14.503 | 1.05x | 15.299 | 14.537 |
| forma conocida | arco 180 | 2556 | 151x80 | 29.162 | 28.403 | 1.03x | 29.473 | 28.588 |
| forma conocida | circulo | 7845 | 101x101 | 13.119 | 12.602 | 1.04x | 13.242 | 12.661 |
| forma conocida | recta dentada | 2419 | 16x203 | 15.646 | 15.218 | 1.03x | 15.870 | 15.346 |
| forma conocida | grumo con muesca | 5815 | 82x101 | 10.170 | 9.969 | 1.02x | 10.193 | 9.999 |
| escala 30px | espiral | 488 | 28x29 | 1.951 | 1.894 | 1.03x | 1.968 | 1.898 |
| escala 30px | blob borde ruidoso | 324 | 23x23 | 1.441 | 1.433 | 1.01x | 1.457 | 1.481 |
| escala 80px | espiral | 1119 | 56x61 | 13.574 | 13.343 | 1.02x | 13.620 | 13.362 |
| escala 80px | blob borde ruidoso | 2287 | 59x55 | 4.896 | 4.743 | 1.03x | 4.926 | 4.786 |
| escala 150px | espiral | 1942 | 100x116 | 54.195 | 52.318 | 1.04x | 54.654 | 52.429 |
| escala 150px | blob borde ruidoso | 8213 | 108x108 | 15.591 | 14.943 | 1.04x | 15.613 | 14.956 |
| escala 300px | espiral | 3800 | 203x235 | 311.507 | 287.076 | 1.09x | 311.776 | 287.944 |
| escala 300px | blob borde ruidoso | 30330 | 208x199 | 75.943 | 69.606 | 1.09x | 75.965 | 70.062 |
| escala 500px | espiral | 6456 | 340x392 | 1250.674 | 1161.009 | 1.08x | 1255.828 | 1169.115 |
| escala 500px | blob borde ruidoso | 82380 | 349x352 | 327.046 | 304.290 | 1.07x | 327.285 | 312.390 |

## Resumen por clase

| clase | mascaras | ref ms total | new ms total | ratio |
|---|---|---|---|---|
| procedural | 8 | 234.720 | 220.096 | 1.07x |
| forma conocida | 8 | 137.905 | 132.151 | 1.04x |
| escala 30px | 2 | 3.393 | 3.327 | 1.02x |
| escala 80px | 2 | 18.471 | 18.086 | 1.02x |
| escala 150px | 2 | 69.786 | 67.262 | 1.04x |
| escala 300px | 2 | 387.450 | 356.682 | 1.09x |
| escala 500px | 2 | 1577.720 | 1465.299 | 1.08x |

## Global

- Tiempo total referencia: **2429.44 ms**
- Tiempo total reescrita: **2262.90 ms**
- Ratio global: **1.07x** (ref/new, sobre 26 mascaras)

> Citado como contexto externo, NO re-medido aqui: ~5.9 ms/particula y 7.1 s para el lote de 10 placas (medido en la maquina Windows del usuario, ANTES de la reescritura).
