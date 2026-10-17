# Poly-X — instrucciones para agentes

**Toda la documentación del proyecto está en [CLAUDE.md](CLAUDE.md).** Léelo
antes de tocar nada: arquitectura, módulos, flujo de datos, clases clave,
paleta, parámetros y las decisiones que ya se tomaron y por qué.

Este archivo existe solo porque algunas herramientas buscan `AGENTS.md` por
convención. **No dupliques contenido aquí.**

> Hasta septiembre de 2026 este archivo era una copia recortada de CLAUDE.md, y
> se quedó atrás: seguía diciendo que el Etiquetador y el Visor estaban «en
> construcción» —llevan cerrados desde agosto— y que el PP fluoresce naranjo,
> cuando la medición mostró que es amarillo verdoso. Mantener dos copias del
> mismo documento no sale gratis: una de las dos miente, y no se sabe cuál.

## Lo mínimo antes de escribir código

- **Comentarios y variables en español**; docstrings en inglés.
- **Rutas siempre con `pathlib.Path`**, nunca strings.
- **Colores desde `polyx/gui/theme.py`** (`T.ACCENT`, `T.OK`…), nunca a mano.
- **Nada específico de un sistema operativo fuera de `polyx/model/plataforma.py`.**
- Imágenes con `cv2.imdecode`/`imencode` sobre `np.fromfile`, no `imread`/`imwrite`:
  en Windows estos fallan en silencio con rutas acentuadas.
- Los tests se corren con `.venv\Scripts\python.exe -m pytest -q` y **pasan los
  128** antes de dar nada por terminado.
