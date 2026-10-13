# Contribuir a Poly-X

## Qué hace el CI

Un único job en GitHub Actions (`.github/workflows/ci.yml`) que corre en
**cada PR** y en cada push a `main`:

1. Instala las librerías de Qt que hacen falta sin pantalla (`libegl1`,
   `libgl1`, `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`).
2. Instala `requirements.txt` **con los pins exactos** (`numpy==1.26.4`,
   `ultralytics==8.3.40`, `PySide6==6.7.2`, …) sobre Python 3.11. Así, si
   alguien sube una dependencia, el CI avisa aquí y no en el estudio.
3. Corre la suite completa: `python -m pytest -q` con `QT_QPA_PLATFORM=offscreen`
   (la suite no abre ventanas; PySide6 es lo único que pide plataforma).

No hay etapa de lint (no hay linter configurado) ni de build (la app es de
escritorio y los instaladores son scripts del propio repo).

## Cómo correr las mismas pruebas en local

### Windows (la máquina de desarrollo)

```batch
.venv\Scripts\python.exe -m pytest -q
```

### macOS / Linux

```bash
python -m pytest -q          # dentro del .venv del proyecto
```

- La suite resuelve idioma al arrancar y usa **español** salvo que pongas
  `POLYX_IDIOMA` (ver `tests/conftest.py`). Si el informe sale en inglés y
  falla un test que lee texto del informe, no toques el test: eso ya está
  fijado.
- No hace falta modelo, GPU ni dataset: los pesos `.pt` y las imágenes no
  entran en la suite (están en .gitignore).

## Depurar un build rojo

1. Abre el PR fallido en GitHub y mira el step que está en rojo:
   - `Install pinned requirements` → alguien cambió un pin. El error de pip
     dice cuál. **No** relajes el pin: `numpy>=2` rompe la validación de
     ultralytics 8.3.40 (le quitaron `np.trapz`), y subir `PySide6` puede
     cambiar el comportamiento de los diálogos Qt.
   - `Run test suite` → la prueba fallida se imprime con su aserción. La
     suite es determinista: si pasa en local, debería pasar en CI; si falla
     solo en CI, revisa el idioma (`POLYX_IDIOMA`) y el locale.
2. Reproduce en local con el mismo comando del step (Linux arriba).
3. Sube la corrección a la misma rama: el job se vuelve a correr solo, y el
   push nuevo cancela la corrida anterior.

## Antes de abrir un PR

- La suite completa pasa en local (`pytest -q`, cero fallos).
- Los comentarios y el código en español, docstrings en inglés (ver
  `AGENTS.md` y `CLAUDE.md`).
