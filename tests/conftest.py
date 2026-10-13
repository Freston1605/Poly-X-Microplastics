"""Configuracion compartida de la suite.

El idioma del programa se resuelve una vez a partir del entorno (ver
``polyx.core.i18n``) y se cachea en el modulo, asi que varios tests que no
fuercen idioma propio --p. ej. ``test_asignacion`` lee texto del informe--
acaban dependiendo del locale de la maquina donde corra pytest. Con un
locale en ingles el informe sale en ingles y la prueba falla; en CI el
locale suele estar vacio y pasa por casualidad.

Aqui fijamos espanol como idioma de la suite, explicito y reproducible. Las
pruebas que si prueban idiomas --``test_informe_bilingue``-- fijan el suyo
mediante ``POLYX_IDIOMA`` directamente, y ``setdefault`` respeta lo que alguien
pida a mano.
"""
import os

os.environ.setdefault("POLYX_IDIOMA", "es")
