"""Construye Manual_PolyX.html y Manual_PolyX.en.html desde cero.

A diferencia de generar_manual.py —que solo reemplazaba las capturas dentro de
un HTML ya escrito— este script arma el manual completo: texto, figuras
numeradas, indice y hoja de estilo. El texto vive en manual_texto_es.py y
manual_texto_en.py (esta misma carpeta); aqui solo esta el motor.

Las imagenes se empotran en base64, de modo que el archivo resultante se abre
con doble clic y sin conexion. Antes de empotrarlas se reescalan, porque las
capturas web van a 2x y sin reducir el manual pasaria de 25 MB.

Uso (desde la raiz del proyecto):
    .venv\\Scripts\\python.exe tools\\manual\\generar_manual.py            # ambos idiomas
    .venv\\Scripts\\python.exe tools\\manual\\generar_manual.py --idioma es
    .venv\\Scripts\\python.exe tools\\manual\\generar_manual.py --pdf      # ademas el PDF

Para regenerar las capturas de la interfaz antes de armar el manual:
    .venv\\Scripts\\python.exe tools\\manual\\capturar_manual.py --salida docs/assets/manual_screenshots/es
    POLYX_IDIOMA=en .venv\\Scripts\\python.exe tools\\manual\\capturar_manual.py --salida docs/assets/manual_screenshots/en
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent  # tools/manual: home of manual_texto_*.{es,en}
PROYECTO = RAIZ.parents[1]              # project root, inputs and outputs of the manual
CAPTURAS = PROYECTO / "docs" / "assets" / "manual_screenshots"
DESTINO_HTML = PROYECTO / "docs" / "manual"

# Ancho maximo al que se reescala cada figura antes de empotrarla. 1300 px
# cubre el ancho de columna del manual en pantallas normales y a 150 dpi de
# impresion; por encima solo se engorda el archivo.
ANCHO_MAX = 1300


# ────────────────────────────────────────────────────────────────────
#  Imagenes
# ────────────────────────────────────────────────────────────────────
def _carpeta_de(prefijo: str, idioma: str) -> Path:
    """Traduce el prefijo de un token de figura a su carpeta de capturas."""
    return {
        "ui": CAPTURAS / idioma,
        "web": CAPTURAS / "web",
        "win": CAPTURAS / "win",
        "mac": CAPTURAS / "mac",
    }[prefijo]


def imagen_base64(ruta: Path) -> tuple[str, int, int]:
    """Devuelve (data-uri, ancho, alto) de la imagen ya reescalada."""
    from PIL import Image

    with Image.open(ruta) as im:
        im = im.convert("RGB") if im.mode in ("P", "RGBA", "LA") else im
        if im.width > ANCHO_MAX:
            alto = round(im.height * ANCHO_MAX / im.width)
            im = im.resize((ANCHO_MAX, alto), Image.LANCZOS)
        buf = io.BytesIO()
        # PNG y no JPEG: son capturas de interfaz, llenas de texto fino y
        # bordes de 1 px, justo donde el artefacto de bloque de JPEG se ve.
        im.save(buf, "PNG", optimize=True)
        datos = buf.getvalue()
        return (
            "data:image/png;base64," + base64.b64encode(datos).decode("ascii"),
            im.width,
            im.height,
        )


# ────────────────────────────────────────────────────────────────────
#  Tokens de figura:  [[fig:web/w01_github_repo|Pie de figura]]
# ────────────────────────────────────────────────────────────────────
TOKEN = re.compile(r"\[\[fig:([a-z]+)/([A-Za-z0-9_\-.]+)\|(.*?)\]\]", re.S)

# Los dialogos que no se pueden fotografiar honestamente —el instalador de
# Python en un equipo que ya lo tiene muestra la pantalla de mantenimiento, no
# la de primera instalacion; y aqui no hay un Mac— van como esquema dibujado, y
# se marcan como tal para que nadie los confunda con una captura.
TOKEN_ESQ = re.compile(r"\[\[esquema:(\w+)\|(.*?)\]\]", re.S)


class Numerador:
    """Numera las figuras en el orden en que aparecen y guarda el indice."""

    def __init__(self) -> None:
        self.n = 0
        self.faltantes: list[str] = []

    def siguiente(self) -> int:
        self.n += 1
        return self.n


def expandir_figuras(html: str, idioma: str, num: Numerador, etiqueta: str) -> str:
    def repl(m: re.Match) -> str:
        prefijo, nombre, pie = m.group(1), m.group(2), m.group(3).strip()
        carpeta = _carpeta_de(prefijo, idioma)
        ruta = carpeta / (nombre if nombre.endswith(".png") else nombre + ".png")
        i = num.siguiente()
        if not ruta.exists():
            num.faltantes.append(str(ruta.relative_to(PROYECTO)))
            return (
                f'<figure class="fig falta"><div class="ph">Falta la captura '
                f'<code>{ruta.name}</code></div>'
                f'<figcaption><b>{etiqueta} {i}.</b> {pie}</figcaption></figure>'
            )
        uri, w, h = imagen_base64(ruta)
        return (
            f'<figure class="fig" id="fig-{i}">'
            f'<img src="{uri}" width="{w}" height="{h}" alt="{pie}" loading="lazy">'
            f'<figcaption><b>{etiqueta} {i}.</b> {pie}</figcaption></figure>'
        )

    return TOKEN.sub(repl, html)


def expandir_esquemas(html: str, esquemas: dict[str, str], num: Numerador,
                      etiqueta: str, palabra: str) -> str:
    def repl(m: re.Match) -> str:
        clave, pie = m.group(1), m.group(2).strip()
        i = num.siguiente()
        svg = esquemas.get(clave, "")
        if not svg:
            num.faltantes.append(f"esquema:{clave}")
            return (f'<figure class="fig falta"><div class="ph">Falta el esquema '
                    f'<code>{clave}</code></div></figure>')
        return (
            f'<figure class="fig esquema" id="fig-{i}">{svg}'
            f'<figcaption><span class="marca-esquema">{palabra}</span>'
            f'<b>{etiqueta} {i}.</b> {pie}</figcaption></figure>'
        )

    return TOKEN_ESQ.sub(repl, html)


def expandir(html: str, idioma: str, num: Numerador, txt) -> str:
    """Sustituye los dos tipos de token en un solo recorrido ordenado.

    Se procesan juntos y no uno detras de otro porque la numeracion tiene que
    seguir el orden de lectura: si se expandieran primero todas las fotos, un
    esquema intercalado recibiria un numero posterior al de la figura que va
    despues de el.
    """
    combinado = re.compile(TOKEN.pattern + "|" + TOKEN_ESQ.pattern, re.S)

    def repl(m: re.Match) -> str:
        if m.group(1) is not None:          # [[fig:...]]
            return expandir_figuras(m.group(0), idioma, num, txt.ETIQUETA_FIGURA)
        return expandir_esquemas(m.group(0), txt.ESQUEMAS, num,
                                 txt.ETIQUETA_FIGURA, txt.PALABRA_ESQUEMA)

    return combinado.sub(repl, html)


# ────────────────────────────────────────────────────────────────────
#  Hoja de estilo
# ────────────────────────────────────────────────────────────────────
CSS = r"""
:root{
  --ink:#1f2328; --ink2:#424a53; --ink3:#656d76; --muted:#8c959f;
  --rule:#d0d7de; --rule-soft:#eaeef2; --bg:#ffffff; --bg-soft:#f6f8fa;
  --accent:#0969da; --accent-d:#0550ae;
  --ok:#1f6b5e; --warn:#9a6700; --err:#cf222e; --vio:#6639ba;
  --pet:#e3342f; --pp:#ff8c00; --ldpe:#d4ac00;
  --serif: "Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans: "Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  --mono: "Cascadia Mono",Consolas,"SF Mono",Menlo,"DejaVu Sans Mono",monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.68;
  text-rendering:optimizeLegibility;
}

/* ── Estructura ─────────────────────────────────────────── */
.wrap{max-width:1180px;margin:0 auto;padding:0 28px}
.layout{display:grid;grid-template-columns:262px minmax(0,1fr);gap:52px;align-items:start}
@media (max-width:1000px){ .layout{grid-template-columns:1fr;gap:0} }

/* ── Portada ────────────────────────────────────────────── */
.portada{
  background:linear-gradient(168deg,#0d1117 0%,#15202b 46%,#1b2a3a 100%);
  color:#fff; padding:76px 0 64px; margin-bottom:44px;
  border-bottom:3px solid var(--accent);
}
.portada .wrap{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:44px;align-items:center}
@media (max-width:820px){ .portada .wrap{grid-template-columns:1fr} }
.marca{display:flex;align-items:center;gap:14px;margin-bottom:26px}
.marca .cuadro{
  width:46px;height:46px;border-radius:10px;background:var(--accent);
  display:grid;place-items:center;font-weight:800;font-size:24px;color:#fff;
  box-shadow:0 6px 20px rgba(9,105,218,.4);
}
.marca .txt b{display:block;font-size:15px;letter-spacing:.16em;line-height:1.2}
.marca .txt span{font-size:12.5px;color:#93a4b8;letter-spacing:.03em}
.portada h1{
  font-family:var(--serif); font-size:58px; line-height:1.04;
  margin:0 0 14px; font-weight:600; letter-spacing:-.018em;
}
.portada h1 em{font-style:normal;color:#79b8ff}
.portada .bajada{font-size:19px;color:#c6d4e3;margin:0 0 26px;max-width:56ch;line-height:1.55}
.portada .meta{font-size:14px;color:#8fa3b8;line-height:1.85}
.portada .meta b{color:#dce6f1;font-weight:600}
.sello{
  display:inline-flex;align-items:center;gap:9px;margin-bottom:22px;
  padding:6px 14px;border:1px solid rgba(121,184,255,.34);border-radius:999px;
  font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;color:#79b8ff;
}
.placa{width:250px;height:250px;flex:none}
@media (max-width:820px){ .placa{display:none} }

/* ── Indice lateral ─────────────────────────────────────── */
.toc{position:sticky;top:26px;max-height:calc(100vh - 52px);overflow-y:auto;
     font-size:13.6px;padding-right:8px}
.toc h2{font-size:11.5px;letter-spacing:.15em;text-transform:uppercase;
        color:var(--muted);margin:0 0 14px;font-weight:700}
.toc ol{list-style:none;margin:0;padding:0;counter-reset:t}
.toc ul{list-style:none;margin:0;padding:0}
.toc li{margin:0 0 1px}
/* Solo las secciones de primer nivel llevan numero: si contaran tambien los
   sub-enlaces, el indice saltaria de 3 a 9 sin que falte ninguna seccion. */
.toc>ol>li{counter-increment:t}
.toc a{
  display:grid;grid-template-columns:26px 1fr;gap:4px;
  padding:5px 8px;border-radius:6px;color:var(--ink2);text-decoration:none;
  border-left:2px solid transparent;
}
.toc a::before{content:counter(t);color:var(--muted);font-variant-numeric:tabular-nums;font-size:12.5px}
.toc a:hover{background:var(--bg-soft);color:var(--accent);border-left-color:var(--accent)}
.toc .sub{margin-left:26px;font-size:13px}
.toc .sub a{display:block;color:var(--ink3);padding:3px 8px}
.toc .sub a::before{content:none}
@media (max-width:1000px){ .toc{position:static;max-height:none;margin-bottom:36px;
  border-bottom:1px solid var(--rule);padding-bottom:24px} }

/* ── Texto ──────────────────────────────────────────────── */
main{min-width:0}
section{scroll-margin-top:20px;margin-bottom:56px}
h2{
  font-family:var(--serif); font-size:32px; line-height:1.2; font-weight:600;
  letter-spacing:-.012em; margin:0 0 6px; padding-top:8px;
}
h2 .num{
  display:inline-block;min-width:1.9em;color:var(--accent);
  font-family:var(--sans);font-size:19px;font-weight:700;vertical-align:.14em;
}
.sub-h2{color:var(--ink3);font-size:16px;margin:0 0 26px;
        padding-bottom:16px;border-bottom:1px solid var(--rule)}
h3{font-size:19.5px;font-weight:650;margin:36px 0 10px;letter-spacing:-.006em}
h4{font-size:16px;font-weight:650;margin:26px 0 8px;color:var(--ink2)}
p{margin:0 0 15px}
ul,ol{margin:0 0 16px;padding-left:22px}
li{margin:0 0 7px}
li>ul,li>ol{margin-top:7px}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
strong,b{font-weight:650}
code{
  font-family:var(--mono);font-size:.875em;background:var(--bg-soft);
  border:1px solid var(--rule-soft);border-radius:5px;padding:1px 5px;
  color:#0a3069;white-space:nowrap;
}
kbd{
  font-family:var(--sans);font-size:12.5px;font-weight:600;
  background:#fff;border:1px solid var(--rule);border-bottom-width:2px;
  border-radius:5px;padding:2px 7px;box-shadow:0 1px 0 rgba(0,0,0,.04);
  white-space:nowrap;color:var(--ink);
}
pre{
  font-family:var(--mono);font-size:13.2px;line-height:1.62;
  background:#0d1117;color:#d5dde6;border-radius:9px;
  padding:16px 18px;overflow-x:auto;margin:0 0 18px;
}
pre code{background:none;border:0;padding:0;color:inherit;white-space:pre}
pre .c{color:#7d8b99}    /* comentario */
pre .p{color:#79b8ff}    /* ruta / comando */
pre .o{color:#7ee787}    /* salida correcta */

/* ── Pasos numerados ────────────────────────────────────── */
.pasos{list-style:none;margin:0 0 8px;padding:0;counter-reset:paso}
.pasos>li{
  counter-increment:paso;position:relative;
  padding:0 0 30px 58px;margin:0;border-left:2px solid var(--rule-soft);
}
.pasos>li:last-child{border-left-color:transparent;padding-bottom:6px}
.pasos>li::before{
  content:counter(paso);position:absolute;left:-17px;top:-2px;
  width:32px;height:32px;border-radius:50%;
  background:var(--accent);color:#fff;display:grid;place-items:center;
  font-weight:700;font-size:15px;font-variant-numeric:tabular-nums;
  box-shadow:0 0 0 5px #fff;
}
.pasos>li>.t{font-weight:650;font-size:17.5px;display:block;margin:0 0 8px;letter-spacing:-.005em}

/* ── Figuras ────────────────────────────────────────────── */
.fig{margin:22px 0 26px;padding:0}
.fig img{
  display:block;width:100%;height:auto;border-radius:9px;
  border:1px solid var(--rule);
  box-shadow:0 1px 2px rgba(31,35,40,.05),0 8px 26px rgba(31,35,40,.08);
}
.fig figcaption{
  font-size:13.6px;color:var(--ink3);line-height:1.55;
  margin-top:11px;padding-left:2px;border-left:2px solid var(--accent);
  padding:2px 0 2px 11px;
}
.fig figcaption b{color:var(--ink2)}
.fig.falta .ph{
  border:2px dashed var(--err);border-radius:9px;padding:36px;
  text-align:center;color:var(--err);background:#fff5f5;
}
.fig.esquema img,.fig.esquema svg{box-shadow:none;background:var(--bg-soft)}
.fig svg{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:9px}
.marca-esquema{
  display:inline-block;background:var(--vio);color:#fff;font-size:10.5px;
  font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  padding:2px 8px;border-radius:4px;margin-right:8px;vertical-align:1px;
}

/* ── Avisos ─────────────────────────────────────────────── */
.aviso{
  border:1px solid var(--rule);border-left:4px solid var(--accent);
  background:var(--bg-soft);border-radius:0 9px 9px 0;
  padding:15px 18px;margin:20px 0;font-size:15.4px;
}
.aviso p:last-child{margin-bottom:0}
.aviso .et{
  display:block;font-size:11.5px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;margin-bottom:6px;color:var(--accent);
}
.aviso.ok{border-left-color:var(--ok)}      .aviso.ok .et{color:var(--ok)}
.aviso.warn{border-left-color:var(--warn);background:#fffbf0} .aviso.warn .et{color:var(--warn)}
.aviso.err{border-left-color:var(--err);background:#fff5f5}   .aviso.err .et{color:var(--err)}
.aviso.vio{border-left-color:var(--vio);background:#faf7ff}   .aviso.vio .et{color:var(--vio)}

/* ── Tablas ─────────────────────────────────────────────── */
.tabla-env{overflow-x:auto;margin:0 0 22px;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14.8px;min-width:440px}
th,td{text-align:left;padding:9px 13px;border-bottom:1px solid var(--rule-soft);vertical-align:top}
thead th{
  background:var(--bg-soft);font-size:12.4px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ink2);font-weight:700;
  border-bottom:2px solid var(--rule);
}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:#fafbfc}
td code,th code{white-space:nowrap}
.num-col{text-align:right;font-variant-numeric:tabular-nums}

/* ── Piezas menores ─────────────────────────────────────── */
.tag{
  display:inline-block;padding:2px 9px;border-radius:999px;
  font-size:12px;font-weight:650;letter-spacing:.02em;
}
.tag.pet{background:#ffe9e8;color:#a40e0e}
.tag.pp{background:#fff0da;color:#8a4b00}
.tag.ldpe{background:#fff8d6;color:#7a6300}
.tag.win{background:#e3f0ff;color:#0550ae}
.tag.mac{background:#f0eaff;color:#4b2a94}

.modulos{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:14px;margin:22px 0 26px}
.mod{border:1px solid var(--rule);border-radius:10px;padding:16px 17px;background:var(--bg-soft)}
.mod .ic{font-size:24px;line-height:1;display:block;margin-bottom:9px}
.mod b{display:block;font-size:16.5px;margin-bottom:5px}
.mod span{font-size:14px;color:var(--ink3);line-height:1.5}

.dl{margin:0 0 18px}
.dl dt{font-weight:650;margin:14px 0 3px;font-size:15.6px}
.dl dd{margin:0 0 0 0;color:var(--ink2);font-size:15.4px}

.atajos{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:0 30px}
.atajos dl{margin:0}
.atajos dt{float:left;clear:left;margin:0 10px 6px 0;font-weight:400}
.atajos dd{margin:0 0 6px;padding-top:1px;font-size:14.8px;color:var(--ink2);overflow:hidden}

hr.sep{border:0;border-top:1px solid var(--rule-soft);margin:34px 0}

footer.pie{
  margin-top:60px;border-top:1px solid var(--rule);
  padding:26px 0 60px;font-size:13.8px;color:var(--ink3);
}
footer.pie b{color:var(--ink2)}

/* ── Impresion ──────────────────────────────────────────── */
@page{ size:A4; margin:17mm 15mm 18mm; }
@media print{
  body{font-size:10.4pt;line-height:1.5}
  .toc,.portada .placa{display:none}
  .layout{display:block}
  .wrap{max-width:none;padding:0}
  /* Sin margenes negativos: Chrome no pinta fondo fuera de la caja de pagina,
     asi que el "sangrado completo" solo conseguia desplazar el texto y que se
     recortara por la izquierda. Un panel con su propio relleno es predecible. */
  .portada{
    background:#0d1117 !important; color:#fff !important;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
    padding:30mm 14mm; margin:0 0 10mm; border-radius:3mm;
    min-height:238mm; display:flex; align-items:center;
    break-after:page; border-bottom:0;
  }
  .portada h1{font-size:34pt}
  .portada .bajada{font-size:12pt}
  /* El circulo del paso vive en left:-17px; sin este relleno queda fuera de la
     caja de pagina y la impresora lo recorta. */
  .pasos{padding-left:22px}
  .pasos>li{padding-left:46px}
  section{break-before:page;margin-bottom:0}
  section:first-of-type{break-before:auto}
  h2{font-size:19pt;break-after:avoid}
  h3,h4{break-after:avoid}
  /* Un paso puede ser mas alto que una pagina —lleva tres figuras—, y pedir que
     no se parta dejaba la pagina anterior casi en blanco. Se protegen las piezas
     que si caben enteras, y se evita que un titulo quede huerfano al pie. */
  .fig,figure,pre,.aviso,tr,.mod{break-inside:avoid}
  thead{display:table-header-group}
  .pasos>li>.t{break-after:avoid}
  .dl dt{break-after:avoid}
  .fig img{box-shadow:none;max-height:118mm;width:auto;max-width:100%}
  .fig figcaption{font-size:8.6pt}
  a{color:var(--ink);text-decoration:none}
  pre{background:#f6f8fa !important;color:#1f2328 !important;
      border:1px solid var(--rule);-webkit-print-color-adjust:exact;print-color-adjust:exact}
  pre .c{color:#57606a}  pre .p{color:#0550ae}  pre .o{color:#1f6b5e}
  thead th{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .aviso,.tag,.mod{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  footer.pie{padding-bottom:0}
}
"""

# La placa de la portada: las particulas son las tres clases del modelo.
PLACA_SVG = """
<svg class="placa" viewBox="0 0 250 250" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Placa Petri con microplasticos fluorescentes">
  <defs>
    <radialGradient id="pg" cx="42%" cy="36%">
      <stop offset="0%" stop-color="#22303f"/><stop offset="100%" stop-color="#0a1017"/>
    </radialGradient>
    <filter id="gl"><feGaussianBlur stdDeviation="3.4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <circle cx="125" cy="125" r="118" fill="url(#pg)" stroke="#3d4d5e" stroke-width="2.5"/>
  <circle cx="125" cy="125" r="112" fill="none" stroke="#2a3947" stroke-width="1.2"/>
  <g filter="url(#gl)" opacity=".95">
    <ellipse cx="92"  cy="80"  rx="15" ry="11" fill="#e3342f" transform="rotate(-22 92 80)"/>
    <ellipse cx="163" cy="104" rx="11" ry="9"  fill="#e3342f"/>
    <ellipse cx="108" cy="163" rx="13" ry="10" fill="#e3342f" transform="rotate(34 108 163)"/>
    <ellipse cx="186" cy="152" rx="8"  ry="7"  fill="#e3342f"/>
    <rect x="70"  y="112" width="9"  height="46" rx="4.5" fill="#ffd700" transform="rotate(16 74 135)"/>
    <rect x="129" y="146" width="8"  height="40" rx="4"   fill="#ffd700" transform="rotate(-28 133 166)"/>
    <ellipse cx="182" cy="76"  rx="11" ry="10" fill="#ff8c00"/>
    <ellipse cx="63"  cy="182" rx="9"  ry="8"  fill="#ff8c00"/>
  </g>
  <g fill="none" stroke="#79b8ff" stroke-width="1.6" opacity=".8">
    <rect x="74" y="65" width="37" height="31" rx="2"/>
    <rect x="146" y="92" width="35" height="26" rx="2"/>
    <rect x="58" y="106" width="34" height="60" rx="2"/>
  </g>
  <text x="125" y="238" text-anchor="middle" fill="#5b7185"
        font-family="ui-monospace,monospace" font-size="11">100 mm</text>
</svg>
"""


# ────────────────────────────────────────────────────────────────────
#  Montaje
# ────────────────────────────────────────────────────────────────────
def construir(idioma: str) -> tuple[str, list[str]]:
    modulo = f"manual_texto_{idioma}"
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    txt = __import__(modulo)

    num = Numerador()
    partes: list[str] = []
    toc: list[str] = []

    for i, sec in enumerate(txt.SECCIONES, start=1):
        cuerpo = expandir(sec["html"], idioma, num, txt)
        sub = sec.get("sub", "")
        partes.append(
            f'<section id="{sec["id"]}">\n'
            f'  <h2><span class="num">{i}</span>{sec["titulo"]}</h2>\n'
            + (f'  <p class="sub-h2">{sub}</p>\n' if sub else "")
            + cuerpo
            + "\n</section>"
        )
        enlaces_sub = "".join(
            f'<li><a href="#{a}">{t}</a></li>' for a, t in sec.get("anclas", [])
        )
        toc.append(
            f'<li><a href="#{sec["id"]}">{sec["titulo"]}</a>'
            + (f'<ul class="sub">{enlaces_sub}</ul>' if enlaces_sub else "")
            + "</li>"
        )

    html = f"""<!doctype html>
<html lang="{idioma}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{txt.TITULO_DOC}</title>
<meta name="author" content="Cristofher Ferrada">
<meta name="description" content="{txt.DESCRIPCION}">
<style>{CSS}</style>
</head>
<body>

<header class="portada">
  <div class="wrap">
    <div>
      <div class="marca">
        <div class="cuadro">P</div>
        <div class="txt"><b>POLY-X</b><span>Microplastics analytics suite</span></div>
      </div>
      <div class="sello">{txt.SELLO}</div>
      <h1>{txt.TITULO_H1}</h1>
      <p class="bajada">{txt.BAJADA}</p>
      <p class="meta">{txt.META}</p>
    </div>
    {PLACA_SVG}
  </div>
</header>

<div class="wrap">
  <div class="layout">
    <nav class="toc">
      <h2>{txt.ETIQUETA_INDICE}</h2>
      <ol>{"".join(toc)}</ol>
    </nav>
    <main>
      {"".join(partes)}
      <footer class="pie">{txt.PIE}</footer>
    </main>
  </div>
</div>

</body>
</html>
"""
    return html, num.faltantes


# ────────────────────────────────────────────────────────────────────
def a_pdf(html: Path, pdf: Path) -> bool:
    """Imprime el manual a PDF con Chrome o Edge en modo headless.

    Se usa el navegador y no una libreria porque el manual ya esta pensado
    para imprimirse desde un navegador: asi el PDF sale identico a lo que ve
    quien pulsa Ctrl+P, sin un segundo motor de maquetacion que mantener.
    """
    candidatos = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    exe = next((c for c in candidatos if c.exists()), None)
    if exe is None:
        print("  [AVISO] no encontre Chrome ni Edge; abre el HTML y usa Ctrl+P -> Guardar como PDF")
        return False
    cmd = [
        str(exe), "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf}", "--print-to-pdf-no-header",
        html.resolve().as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if pdf.exists() and pdf.stat().st_size > 0:
        print(f"  [OK]  {pdf.name}  ({pdf.stat().st_size/1_048_576:.1f} MB)")
        return True
    print(f"  [ERR] no se genero el PDF: {r.stderr[:300]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Construye el manual de Poly-X")
    ap.add_argument("--idioma", default="ambos", choices=["es", "en", "ambos"])
    ap.add_argument("--pdf", action="store_true", help="ademas, imprimir a PDF")
    args = ap.parse_args()

    idiomas = ["es", "en"] if args.idioma == "ambos" else [args.idioma]
    for idioma in idiomas:
        destino = DESTINO_HTML / (
            "Manual_PolyX.html" if idioma == "es" else f"Manual_PolyX.{idioma}.html"
        )
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idioma}] {destino.name}")
        html, faltantes = construir(idioma)
        destino.write_text(html, encoding="utf-8")
        print(f"  [OK]  {destino.stat().st_size/1_048_576:.1f} MB")
        if faltantes:
            print(f"  [AVISO] {len(faltantes)} captura(s) sin archivo:")
            for f in faltantes:
                print(f"          {f}")
        if args.pdf:
            a_pdf(destino, destino.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
