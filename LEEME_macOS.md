# Poly-X on macOS

A Nile Red fluorescence microplastics detection suite.
Cristofher Ferrada · PhD (c) in Science, Chemistry · PUCV

---

## Installation (only once)

1. Download the project and unzip it wherever you like (for example, in Documents).
2. **Right-click** on `Lanzar_macOS.command` → **Open** → **Open**.
3. Wait 10-15 minutes. Everything installs by itself.

From then on, just double-click `Lanzar_macOS.command` every time you want to use it.

When installation ends it offers to leave **`Poly-X.app` on the Desktop**:
accept it and use that icon for day-to-day work — it opens without a Terminal window.

---

## Opening without the black Terminal window

`Lanzar_macOS.command` **always** leaves a Terminal window open behind the
program. It is not a bug: a `.command` is, by definition, a script that
Finder hands over to Terminal, and that window lives as long as the program
lives. If you close it, you close Poly-X.

To open it without that window, double-click **`crear_icono_macOS.command`**. It
creates a `Poly-X.app` —Desktop, Applications or next to the program, your
choice— and from then on you open from that icon. It is the equivalent of
`Poly-X.vbs` on Windows.

- Takes **one second** and weighs a few KB: it bundles nothing, it just calls
  the environment that is already installed. Nothing to do with
  `construir_app_macOS.command`.
- Because it is created **on your own Mac**, it carries no quarantine mark, so
  Gatekeeper never complains: normal double-click from the very first time.
- With no console there is nowhere to read errors, so everything the program
  writes goes to `~/Library/Logs/Poly-X.log`.
- The program's path is written inside the `.app`. **If you move the Poly-X
  folder, run `crear_icono_macOS.command` again** from its new location.

Keep `Lanzar_macOS.command` around: it is the one to open when something
breaks, because it shows the error on screen instead of hiding it in a log.

### ⚠️ `No se puede abrir porque es de un desarrollador no identificado`

That is expected and **does not mean it is broken**. macOS blocks, by
default, any script downloaded from the internet that is not signed by a
developer registered with Apple.

**Fix:** right-click the file → **Open** → confirm **Open** in the dialog.
Only needed the first time; after that the double-click works.

If it still resists, in Terminal:

```bash
xattr -d com.apple.quarantine Lanzar_macOS.command
```

### Prerequisite

**Python 3.9 or newer.** Check it in Terminal with `python3 --version`.
If you do not have it, download it from [python.org/downloads](https://www.python.org/downloads/).

---

## Performance: what to expect

Depends on the Mac's processor, and the difference is big.

| Mac | Speed-up | Detection |
|---|---|---|
| **Apple Silicon** (M1, M2, M3, M4) | Built-in GPU through MPS | Fast |
| **Intel** | CPU only | Slow: counts ~1 min per photo in batches with tiling |

The installer detects which one you have and configures the device by
itself. There is nothing to touch, but it is worth knowing before launching
a batch of 500 photos on an Intel Mac.

> **Note for Intel Macs:** PyTorch stopped shipping releases for Intel
> processors after 2.2.2, so the installer pins that version. It is the last
> one that exists for that architecture and it works correctly.

---

## Updating

Double-click `actualizar_macOS.command`. It downloads the new release from
GitHub and overwrites only the program files: **your models, results and
data are left untouched**.

Poly-X also shows an `Actualizar` button inside it when a new release exists.

---

## Bundling with PyInstaller (optional, and almost never needed)

`construir_app_macOS.command` packs PyTorch, ultralytics and the whole
interpreter into a self-contained `.app`, to carry over to a Mac with
nothing installed.

> Do not confuse it with `crear_icono_macOS.command`, which is what you want
> if you only want to open without Terminal: that one takes a second, this
> one 15 to 40 minutes.

**You do not need it to use Poly-X**, and it has real downsides:

- Takes 15-40 minutes and weighs 2-5 GB (PyTorch goes inside the bundle).
- Without an Apple signature (USD 99/year) Gatekeeper blocks it just the
  same on another Mac, so it **does not avoid** the
  "unidentified developer" warning either.
- Packing PyTorch with PyInstaller is fragile.

The recommended way is `Lanzar_macOS.command`.

---

## If something fails

| Symptom | What happens |
|---|---|
| `bad interpreter: /bin/bash^M` | The file arrived with Windows line endings. Download it again from GitHub, do not copy it through WhatsApp/email. |
| Double-click and nothing happens | The executable permission is missing: `chmod +x Lanzar_macOS.command` in Terminal. |
| The `Poly-X.app` bounces in the Dock and quits | Open `~/Library/Logs/Poly-X.log`: the error the Terminal would have shown is there. |
| The `.app` reports that it cannot find the installation | You moved the program folder. Run `crear_icono_macOS.command` again from its new location. |
| `No se encontró Python 3.9 o superior` | Install it from python.org/downloads. |
| PyTorch fails to install | Usually the connection. Run the launcher again: it resumes from where it left off. |
| Detection finds nothing | Lower confidence to 0.10 in `Parámetros`. |

---

## What has been tested and what has not

Being honest about the state of this release:

- **Verified on Windows:** all the shared logic — the app, the report, the
  particle measurement, the automated test suite.
- **Written for macOS but still untested on a Mac:** the launchers, the
  updater, MPS detection and the `.app` bundling.

If you find a bug, write down the exact message from the Terminal window:
that is what makes it fixable.
