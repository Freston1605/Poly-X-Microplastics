# Evidence — stack

Dimension: stack
Questions assigned: Which PyInstaller hooks are needed for ultralytics/torch on each OS?; Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers?; Singularity vs Apptainer vs conda-pack for the cluster image?

## Finding: pyinstaller-hooks-contrib ships first-party hooks for both torch and ultralytics (2024.9+)

- **Claim**: Since hooks-contrib 2024.9, dedicated hooks `hook-torch` and `hook-ultralytics` exist; the ultralytics hook was added in #786 and includes a functional test. No custom hook is strictly required if the installed hooks-contrib version is current, but data files for ultralytics (cfg/default.yaml) must still be collected explicitly on some setups.
- **Source**: https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/v2026.5/CHANGELOG.rst (entries "Add hook for ultralytics package. (#786)" under 2024.9; "Add hook for torch" historical); https://github.com/pyinstaller/pyinstaller/issues/8032 (rokm comment: collect_data_files('ultralytics'))
- **Confidence**: high
- **Why it matters here**: Poly-X uses ultralytics 8.3.40 + torch; a PyInstaller build on a current hooks-contrib will get hidden imports collected automatically, but the model/weights and ultralytics YAML configs still need `datas`/`collect_data_files`.

## Finding: torch hook has OS-specific binary handling (Linux symlink suppression, Windows MKL DLLs)

- **Claim**: The torch hook behaves differently per OS: on Linux it suppresses creation of symlinks for shared libs in torch/lib (fixing libtorch_cuda_linalg.so discovery) and is needed for PyInstaller >= 6.11; on Windows it collects MKL/intel-openmp/tbb DLLs (with path-separator matching bugs fixed in 2025.1+); on macOS it relies on findlibs override. GPU (CUDA) nvidia.* package hooks also suppress symlinks on Linux.
- **Source**: https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/v2026.5/CHANGELOG.rst (2024.11 Linux torch/nvidia.* entries; 2025.1 torch MKL issue #879; 2025.9 torch Anaconda/mkl update)
- **Confidence**: high
- **Why it matters here**: A cross-OS CI build needs PyInstaller >= 6.11 and current hooks-contrib on Linux to avoid CUDA DLL discovery failures; Windows builds must verify MKL DLL collection.

## Finding: ultralytics data files (cfg/default.yaml) are not auto-collected by the hook in older setups

- **Claim**: PyInstaller builds of ultralytics apps fail at runtime with FileNotFoundError for `ultralytics/cfg/default.yaml` unless `collect_data_files('ultralytics')` is added to `datas` (rokm's recommended fix in issue #8032). The hook added in 2024.9 may collect some, but multiple community reports still require explicit data collection.
- **Source**: https://github.com/pyinstaller/pyinstaller/issues/8032 (rokm comment 2023-10-24)
- **Confidence**: medium
- **Why it matters here**: Poly-X's detector imports ultralytics for inference; the frozen exe will crash on first model load without the YAML configs bundled.

## Finding: PyInstaller does not support module:entry_point syntax; a stub script is required

- **Claim**: PyInstaller cannot be pointed at a package/module directly (issue #8074, open). The supported approach is a small stub script outside the package that imports and calls the entry point, or freezing a `__main__.py` with no relative imports. The existing `lanzar_polyx.py` stub is exactly this pattern.
- **Source**: https://github.com/pyinstaller/pyinstaller/issues/8074 (bwoodsend/rokm comments); https://github.com/pyinstaller/pyinstaller/issues/2560
- **Confidence**: high
- **Why it matters here**: The milestone's `src/polyx` layout must keep a top-level entry-point stub (or `__main__.py`) for PyInstaller; `lanzar_polyx.py` already serves this role and can be relocated.

## Finding: src/ layout does not inherently break PyInstaller if the package is installed or pathex is set

- **Claim**: PyInstaller adds the directory of the entry-point script's top-level package to sys.path. With a src layout, if the entry-point stub lives at repo root (outside src/), PyInstaller will not find `polyx` unless src/ is on pathex or the package is pip-installed (editable or wheel). The recommended fix is either install the package (`pip install -e .`) before building, or set `pathex=["src"]` in the spec, or move the stub into src/polyx and keep relative-import-free code.
- **Source**: https://github.com/pyinstaller/pyinstaller/issues/7742 (bwoodsend: install package with pip then PyInstaller); https://github.com/pyinstaller/pyinstaller/issues/6713 (get_path_to_toplevel_modules logic); https://pyinstaller.org/en/latest/spec-files.html (pathex semantics)
- **Confidence**: high
- **Why it matters here**: The existing `lanzar_polyx.py` does `from polyx.gui.launcher import main`; under src layout this import fails unless src/ is on sys.path. The VBS launcher runs `python -m polyx.launcher` from repo root, which also fails to find `polyx` unless installed or PYTHONPATH includes src/.

## Finding: Poly-X.vbs assumes polyx/ sits at repo root — breaks under src/ layout

- **Claim**: Poly-X.vbs checks `fso.FolderExists(raiz & "\polyx")` and runs `pythonw -m polyx.launcher` with cwd=raiz. After moving to src/polyx, that folder check fails and the module is not importable from repo root without an installed package or PYTHONPATH. The launcher must be updated to point at src/ or be replaced by the installed console script.
- **Source**: repo file /var/home/Franco/Projects/Poly-X-Microplastics/Poly-X.vbs lines 27-40
- **Confidence**: high
- **Why it matters here**: SC1 requires `python -m polyx` / `python -m polyx.gui` to work on a clean install; the VBS launcher is a Windows-only convenience that must be migrated to use the installed entry point or src path.

## Finding: Apptainer (renamed Singularity) is the HPC-standard container runtime; conda-pack is a different, lighter artifact

- **Claim**: Apptainer (formerly Singularity, moved to Linux Foundation 2021) is the HPC-native container runtime: no daemon, runs as the user, GPU support via `--nv`, SLURM integration, bind-mounts all filesystems by default. Singularity (Sylabs) and SingularityCE are the same lineage. conda-pack is not a container runtime but a tool to package a conda environment into a relocatable tarball; it does not bundle OS/CUDA libs and is sensitive to host glibc/CUDA. Most HPC docs (NHR@FAU, Delta, FASRC, CSC, WVU) recommend Apptainer over conda on shared filesystems due to inode/quota and I/O load.
- **Source**: https://rc.dartmouth.edu/hpc-cookbook/recipes/containers/intro/ (Docker vs Apptainer table); https://doc.nhr.fau.de/environment/apptainer/ ; https://csc-training.github.io/csc-env-eff/hands-on/singularity/singularity_extra_replicating-conda.html ; https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/containers.html
- **Confidence**: high
- **Why it matters here**: INTENT SC7 mandates SLURM + Singularity/Apptainer; choosing Apptainer (the Linux Foundation fork most clusters ship) is the safest default, with a Docker-derived .def file so the image is reproducible and GPU-enabled via --nv.

## Finding: conda environments directly on HPC filesystems are discouraged; containerize them

- **Claim**: Multiple HPC centers explicitly discourage installing conda directly on shared filesystems (inode quotas, I/O load, system-library mismatches) and recommend either Apptainer containers (wrapping conda or pip) or modules. conda-pack can ship a packed env but still carries tens of thousands of files and host-coupling.
- **Source**: https://docs.hpc.uio.no/software/userinstallsw/conda.html ; https://csc-training.github.io/csc-env-eff/hands-on/singularity/singularity_extra_replicating-conda.html ; https://wvuhpc.github.io/2025-Introduction-HPC/401-software.html
- **Confidence**: high
- **Why it matters here**: For Poly-X's HPCC track, an Apptainer SIF built from a Docker base (pytorch/pytorch or nvidia/cuda) + requirements.txt is the recommended, reproducible artifact; conda-pack is a fallback only if the cluster lacks Apptainer.

## Assigned questions — answers

- Which PyInstaller hooks are needed for ultralytics/torch on each OS? → Rely on pyinstaller-hooks-contrib's built-in `hook-torch` and `hook-ultralytics` (available since 2024.9) rather than writing custom hooks; additionally call `collect_data_files('ultralytics')` for cfg/default.yaml. OS-specific caveats: Linux needs PyInstaller >= 6.11 and current hooks-contrib so the torch/nvidia.* hooks suppress symlink creation (libtorch_cuda_linalg.so, cudnn); Windows needs MKL/intel-openmp/tbb DLL collection (verify on 2025.1+); macOS needs findlibs override (provided by hooks-contrib runtime hook). Source: hooks-contrib CHANGELOG 2024.9–2025.9; issues #8032, #879.
- Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers? → Yes, unless adapted. `lanzar_polyx.py` (`from polyx.gui.launcher import main`) and Poly-X.vbs (`python -m polyx.launcher` + `polyx/` folder check) both assume `polyx` is importable from repo root. Under src/, they must be updated to add src/ to sys.path/PYTHONPATH or be replaced by the installed console script (`python -m polyx`). Source: repo files; PyInstaller issues #7742/#6713.
- Singularity vs Apptainer vs conda-pack for the cluster image? → Use Apptainer (the Linux Foundation fork of Singularity; same commands) as the runtime, building a SIF from a Docker base image (pytorch/pytorch or nvidia/cuda) + requirements.txt via a .def file, run with `--nv` for GPU. Singularity/SingularityCE are the same lineage; conda-pack is a different, lighter-weight packaging tool, not a container runtime, and is discouraged for shared HPC filesystems. Source: HPC docs (NHR@FAU, Delta, CSC, Dartmouth, UiO).

## Dead ends

- Writing a custom PyInstaller hook for ultralytics/torch from scratch — unnecessary; hooks-contrib already provides and tests them (2024.9+). Only data-file collection and OS-specific binary quirks need handling.
- Keeping Poly-X.vbs unchanged after src/ migration — fails the folder-existence check and module import; must be updated or dropped in favor of installed entry point.
- Installing conda directly on HPCC shared filesystem — explicitly discouraged by multiple centers (inode quotas, I/O load); use an Apptainer SIF instead.
- Pointing PyInstaller at `src/polyx` as a directory — not supported (issue #8074); a stub script / `__main__.py` is required.
