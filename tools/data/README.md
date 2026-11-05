# Dataset split tool

The labelled YOLO dataset for this project is **not** versioned in this
repository. It lives under the gitignored `data/` directory and is shared
through the project owner’s storage (institutional/author download).

`make_splits.py` turns a local copy of the dataset into a deterministic
`train` / `val` / `test` split. It:

1. verifies every image has a corresponding label (and vice-versa),
2. mechanically moves files instead of copying them,
3. drops the two byte-identical duplicates (`LDPE 10`, `LDPE 12`),
4. rewrites `data/dataset.yaml` with `test: images/test`,
5. refuses to do anything if anything is missing or unexpected.

## Usage

```bash
.venv/bin/python tools/data/make_splits.py --root data
.venv/bin/python tools/data/make_splits.py --root data --apply
```

Without `--apply` the command is a dry run and changes nothing.

## Split design

Splits are **group-stratified by plate**, not randomly shuffled by image.
Images that came from the same sampling event or re-export are always kept
together:

- the three captures of `04-05-23 Sedcalci (4)` / `(5)` / `(6)`;
- the paired captures of `30-03-23 sed marino (6)` and `(7)`;
- all other images listed in `split_manifest.py`.

This is required because the same physical plate appears as multiple images.
A random split would leak information between training and validation and
would inflate the reported metrics.

The assignment was also optimised so the three splits contain roughly
proportional numbers of particles of each class while preserving plate groups.
It contains all three acquisition domains (spiked sediment, mixtures and pure
plates) in both val and test, so the metrics reflect the full distribution
rather than a single acquisition type.

`LDPE 11` is deliberately **not** in any split: it is kept in
`data/pending/` until a human annotates it.

## After the split

- `data/images/train`, `data/images/val`, `data/images/test`
- `data/labels/train`, `data/labels/val`, `data/labels/test`
- `data/dataset.yaml` — unchanged class names, plus the `test` split

Run `python -m pytest -q` across the repository after re-applying the split.
