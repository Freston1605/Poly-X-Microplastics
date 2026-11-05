"""Tracked manifest of the Poly-X labelled dataset split.

The dataset itself is NOT committed to the repository (see .gitignore: /data/).
This module records which image belongs to which split, so a copy of the
labelled dataset can be reconstructed deterministically without guessing.

The unit of splitting is the physical plate, never an individual photo of one.
All captures of the same plate (the ``(4)``, ``(4.1)``, ``(4.2)`` suffixes of
the Sedcalci series, and the ``(6)``/``(6.1)`` or ``(7)``/``(7.1)`` pairs of
the marine sediment series) are kept together.

Two files are removed by ``make_splits.py``:

- ``LDPE 10``: a byte-identical copy of ``LDPE 8`` with a partial,
  differently-aligned re-annotation (698 boxes against LDPE 8's 1052).
- ``LDPE 12``: a byte-identical copy of ``LDPE 9`` carrying no label at all.

``LDPE 11`` is deliberately excluded from all three splits and lives in
``data/pending/`` because it has not been annotated yet. Re-add it only after
a human labeller creates its ground-truth file.
"""

from __future__ import annotations

PLATE_GROUPS: dict[str, list[str]] = {
    "sedcalci_4": [
        "04-05-23 Sedcalci (4),PET(F10),PE (9z), PP(1z),PS(16z) 100mm",
        "04-05-23 Sedcalci (4.1),PET(F10),PE (9z), PP(1z),PS(16z) 100mm",
        "04-05-23 Sedcalci (4.2),PET(F10),PE (9z), PP(1z),PS(16z) 100mm",
    ],
    "sedcalci_5": [
        "04-05-23 Sedcalci (5),PET(F11),PE (10z), PP(2z),PS(17z) 100mm",
        "04-05-23 Sedcalci (5.1),PET(F11),PE (10z), PP(2z),PS(17z) 100mm",
        "04-05-23 Sedcalci (5.2),PET(F11),PE (10z), PP(2z),PS(17z) 100mm",
    ],
    "sedcalci_6": [
        "04-05-23 Sedcalci (6),PET(F12),PE (11z), PP(3z),PS(18z) 100mm",
        "04-05-23 Sedcalci (6.1),PET(F12),PE (11z), PP(3z),PS(18z) 100mm",
        "04-05-23 Sedcalci (6.2),PET(F12),PE (11z), PP(3z),PS(18z) 100mm",
    ],
    "sed_marino_6": [
        "30-03-23 sed marino (6) + PET (F1) 100 mm",
        "30-03-23 sed marino (6.1) + PET (F1) 100 mm",
    ],
    "sed_marino_7": [
        "30-03-23 sed marino (7) + PET (F2) 100 mm",
        "30-03-23 sed marino (7.1) + PET (F2) 100 mm",
    ],
    "mezc_1": ["Mezc 1"],
    "mezc_2": ["Mezc 2"],
    "mezc_3": ["Mezc 3"],
    "mezc_4": ["Mezc 4"],
    "mezc_5": ["Mezc 5"],
    "pet_1": ["PET 1_limpia"],
    "pet_2": ["PET 2_limpia"],
    "pet_3": ["PET 3_limpia"],
    "pet_4": ["PET 4_limpia"],
    "pet_5": ["PET 5 60mm_limpia"],
    "pet_6": ["PET 6_limpia"],
    "pet_7": ["PET 7 60mm_limpia"],
    "pet_8": ["PET 8 60mm_limpia"],
    "pet_9": ["PET 9 60mm_limpia"],
    "pet_10": ["PET 10 60mm_limpia"],
    "pet_11": ["PET 11 60mm_limpia"],
    "pet_12": ["PET 12 60mm_limpia"],
    "pet_13": ["PET 13 60mm_limpia"],
    "pet_14": ["PET 14 60mm_limpia"],
    "pet_15": ["PET 15 60mm_limpia"],
    "pet_16": ["PET 16 60mm_limpia"],
    "pet_17": ["PET 17 60mm_limpia"],
    "pet_18": ["PET 18_limpia"],
    "pet_19": ["PET 19_limpia"],
    "pet_20": ["PET 20_limpia"],
    "pet_21": ["PET 21_limpia"],
    "pp_1": ["PP 1"],
    "pp_2": ["PP 2"],
    "pp_3": ["PP 3 60mm"],
    "pp_4": ["PP 4 60mm"],
    "pp_5": ["PP 5"],
    "pp_6": ["PP 6"],
    "pp_7": ["PP 7"],
    "ldpe_1": ["LDPE 1"],
    "ldpe_2": ["LDPE 2"],
    "ldpe_3": ["LDPE 3"],
    "ldpe_4": ["LDPE 4"],
    "ldpe_5": ["LDPE 5"],
    "ldpe_6": ["LDPE 6"],
    "ldpe_7": ["LDPE 7"],
    "ldpe_8": ["LDPE 8"],
    "ldpe_9": ["LDPE 9"],
    "ldpe_11": ["LDPE 11"],
    "ldpe_13": ["LDPE 13"],
}

SPLIT_GROUPS: dict[str, list[str]] = {
    "train": [
        "sedcalci_4",
        "sed_marino_7",
        "mezc_1",
        "pet_1",
        "pet_3",
        "pet_4",
        "pet_5",
        "pet_6",
        "pet_7",
        "pet_8",
        "pet_9",
        "pet_11",
        "pet_12",
        "pet_13",
        "pet_14",
        "pet_15",
        "pet_16",
        "pet_18",
        "pet_19",
        "pet_20",
        "pet_21",
        "pp_2",
        "pp_3",
        "pp_5",
        "pp_6",
        "pp_7",
        "ldpe_2",
        "ldpe_4",
        "ldpe_7",
        "ldpe_8",
        "ldpe_9",
        "ldpe_13",
    ],
    "val": [
        "sedcalci_6",
        "sed_marino_6",
        "mezc_2",
        "mezc_4",
        "mezc_5",
        "pet_10",
        "pp_1",
        "ldpe_5",
    ],
    "test": [
        "sedcalci_5",
        "mezc_3",
        "pet_2",
        "pet_17",
        "pp_4",
        "ldpe_1",
        "ldpe_3",
        "ldpe_6",
    ],
}

PENDING_ANNOTATION: tuple[str, ...] = ("ldpe_11",)

SPLIT_IMAGES: dict[str, list[str]] = {
    split: [image for group in groups for image in PLATE_GROUPS[group]]
    for split, groups in SPLIT_GROUPS.items()
}

REMOVED_DUPLICATES: tuple[str, ...] = ("LDPE 10", "LDPE 12")

EXPECTED_COUNTS: dict[str, int] = {split: len(images) for split, images in SPLIT_IMAGES.items()}
EXPECTED_TOTAL_IMAGES: int = sum(EXPECTED_COUNTS.values())
