"""The report's particle sample must show what needs to be checked.

The cards are the only place in the report where one can SEE whether a
measurement was made correctly. Which particles enter them is not a
presentation detail: it decides what can be audited and what cannot.

Before, the first particles by file order were used, which meant the largest
particle could be missing and fibres never appeared -- they are ~1% of the
material. These tests pin that split so editing the report cannot lose it.
"""

from __future__ import annotations

from polyx.gui.report.samples import choose_records


class _Particle:
    """The minimum `choose_records` reads from a Detection."""

    def __init__(self, length_um, morphotype):
        self.length_um = length_um
        self.morphotype = morphotype

    def __repr__(self):  # so a failure reads clearly
        return f"<{self.morphotype} {self.length_um}>"


def _lot(n_fibres: int, n_others: int) -> list:
    """Fibres of 100, 200, ... and particles of 1000, 2000, ... (distinct
    and separated sizes so the order by size is unambiguous when checking)."""
    shapes = [(_Particle((i + 1) * 100.0, "fibre"), f"f{i}.jpg") for i in range(n_fibres)]
    shapes += [(_Particle((i + 1) * 1000.0, "fragment"), f"p{i}.jpg") for i in range(n_others)]
    return shapes


def _count(chosen):
    fib = sum(1 for d, _ in chosen if d.morphotype == "fibre")
    return fib, len(chosen) - fib


# ── The requested split ──
def test_with_plenty_of_material_it_splits_evenly():
    fib, others = _count(choose_records(_lot(20, 20), 12))
    assert (fib, others) == (6, 6)


def test_the_largest_of_each_morphotype_enters():
    """That is the reason for the split: any claim about maximum size rests on
    the largest of each type."""
    chosen = choose_records(_lot(20, 20), 12)
    largest_fibre = max(d.length_um for d, _ in chosen if d.morphotype == "fibre")
    largest_other = max(d.length_um for d, _ in chosen if d.morphotype != "fibre")
    assert largest_fibre == 20 * 100.0
    assert largest_other == 20 * 1000.0


def test_within_each_group_size_rules():
    chosen = choose_records(_lot(20, 20), 12)
    fibres = [d.length_um for d, _ in chosen if d.morphotype == "fibre"]
    assert fibres == sorted(fibres, reverse=True)
    assert min(fibres) == 15 * 100.0, "must be the 6 largest, not arbitrary ones"


# ── Real batches: almost never enough fibres ──
def test_without_fibres_particles_fill_the_gap():
    """The normal case for this material. It must not return a short sample."""
    fib, others = _count(choose_records(_lot(0, 30), 12))
    assert (fib, others) == (0, 12)


def test_with_few_fibres_all_enter_and_the_rest_is_filled():
    """Two fibres in the batch: both enter, the other 10 are particles.
    If this breaks, fibres disappear from the report again."""
    fib, others = _count(choose_records(_lot(2, 30), 12))
    assert (fib, others) == (2, 10)


def test_with_few_particles_fibres_fill_the_gap():
    fib, others = _count(choose_records(_lot(30, 3), 12))
    assert (fib, others) == (9, 3)


# ── Edges ──
def test_a_batch_smaller_than_the_quota_enters_whole():
    chosen = choose_records(_lot(2, 3), 12)
    assert len(chosen) == 5


def test_an_empty_batch_or_zero_quota_does_not_crash():
    assert choose_records([], 12) == []
    assert choose_records(_lot(5, 5), 0) == []


def test_it_never_returns_more_than_requested():
    for quota in (1, 2, 5, 12, 40):
        assert len(choose_records(_lot(50, 50), quota)) <= quota


def test_it_does_not_repeat_the_same_particle():
    """Filling the gap could let the same particle through twice if the
    comparison is wrong."""
    chosen = choose_records(_lot(4, 4), 12)
    ids = [id(d) for d, _ in chosen]
    assert len(ids) == len(set(ids))


def test_an_unmeasured_size_does_not_crash():
    """Without calibration, length_um stays None. The card is still useful
    -- the mask looks the same -- so it must not be discarded or crash."""
    shapes = [(_Particle(None, "fragment"), "a.jpg"), (_Particle(None, "fibre"), "b.jpg")]
    assert len(choose_records(shapes, 12)) == 2
