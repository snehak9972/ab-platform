import pytest

from ab_platform import sequential


def test_obrien_fleming_boundary_shrinks_toward_end():
    early = sequential.obrien_fleming_boundary(0.05, 0.25)
    late = sequential.obrien_fleming_boundary(0.05, 1.0)
    assert early > late


def test_boundary_at_full_information_close_to_fixed_sample_z():
    boundary = sequential.obrien_fleming_boundary(0.05, 1.0)
    assert boundary == pytest.approx(1.96, abs=0.01)


def test_spending_schedule_length_matches_looks():
    schedule = sequential.spending_schedule(0.05, 4)
    assert len(schedule) == 4
    assert schedule[-1]["information_fraction"] == 1.0


def test_evaluate_interim_look_structure():
    result = sequential.evaluate_interim_look(z_stat=2.5, alpha=0.05, information_fraction=1.0)
    assert "crosses_boundary" in result
    assert isinstance(result["crosses_boundary"], bool)


def test_naive_peeking_inflates_false_positive_rate_under_null():
    """The core justification for sequential correction: peeking multiple
    times inflates false positives well above the nominal alpha."""
    fpr_single_look = sequential.naive_peeking_false_positive_rate(
        true_effect_pp=0, n_peeks=1, n_per_arm=2000, baseline_rate=0.34,
        alpha=0.05, n_sims=500, seed=1,
    )
    fpr_many_peeks = sequential.naive_peeking_false_positive_rate(
        true_effect_pp=0, n_peeks=10, n_per_arm=2000, baseline_rate=0.34,
        alpha=0.05, n_sims=500, seed=1,
    )
    # Single look should be near nominal alpha; many peeks should be higher.
    assert fpr_single_look < 0.10
    assert fpr_many_peeks > fpr_single_look
