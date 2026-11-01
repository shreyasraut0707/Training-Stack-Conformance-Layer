import numpy as np
import pytest
import torch

from conform.oracle import (
    IGNORE_INDEX, accumulated_loss, loss_sum_and_count, mean_of_means, reference_loss,
)

VOCAB = 11


def random_batch(seed, batch=2, seq=6, ignore=()):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, VOCAB, generator=g, dtype=torch.float64)
    labels = torch.randint(0, VOCAB, (batch, seq), generator=g)
    for b, t in ignore:
        labels[b, t] = IGNORE_INDEX
    return logits, labels


def numpy_nll(logits, labels):
    """A second, independent implementation in plain NumPy, for cross-checking."""
    lg = logits[:, :-1].numpy().astype(np.float64)
    tg = labels[:, 1:].numpy()
    total, count = 0.0, 0
    for b in range(tg.shape[0]):
        for t in range(tg.shape[1]):
            if tg[b, t] == IGNORE_INDEX:
                continue
            row = lg[b, t]
            log_z = row.max() + np.log(np.exp(row - row.max()).sum())
            total += log_z - row[tg[b, t]]
            count += 1
    return total, count


def test_figure_3_example():
    batches = [(10 * 2.0, 10), (30 * 1.0, 30)]
    assert accumulated_loss(batches).item() == pytest.approx(1.25)
    assert mean_of_means(batches).item() == pytest.approx(1.50)


def test_equal_token_counts_hide_the_defect():
    batches = [(20 * 2.0, 20), (20 * 1.0, 20)]
    assert accumulated_loss(batches).item() == pytest.approx(mean_of_means(batches).item())


def test_matches_plain_numpy():
    logits, labels = random_batch(0, ignore=[(0, 2), (1, 5)])
    loss_sum, count = loss_sum_and_count(logits, labels)
    ref_sum, ref_count = numpy_nll(logits, labels)
    assert count == ref_count
    assert loss_sum.item() == pytest.approx(ref_sum, rel=1e-12)


def test_ignored_positions_do_not_matter():
    logits, labels = random_batch(1, ignore=[(0, 3)])
    before = reference_loss([(logits, labels)])
    changed = logits.clone()
    changed[0, 2] += 100.0  # position 2 predicts label 3, which is ignored
    assert reference_loss([(changed, labels)]).item() == pytest.approx(before.item(), rel=1e-12)


def test_first_label_is_never_scored():
    logits, labels = random_batch(2)
    other = labels.clone()
    other[:, 0] = (labels[:, 0] + 1) % VOCAB
    assert torch.equal(reference_loss([(logits, labels)]), reference_loss([(logits, other)]))


def test_splitting_into_micro_batches_matches_one_batch():
    # Rows end up with 2, 5, 4 and 5 supervised tokens: deliberately unequal.
    logits, labels = random_batch(3, batch=4, ignore=[(0, 1), (0, 2), (0, 3), (2, 5)])
    whole = reference_loss([(logits, labels)])
    split = reference_loss([
        (logits[:1], labels[:1]), (logits[1:3], labels[1:3]), (logits[3:], labels[3:]),
    ])
    assert split.item() == pytest.approx(whole.item(), rel=1e-12)

    per_row = [loss_sum_and_count(logits[i:i + 1], labels[i:i + 1]) for i in range(4)]
    assert mean_of_means(per_row).item() != pytest.approx(whole.item(), rel=1e-6)


def test_all_ignored_raises():
    logits, labels = random_batch(4)
    labels[:] = IGNORE_INDEX
    with pytest.raises(ValueError):
        reference_loss([(logits, labels)])


def test_float32_path_is_close():
    logits, labels = random_batch(6, seq=8)
    exact = reference_loss([(logits, labels)])
    rough = reference_loss([(logits, labels)], dtype=torch.float32)
    assert rough.item() == pytest.approx(exact.item(), rel=1e-5)


def test_gradients_flow():
    logits, labels = random_batch(5)
    logits.requires_grad_(True)
    reference_loss([(logits, labels)]).backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
