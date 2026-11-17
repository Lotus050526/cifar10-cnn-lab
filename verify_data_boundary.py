"""Regression guard: development must never open the official test batch."""
import builtins
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import experiment as e

original_open = builtins.open


class WithOriginalIndex(e.torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, _ = self.subset[index]
        return image, self.subset.indices[index]


def make_indexed(split, augmented, generator):
    original = e.loader(split, 'train', aug=augmented, limit=384).dataset
    return e.DataLoader(WithOriginalIndex(original), batch_size=128, shuffle=True,
                        num_workers=0, generator=generator)


def consume(data):
    return [index for _, indices in data for index in indices.tolist()]


def check_sampler(split):
    orders = {}
    for group in ['A', 'B']:
        e.seed_all(42)
        generator = e.torch.Generator().manual_seed(42)
        data = make_indexed(split, group == 'B', generator)
        first = consume(data); second = consume(data)
        checkpoint = io.BytesIO()
        e.torch.save({'sampler_rng': generator.get_state(), 'torch_rng': e.torch.get_rng_state()}, checkpoint)
        third = consume(data)
        # A new loader plus unrelated global RNG draws must not change restored order.
        e.torch.rand(137)
        restored = e.torch.Generator().manual_seed(999)
        resumed = make_indexed(split, group == 'B', restored)
        checkpoint.seek(0); state = e.torch.load(checkpoint, weights_only=True)
        restored.set_state(state['sampler_rng']); e.torch.set_rng_state(state['torch_rng'])
        assert consume(resumed) == third, f'{group}: resumed epoch order differs'
        assert first != second and second != third
        assert all(len(set(order)) == 384 for order in [first, second, third])
        orders[group] = [first, second, third]
    assert orders['A'] == orders['B'], 'A/B sample orders differ across epochs'
    return [hashlib.sha256(json.dumps(order).encode()).hexdigest() for order in orders['A']]


def check_test_hash_gate():
    good = hashlib.sha256(Path('data/split.npz').read_bytes()).hexdigest()
    e.verify_split_hash([{'split_sha256': good}])
    with tempfile.TemporaryDirectory(dir='runs', prefix='boundary_check_') as tmp:
        paths = []
        for group in ['A', 'B']:
            path = Path(tmp)/group; path.mkdir(); paths.append(str(path))
            e.save(path/'config.json', dict(group=group, split_sha256='incorrect'))
        with patch.object(e.datasets, 'CIFAR10', side_effect=AssertionError('Test dataset opened')) as test_ctor:
            try:
                e.final_test(paths)
            except ValueError as exc:
                assert 'split.npz' in str(exc)
            else:
                raise AssertionError('Modified split was accepted')
            test_ctor.assert_not_called()


def guarded_open(file, *args, **kwargs):
    if str(file).replace('\\', '/').endswith('/test_batch'):
        raise AssertionError('Official test batch access during development')
    return original_open(file, *args, **kwargs)


if __name__ == '__main__':
    e.torch.set_num_threads(4)
    with patch('builtins.open', guarded_open):
        split = e.prepare()
        train = e.loader(split, 'train', aug=True, limit=128)
        val = e.loader(split, 'val', limit=128)
        assert len(next(iter(train))[1]) == 128
        x1, y1 = next(iter(val)); x2, y2 = next(iter(val))
        e.torch.testing.assert_close(x1, x2, rtol=0, atol=0)
        assert e.torch.equal(y1, y2)
        digests = check_sampler(split)
        check_test_hash_gate()
    result = dict(passed=True, samples_per_epoch=384, checked_epochs=3, sampler_seed=42,
                  identical_AB_order=True, resumed_next_epoch_identical=True,
                  epoch_order_sha256=digests, modified_split_rejected_before_test_open=True,
                  official_test_batch_opened=False)
    Path('results').mkdir(exist_ok=True)
    e.save('results/pretrain_checks.json', result)
    print(json.dumps(result, indent=2))
