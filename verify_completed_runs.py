"""Check completed A/B artifacts and export portable evidence without weights."""
import hashlib
import json
import shutil
from pathlib import Path

import torch

from experiment import CNN, save, verify_split_hash


def main():
    paths = [Path('runs/A_seed42'), Path('runs/B_seed42')]
    configs = [json.loads((p/'config.json').read_text()) for p in paths]
    assert {c['group'] for c in configs} == {'A', 'B'}
    assert {k:v for k,v in configs[0].items() if k != 'group'} == {
        k:v for k,v in configs[1].items() if k != 'group'}
    verify_split_hash(configs)
    states = []
    summary = []
    for path, cfg in zip(paths, configs):
        rows = json.loads((path/'metrics.json').read_text())
        completed = json.loads((path/'completed.json').read_text())
        assert len(rows) == cfg['epochs'] == completed['epochs'] == 20
        assert [row['epoch'] for row in rows] == list(range(1, 21))
        assert all(row['train']['images'] == 45000 and row['val']['images'] == 5000
                   and row['deterministic_train']['images'] == 1000 for row in rows)
        best_row = max(rows, key=lambda r: r['val']['accuracy'])
        best = torch.load(path/'best.pt', weights_only=True)
        last = torch.load(path/'last.pt', weights_only=False)
        assert best['epoch'] == best_row['epoch']
        assert best['val'] == best_row['val']
        assert best['val']['accuracy'] == completed['best_accuracy'] == last['best']
        assert last['metrics'] == rows and last['epoch'] == 19
        assert last['scheduler']['last_epoch'] == 20
        model = CNN(); model.load_state_dict(best['model'], strict=True)
        model.load_state_dict(last['model'], strict=True)
        states.append(last['sampler_rng'])
        assert not (path/'locked.json').exists() and not (path/'test').exists()
        dest = Path('results')/path.name; dest.mkdir(exist_ok=True)
        for filename in ['config.json', 'metrics.json', 'environment.json', 'completed.json']:
            shutil.copy2(path/filename, dest/filename)
        hashes = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                  for name in ['config.json', 'metrics.json', 'best.pt', 'last.pt']}
        save(dest/'artifact_hashes.json', hashes)
        summary.append(dict(group=cfg['group'], run=path.name, epochs=20,
                            best_epoch=best_row['epoch'], val_accuracy=best_row['val']['accuracy'],
                            val_loss=best_row['val']['loss'],
                            deterministic_train_accuracy_at_best=best_row['deterministic_train']['accuracy'],
                            seconds=sum(row['seconds'] for row in rows),
                            train_images_processed=900000, val_images_processed=100000,
                            deterministic_train_eval_images_processed=20000))
    assert torch.equal(states[0], states[1]), 'Final A/B sampler RNG states differ'
    save('results/artifact_checks.json', dict(passed=True, completed_epochs=40,
         shared_config_and_split=True, final_sampler_states_identical=True,
         best_checkpoints_match_metrics=True, checkpoint_state_dicts_load=True,
         lock_or_test_executed=False))
    save('results/summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
