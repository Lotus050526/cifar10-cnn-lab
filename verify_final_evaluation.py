"""Audit fixed selected checkpoints before test access, then verify saved test outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def precheck():
    split_path = Path('data/split.npz')
    split_hash = digest(split_path)
    configs = []
    result = dict(split_sha256=split_hash, official_test_opened=False, groups={})
    for group, expected_epoch, expected_accuracy in [('A', 20, .7576), ('B', 19, .7018)]:
        run = Path('runs')/f'{group}_seed42'
        cfg = read(run/'config.json'); configs.append(cfg)
        check(cfg['group'] == group, f'{group}: wrong group')
        check(cfg['seed'] == 42 and cfg['epochs'] == 20, f'{group}: wrong seed/epochs')
        check(cfg['selection'] == 'highest validation accuracy; ties earliest epoch', f'{group}: selection changed')
        check(cfg['split_sha256'] == split_hash, f'{group}: split hash mismatch')
        with np.load(split_path) as split:
            check(np.array_equal(split['mean'], cfg['normalization_mean']), f'{group}: normalization mean mismatch')
            check(np.array_equal(split['std'], cfg['normalization_std']), f'{group}: normalization std mismatch')
        rows = read(run/'metrics.json'); completed = read(run/'completed.json')
        check(len(rows) == completed['epochs'] == 20, f'{group}: incomplete run')
        selected = max(rows, key=lambda r: r['val']['accuracy'])
        ck = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        check(ck['epoch'] == selected['epoch'] == expected_epoch, f'{group}: selected epoch mismatch')
        check(ck['val'] == selected['val'], f'{group}: checkpoint/metrics mismatch')
        check(ck['val']['accuracy'] == completed['best_accuracy'] == expected_accuracy, f'{group}: validation accuracy mismatch')
        hashes = {name: digest(run/name) for name in ['config.json', 'best.pt']}
        previous = read(Path('results')/run.name/'artifact_hashes.json')
        check(all(previous[k] == v for k,v in hashes.items()), f'{group}: training artifact changed since verification')
        if (run/'locked.json').exists():
            check(read(run/'locked.json') == hashes, f'{group}: existing lock mismatch')
        result['groups'][group] = dict(epoch=ck['epoch'], val_accuracy=ck['val']['accuracy'],
            hashes=hashes, existing_lock=(run/'locked.json').exists(), existing_test=(run/'test').exists())
    check({k:v for k,v in configs[0].items() if k!='group'} ==
          {k:v for k,v in configs[1].items() if k!='group'}, 'A/B common configuration mismatch')
    return result


def audit_test(pre):
    result = dict(passed=True, groups={})
    for group in ['A','B']:
        run = Path('runs')/f'{group}_seed42'; dest = run/'test'
        check(read(run/'locked.json') == pre['groups'][group]['hashes'], f'{group}: lock mismatch')
        report = read(dest/'metrics.json')
        with (dest/'predictions.csv').open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        check(len(rows) == 10000, f'{group}: expected exactly 10000 predictions')
        check([int(r['index']) for r in rows] == list(range(10000)), f'{group}: indices missing/duplicated/out of order')
        truth = np.array([int(r['true']) for r in rows]); pred = np.array([int(r['predicted']) for r in rows])
        confidence = np.array([float(r['confidence']) for r in rows])
        check(np.all((truth>=0)&(truth<10)&(pred>=0)&(pred<10)), f'{group}: invalid labels')
        check(np.all(np.isfinite(confidence)&(confidence>=0)&(confidence<=1)), f'{group}: invalid confidence')
        cm = np.loadtxt(dest/'confusion_raw.csv', delimiter=',', dtype=int)
        rebuilt = np.bincount(truth*10+pred, minlength=100).reshape(10,10)
        check(cm.shape == (10,10) and np.array_equal(cm,rebuilt), f'{group}: confusion matrix/predictions differ')
        check(int(cm.sum()) == 10000, f'{group}: confusion matrix sum incorrect')
        check(np.array_equal(cm.sum(1),np.full(10,1000)), f'{group}: per-class counts incorrect')
        accuracy = float(cm.trace()/10000)
        check(accuracy == report['accuracy'] == float((truth==pred).mean()), f'{group}: accuracy mismatch')
        check(report['counts'] == cm.sum(1).tolist(), f'{group}: report counts mismatch')
        norm = cm/cm.sum(1,keepdims=True)
        check(np.allclose(np.loadtxt(dest/'confusion_normalized.csv',delimiter=','), norm), f'{group}: row normalization mismatch')
        check(np.allclose(report['per_class_accuracy'],np.diag(norm)), f'{group}: per-class accuracy mismatch')
        selection = read(dest/'error_selection.json')
        off = cm.copy(); np.fill_diagonal(off,0)
        pairs = sorted([(int(off[a,b]),a,b) for a in range(10) for b in range(10) if a!=b], key=lambda x:(-x[0],x[1],x[2]))[:2]
        expected = []
        for count,a,b in pairs:
            candidates = sorted([r for r in rows if int(r['true'])==a and int(r['predicted'])==b], key=lambda r:(float(r['confidence']),int(r['index'])))
            if candidates:
                expected.extend([int(candidates[0]['index']),int(candidates[-1]['index'])])
        check([s['index'] for s in selection['samples']] == expected, f'{group}: error selection rule mismatch')
        for s in selection['samples']:
            r = rows[s['index']]
            check(s['true']==int(r['true']) and s['predicted']==int(r['predicted']) and s['true']!=s['predicted']
                  and s['confidence']==float(r['confidence']), f'{group}: error example mismatch')
        for name in ['confusion_raw.png','confusion_normalized.png','errors.png']:
            check((dest/name).stat().st_size>0, f'{group}: missing figure {name}')
        result['groups'][group] = dict(images=10000, correct=int(cm.trace()), accuracy=accuracy,
            selected_epoch=pre['groups'][group]['epoch'], val_accuracy=pre['groups'][group]['val_accuracy'],
            classes=report['classes'], counts=report['counts'], per_class_accuracy=report['per_class_accuracy'],
            files={p.name: dict(path=p.as_posix(),sha256=digest(p)) for p in sorted(dest.iterdir()) if p.is_file()})
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['pre','post']); args=parser.parse_args()
    pre=precheck()
    result=pre if args.mode=='pre' else audit_test(pre)
    path=Path('results')/('final_precheck.json' if args.mode=='pre' else 'final_test_audit.json')
    if path.exists():
        print('Existing audit retained:',path)
    else:
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
