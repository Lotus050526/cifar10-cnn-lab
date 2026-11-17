"""Verify archived results offline; never open CIFAR data or execute a model."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matrix(path, cast):
    with path.open(newline='', encoding='utf-8') as file:
        return [[cast(value) for value in row] for row in csv.reader(file)]


def main():
    configs = []
    for group, correct, best_epoch, val_accuracy in [('A', 7601, 20, .7576), ('B', 7109, 19, .7018)]:
        folder = ROOT/'results'/f'{group}_seed42'
        report = read(folder/f'{group}_test_metrics.json')
        require(isinstance(report, dict), f'{group}: test metrics must be an object, not epoch logs')
        require(report['group'] == group and report['images'] == 10000, f'{group}: wrong group/count')
        require(len(report['classes']) == len(report['per_class_accuracy']) == len(report['counts']) == 10, f'{group}: expected 10 classes')
        require(report['counts'] == [1000]*10, f'{group}: per-class counts differ')
        raw = matrix(folder/'confusion_raw.csv', int)
        require(len(raw) == 10 and all(len(row) == 10 for row in raw), f'{group}: matrix dimensions')
        require(all(value >= 0 for row in raw for value in row), f'{group}: negative count')
        require(sum(map(sum, raw)) == 10000, f'{group}: matrix total')
        require(sum(raw[i][i] for i in range(10)) == correct == report['correct'], f'{group}: correct count')
        require(report['accuracy'] == correct/10000, f'{group}: overall accuracy')
        norm = matrix(folder/'confusion_normalized.csv', float)
        require(len(norm) == 10 and all(len(row) == 10 for row in norm), f'{group}: normalized dimensions')
        for i, row in enumerate(raw):
            require(sum(row) == report['counts'][i], f'{group}: row count')
            require(math.isclose(row[i]/sum(row),report['per_class_accuracy'][i]), f'{group}: class accuracy')
            require(all(math.isclose(row[j]/sum(row),norm[i][j],abs_tol=1e-12) for j in range(10)), f'{group}: normalization')
        with (folder/'predictions.csv').open(newline='',encoding='utf-8') as file:
            predictions = list(csv.DictReader(file))
        require([int(r['index']) for r in predictions] == list(range(10000)), f'{group}: sample indices')
        reconstructed = [[0]*10 for _ in range(10)]
        for row in predictions:
            a,b = int(row['true']),int(row['predicted'])
            require(0 <= a < 10 and 0 <= b < 10, f'{group}: invalid label')
            require(0 <= float(row['confidence']) <= 1, f'{group}: invalid confidence')
            reconstructed[a][b] += 1
        require(reconstructed == raw, f'{group}: predictions/matrix mismatch')
        training = read(folder/f'{group}_train_metrics.json')
        require(isinstance(training,list) and [r['epoch'] for r in training] == list(range(1,21)), f'{group}: training log')
        best = max(training,key=lambda r:r['val']['accuracy'])
        require(best['epoch'] == best_epoch == report['selected_epoch'], f'{group}: selected epoch')
        require(best['val']['accuracy'] == val_accuracy == report['selected_validation_accuracy'], f'{group}: validation accuracy')
        cfg = read(folder/'config.json'); configs.append(cfg)
        require(cfg['group'] == group and cfg['epochs'] == 20 and cfg['seed'] == 42, f'{group}: config')
        original_hashes = read(folder/'artifact_hashes.json')
        require(hashlib.sha256((folder/f'{group}_train_metrics.json').read_bytes()).hexdigest() == original_hashes['metrics.json'], f'{group}: training bytes changed')
        require(read(folder/'locked.json')['best.pt'] == report['checkpoint_sha256'], f'{group}: selected weight hash')
        print(f'{group}: PASS; {correct}/10000 = {report["accuracy"]:.2%}; 20 training epochs; 10 classes')
    require({k:v for k,v in configs[0].items() if k!='group'} == {k:v for k,v in configs[1].items() if k!='group'}, 'Shared configuration differs')
    checksums = read(ROOT/'results'/'checksums.json')
    for relative, expected in checksums.items():
        path = ROOT/relative
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Archived bytes changed: {relative}')
    print(f'PASS: {len(checksums)} archived files match SHA256. No data loading, training or test inference performed.')


if __name__ == '__main__':
    main()
