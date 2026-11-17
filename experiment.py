"""CPU CIFAR-10 experiment. Test data is opened only by the locked test command."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from torchvision.datasets.utils import check_integrity


class TrainCIFAR10(datasets.CIFAR10):
    """Do not even checksum the official test batch during development."""
    def __init__(self, root, **kwargs):
        assert kwargs.get('train', True)
        super().__init__(root, **kwargs)

    def _check_integrity(self):
        return all(check_integrity(os.path.join(self.root, self.base_folder, name), md5)
                   for name, md5 in self.train_list)


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in [(3, 32), (32, 64), (64, 128)]:
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, device='cpu', cuda_available=torch.cuda.is_available(),
                threads=torch.get_num_threads(), logical_cpus=os.cpu_count(),
                cpu=platform.processor(), disk_free_bytes=shutil.disk_usage('.').free,
                parameters=sum(p.numel() for p in CNN().parameters() if p.requires_grad))


def prepare(download=False):
    raw = TrainCIFAR10('data', train=True, download=download)
    path = Path('data/split.npz')
    if not path.exists():
        rng = np.random.default_rng(2026)
        tr, va = [], []
        labels = np.array(raw.targets)
        for c in range(10):
            ix = rng.permutation(np.flatnonzero(labels == c))
            va.extend(ix[:500]); tr.extend(ix[500:])
        tr, va = np.array(tr), np.array(va)
        total = np.zeros(3); squares = np.zeros(3); count = 0
        for chunk in np.array_split(tr, 90):
            pixels = raw.data[chunk].astype(np.float64) / 255
            total += pixels.sum((0,1,2)); squares += (pixels*pixels).sum((0,1,2)); count += len(chunk)*1024
        mean = total/count; std = np.sqrt(squares/count-mean*mean)
        np.savez(path, train=tr, val=va, mean=mean, std=std, split_seed=2026)
    with np.load(path) as z:
        split = {k: z[k].copy() for k in z.files}
    assert len(split['train']) == 45000 and len(split['val']) == 5000
    assert not set(split['train']) & set(split['val'])
    assert set(np.concatenate([split['train'], split['val']])) == set(range(50000))
    for key, count in [('train', 4500), ('val', 500)]:
        assert np.bincount(np.array(raw.targets)[split[key]], minlength=10).tolist() == [count] * 10
    return split


def transform(split, aug=False):
    return T.Compose(([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if aug else []) +
                     [T.ToTensor(), T.Normalize(split['mean'].tolist(), split['std'].tolist())])


def loader(split, key, aug=False, limit=None, batch=128, shuffle=False, generator=None):
    ds = TrainCIFAR10('data', train=True, transform=transform(split, aug))
    ix = split[key]
    # Shuffle the balanced class-blocked saved indices before taking a smoke subset.
    ix = np.random.default_rng(123).permutation(ix)
    if limit:
        ix = ix[:limit]
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    return DataLoader(Subset(ds, ix.tolist()), batch_size=batch, shuffle=shuffle,
                      num_workers=0, generator=generator)


def epoch(model, data, optimizer=None):
    model.train(optimizer is not None)
    start = time.perf_counter(); loss_sum = correct = n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in data:
            logits = model(x); loss = nn.functional.cross_entropy(logits, y)
            if optimizer:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            assert torch.isfinite(loss)
            n += len(y); loss_sum += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item()
    return dict(loss=loss_sum/n, accuracy=correct/n, images=n, seconds=time.perf_counter()-start)


def config(args, split):
    return dict(group=args.group, seed=args.seed, sampler_seed=args.seed,
                sampler_rng='independent torch.Generator; checkpointed', config_version=2,
                epochs=args.epochs, batch=args.batch,
                threads=args.threads, optimizer='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005,
                scheduler='CosineAnnealingLR', selection='highest validation accuracy; ties earliest epoch',
                split_sha256=hashlib.sha256(Path('data/split.npz').read_bytes()).hexdigest(),
                normalization_mean=split['mean'].tolist(), normalization_std=split['std'].tolist(),
                deterministic_train_eval_images=1000, model='CNN 32-64-128', device='cpu')


def run(args):
    split = prepare(args.download)
    seed_all(args.seed)
    if args.mode == 'smoke':
        out = Path('runs') / ('smoke_' + time.strftime('%Y%m%d_%H%M%S'))
        out.mkdir(parents=True, exist_ok=False)
        save(out/'environment.json', environment())
        save(out/'config.json', dict(seed=args.seed, batch=args.batch, threads=args.threads,
                                    benchmark_images=args.benchmark_images, projected_epochs=args.epochs,
                                    split_sha256=hashlib.sha256(Path('data/split.npz').read_bytes()).hexdigest(),
                                    mean=split['mean'].tolist(), std=split['std'].tolist(),
                                    benchmark_optimizer='SGD lr=0.05 momentum=0.9 weight_decay=0.0005'))
        results = {}
        for group in ['A', 'B']:
            seed_all(args.seed); model = CNN(); opt = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9, weight_decay=.0005)
            generator = torch.Generator().manual_seed(args.seed)
            train = loader(split, 'train', group == 'B', 256, args.batch, True, generator)
            val = loader(split, 'val', limit=256, batch=args.batch)
            warm = epoch(model, train, opt)
            checkpoint = out/f'{group}.pt'
            torch.save(model.state_dict(), checkpoint)
            model.eval(); x, _ = next(iter(val))
            with torch.no_grad(): reference = model(x)
            restored = CNN(); restored.load_state_dict(torch.load(checkpoint, weights_only=True)); restored.eval()
            with torch.no_grad(): torch.testing.assert_close(reference, restored(x), rtol=0, atol=0)
            validation = epoch(restored, val)
            timed = epoch(model, loader(split, 'train', group == 'B', args.benchmark_images, args.batch, True, generator), opt)
            vtimed = epoch(model, loader(split, 'val', limit=1024, batch=args.batch))
            seconds = timed['seconds']/timed['images']*45000 + vtimed['seconds']/vtimed['images']*6000
            results[group] = dict(warmup=warm, validation=validation, checkpoint_reload_exact=True,
                                  benchmark_train=timed, benchmark_validation=vtimed,
                                  estimated_epoch_seconds_including_1000_train_eval=seconds,
                                  estimated_formal_seconds=seconds*args.epochs)
            save(out/'results.json', results)
            print(group, json.dumps(results[group]), flush=True)
        print('Artifacts:', out, flush=True)
        return
    if args.resume:
        out = Path(args.resume); cfg = json.loads((out/'config.json').read_text())
        assert cfg == config(args, split), 'Resume arguments must match saved configuration'
    else:
        out = Path('runs') / (args.name or f'{args.group}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}')
        out.mkdir(parents=True, exist_ok=False); cfg = config(args, split)
        save(out/'config.json', cfg); save(out/'environment.json', environment())
    if (out/'locked.json').exists(): raise ValueError('Run locked for final test')
    model = CNN(); opt = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    rows = []; best = -1; start = 0
    if args.resume:
        ck = torch.load(out/'last.pt', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        sampler_generator.set_state(ck['sampler_rng'])
        rows = ck['metrics']; best = ck['best']; start = ck['epoch'] + 1
    train = loader(split, 'train', args.group == 'B', batch=args.batch, shuffle=True,
                   generator=sampler_generator)
    val = loader(split, 'val', batch=args.batch)
    fixed = loader(split, 'train', limit=1000, batch=args.batch)
    for e in range(start, args.epochs):
        begin = time.perf_counter(); lr = opt.param_groups[0]['lr']
        tr = epoch(model, train, opt); va = epoch(model, val); det = epoch(model, fixed)
        scheduler.step()
        row = dict(epoch=e+1, lr=lr, train=tr, val=va, deterministic_train=det, seconds=time.perf_counter()-begin)
        rows.append(row)
        if va['accuracy'] > best:
            best = va['accuracy']; torch.save(dict(model=model.state_dict(), epoch=e+1, val=va), out/'best.pt')
        ck = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                  torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                  sampler_rng=sampler_generator.get_state(),
                  epoch=e, metrics=rows, best=best)
        torch.save(ck, out/'last.tmp'); os.replace(out/'last.tmp', out/'last.pt')
        save(out/'metrics.json', rows); print(json.dumps(row), flush=True)
    save(out/'completed.json', dict(epochs=len(rows), best_accuracy=best))


def plot(paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary = []
    for path in paths:
        p = Path(path); rows = json.loads((p/'metrics.json').read_text()); cfg = json.loads((p/'config.json').read_text())
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            for key,style in [('val','-'),('train','--'),('deterministic_train',':')]:
                ax.plot([r['epoch'] for r in rows], [r[key][metric] for r in rows], style, label=cfg['group']+' '+key)
            ax.set(xlabel='Epoch', ylabel=metric); ax.legend(fontsize=7)
        best = max(rows, key=lambda r:r['val']['accuracy'])
        summary.append(dict(run=p.name, best_epoch=best['epoch'], val_accuracy=best['val']['accuracy'], seconds=sum(r['seconds'] for r in rows)))
    dest = Path('results'); dest.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(dest/'validation_curves.png'); save(dest/'summary.json', summary)


def final_test(paths, lock=False):
    # Lock both completed configurations together before opening the official test dataset.
    runs = [Path(p) for p in paths]
    assert len(runs) == 2
    cfgs = [json.loads((p/'config.json').read_text()) for p in runs]
    assert {c['group'] for c in cfgs} == {'A', 'B'}
    assert {k:v for k,v in cfgs[0].items() if k!='group'} == {k:v for k,v in cfgs[1].items() if k!='group'}
    verify_split_hash(cfgs)
    for p in runs:
        assert (p/'completed.json').exists()
        hashes = {name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['config.json','best.pt']}
        if lock:
            if (p/'locked.json').exists():
                assert json.loads((p/'locked.json').read_text()) == hashes, 'Existing lock differs'
            else:
                save(p/'locked.json', hashes)
        else: assert json.loads((p/'locked.json').read_text()) == hashes
    if lock: return
    os.environ.setdefault('MPLCONFIGDIR', str(Path('data/.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    split = prepare()
    verify_split_hash(cfgs)
    ds = datasets.CIFAR10('data', train=False, transform=transform(split), download=False)
    for p, cfg in zip(runs, cfgs):
        if (p/'test').exists(): raise ValueError('Test output already exists')
        dest = p/'test'; dest.mkdir()
        checkpoint = torch.load(p/'best.pt', weights_only=True)
        model = CNN(); model.load_state_dict(checkpoint['model']); model.eval()
        records = []; cm = np.zeros((10,10), dtype=int)
        with torch.no_grad():
            for x,y in DataLoader(ds,batch_size=128):
                prob = model(x).softmax(1); conf,pred = prob.max(1)
                for truth, guess, confidence in zip(y.tolist(),pred.tolist(),conf.tolist()):
                    records.append([len(records),truth,guess,confidence]); cm[truth,guess] += 1
        with (dest/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['index','true','predicted','confidence']); w.writerows(records)
        norm=cm/cm.sum(1,keepdims=True)
        assert len(ds) == len(records) == int(cm.sum()) == 10000
        save(dest/'metrics.json',dict(group=cfg['group'], images=len(records), correct=int(cm.trace()),
             selected_epoch=checkpoint['epoch'], selected_validation_accuracy=checkpoint['val']['accuracy'],
             checkpoint_sha256=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest(), split_sha256=cfg['split_sha256'],
             accuracy=float(cm.trace()/cm.sum()),classes=ds.classes,counts=cm.sum(1).tolist(),per_class_accuracy=np.diag(norm).tolist()))
        np.savetxt(dest/'confusion_raw.csv',cm,delimiter=',',fmt='%d'); np.savetxt(dest/'confusion_normalized.csv',norm,delimiter=',')
        for name,matrix in [('raw',cm),('normalized',norm)]:
            fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(matrix); fig.colorbar(im,ax=ax)
            ax.set(xticks=range(10),yticks=range(10),xticklabels=ds.classes,yticklabels=ds.classes,xlabel='Predicted',ylabel='True',title=f"Group {cfg['group']} - test confusion ({name}), n=10000"); plt.setp(ax.get_xticklabels(),rotation=90)
            fig.tight_layout(); fig.savefig(dest/f'confusion_{name}.png'); plt.close(fig)
        off=cm.copy(); np.fill_diagonal(off,0); pairs=sorted([(int(off[a,b]),a,b) for a in range(10) for b in range(10) if a!=b],key=lambda z:(-z[0],z[1],z[2]))[:2]
        selected=[]; fig,axes=plt.subplots(2,2,figsize=(8,8))
        fig.suptitle(f"Group {cfg['group']} - test errors (low / high confidence)")
        for row,(count,a,b) in enumerate(pairs):
            examples=sorted([r for r in records if r[1]==a and r[2]==b],key=lambda r:(r[3],r[0]))
            for col,r in enumerate([examples[0],examples[-1]] if examples else []):
                selected.append(dict(index=r[0],true=a,predicted=b,confidence=r[3],pair_count=count,selection='lowest confidence' if col==0 else 'highest confidence'))
                axes[row,col].imshow(ds.data[r[0]]); axes[row,col].set_title(f'{ds.classes[a]} -> {ds.classes[b]}\nindex={r[0]}, p={r[3]:.3f}'); axes[row,col].axis('off')
        fig.tight_layout(); fig.savefig(dest/'errors.png'); plt.close(fig)
        save(dest/'error_selection.json',dict(group=cfg['group'],rule='two highest directed off-diagonal counts; ties label order; confidence extremes, index tie break',samples=selected))
        print(json.dumps(dict(group=cfg['group'], images=len(records), accuracy=float(cm.trace()/cm.sum()), output=dest.as_posix())), flush=True)


def verify_split_hash(configs):
    actual = hashlib.sha256(Path('data/split.npz').read_bytes()).hexdigest()
    if any(cfg['split_sha256'] != actual for cfg in configs):
        raise ValueError('Current data/split.npz differs from saved training split; refusing final evaluation')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','smoke','train','plot','lock','test','env'])
    p.add_argument('--group',choices=['A','B'],default='A'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=128)
    p.add_argument('--threads',type=int,default=4); p.add_argument('--benchmark-images',type=int,default=4096)
    p.add_argument('--download',action='store_true'); p.add_argument('--name'); p.add_argument('--resume'); p.add_argument('--runs',nargs='+')
    args=p.parse_args(); torch.set_num_threads(args.threads)
    if args.mode=='env': print(json.dumps(environment(),indent=2))
    elif args.mode=='prepare': prepare(args.download)
    elif args.mode=='plot': plot(args.runs)
    elif args.mode in ['lock','test']: final_test(args.runs,args.mode=='lock')
    else: run(args)
