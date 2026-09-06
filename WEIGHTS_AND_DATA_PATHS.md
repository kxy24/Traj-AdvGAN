# Data and weight paths

This repository intentionally does **not** contain TT100K, CCTSDB, detector weights, or trained generator checkpoints.
Do not commit private/local absolute paths to the repository.

## Public dataset YAMLs

- `configs/datasets/tt100k.yaml`
- `configs/datasets/cctsdb.yaml`

They use portable repository-relative defaults such as `data/TT100K` and `data/CCTSDB2021`.
If your datasets are stored elsewhere, either edit the local YAML after cloning or, preferably, pass an override at runtime:

```bash
--data-root /your/local/dataset/root
```

## Victim detector weights

Pass the detector weights explicitly:

```bash
--victim-weights /your/local/path/best.pt
```

For a detector implementation that must be imported from a local source repository, also pass:

```bash
--yolo-repo /your/local/path/to/yolo-repository
```

These local paths are runtime arguments and should not be committed to GitHub.

## Suggested local layout

```text
Traj-AdvGAN/
├── data/                  # ignored by git
│   ├── TT100K/
│   └── CCTSDB2021/
├── weights/               # ignored by git
└── ... source code ...
```
