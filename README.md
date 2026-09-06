Traj-AdvGAN

Research-code release for Dual-Loop Trajectory Guidance with Channel Attention for Efficient Adversarial Attacks on YOLO-Based Traffic Sign Detection.

Traj-AdvGAN combines an AdvGAN-style generator with an AdaAD-inspired PGD inner-loop reference search. The generator uses residual bottleneck features followed by SE-style channel attention, and the final trained generator produces perturbations with one forward pass.

Core method settings

input size: 640 x 640

epsilon = 8/255

inner-loop steps: K = 10

step size: eta = 2/255

lambda_adv = 1

lambda_det = 10

lambda_traj = 10

lambda_smooth = 0.05

clean-box confidence threshold: 0.25

disappearance IoU threshold: 0.5

The channel-attention implementation follows GAP -> FC -> ReLU -> FC -> Sigmoid -> channel recalibration. The manuscript does not specify the reduction ratio; this release uses 4 by default.

Repository structure

Traj-AdvGAN/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE_NOTE.md
├── WEIGHTS_AND_DATA_PATHS.md
│
├── train.py
├── eval.py
│
├── attacks/
│   └── traj_advgan.py
├── models/
│   ├── generator.py
│   └── discriminator.py
├── datasets/
│   ├── common.py
│   ├── tt100k.py
│   └── cctsdb.py
├── utils/
│   ├── config.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── yolo_wrapper.py
│
├── configs/
│   ├── attack.yaml
│   └── datasets/
│       ├── tt100k.yaml
│       └── cctsdb.yaml
│
├── tools/
│   ├── generate_adv_dataset.py
│   ├── visualize_perturbation.py
│   ├── perturbation_enrichment.py
│   └── compute_generator_latency.py
├── scripts/
│   ├── train_tt100k.sh
│   ├── train_cctsdb.sh
│   └── ablation_tt100k_yolov10.sh
└── checkpoints/
    └── README.md

Installation

conda create -n trajadvgan python=3.10 -y
conda activate trajadvgan
pip install -r requirements.txt

Data

The datasets are not included in this repository. Public, portable dataset definitions are in:

configs/datasets/tt100k.yaml
configs/datasets/cctsdb.yaml

The default roots are data/TT100K and data/CCTSDB2021. To keep your machine-specific path out of GitHub, use --data-root at runtime.

See WEIGHTS_AND_DATA_PATHS.md for details.

Training

TT100K example:

python train.py \
  --data-config configs/datasets/tt100k.yaml \
  --attack-config configs/attack.yaml \
  --data-root /path/to/TT100K \
  --victim-weights /path/to/yolov10_tt100k.pt

CCTSDB example:

python train.py \
  --data-config configs/datasets/cctsdb.yaml \
  --attack-config configs/attack.yaml \
  --data-root /path/to/CCTSDB2021 \
  --victim-weights /path/to/yolov10_cctsdb.pt

If a YOLO implementation must be imported from a local repository, add:

--yolo-repo /path/to/yolo-repository

Clean-box disappearance evaluation

python eval.py \
  --data-config configs/datasets/tt100k.yaml \
  --attack-config configs/attack.yaml \
  --data-root /path/to/TT100K \
  --victim-weights /path/to/yolov10_tt100k.pt \
  --checkpoint outputs/tt100k_yolov10/generator_best.pth

The evaluator reports clean-box disappearance ASR, average L_inf, and average L2 perturbation norm. ASR is class-agnostic: a clean detection is counted as vanished if no adversarial post-NMS box overlaps it with IoU >= 0.5.

mAP / Recall / F1

Generate adversarial validation images:

python tools/generate_adv_dataset.py \
  --data-config configs/datasets/tt100k.yaml \
  --attack-config configs/attack.yaml \
  --data-root /path/to/TT100K \
  --checkpoint /path/to/generator_best.pth \
  --output generated/tt100k_adv

Then run the victim detector's standard validation pipeline on the generated images using the original ground-truth labels. This keeps mAP50, mAP50-95, Recall, and F1 consistent with the detector implementation used in the experiment.

Ablation study

bash scripts/ablation_tt100k_yolov10.sh /path/to/yolov10_tt100k.pt --data-root /path/to/TT100K

The script runs:

Base: no trajectory guidance, no channel attention



trajectory guidance

Full: trajectory guidance + channel attention

Perturbation visualization

Figure-4(b)-style perturbation intensity heatmap:

python tools/visualize_perturbation.py \
  --image /path/to/example.jpg \
  --checkpoint /path/to/generator_best.pth \
  --output heatmap.png

This visualizes the spatial magnitude distribution of the final perturbation. It is not a direct visualization of channel-attention weights.

Top-5% perturbation enrichment analysis:

python tools/perturbation_enrichment.py \
  --images /path/to/images \
  --labels /path/to/yolo_labels \
  --checkpoint /path/to/generator_best.pth

Generator-only latency

python tools/compute_generator_latency.py \
  --checkpoint /path/to/generator_best.pth \
  --device cuda:0

What should not be uploaded

The .gitignore excludes common local/large assets. Do not commit:

TT100K or CCTSDB image/label files

detector .pt weights

generator checkpoints unless intentionally released

outputs/, logs, caches, or private absolute paths

Citation

@article{si2026trajadvgan,
  title={Dual-Loop Trajectory Guidance with Channel Attention for Efficient Adversarial Attacks on YOLO-Based Traffic Sign Detection},
  author={Si, Huachao and Wu, Ting and Zhou, Sizhe and Fang, Weijia and Kang, Xinyuan and Gao, Jingjie and Su, Tongtong},
  year={2026},
  note={Preprint}
}
