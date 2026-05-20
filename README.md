# topo3d-mito

Weakly-supervised topology-preserving 3D mitochondria segmentation in fluorescence microscopy.

This codebase is built on top of [PyTorch Connectomics (v1.0.0)](https://github.com/zudi-lin/pytorch_connectomics/tree/v1.0.0) with added topology-aware losses (3D topological loss, slice-wise topological loss, clDice), validation/Betti-1 evaluation scripts, and trained checkpoints for the mitochondria experiments.

---

## Installation

### 1. Clone with Git LFS

Checkpoint files (`*.pth.tar`) are stored using [Git LFS](https://git-lfs.com/). Install LFS first, then clone:

```bash
git lfs install
git clone https://github.com/seo0229/topo3d-mito.git
cd topo3d-mito
```

### 2. Environment

Tested with Python 3.8+ and PyTorch 1.8+ on Linux with CUDA.

```bash
conda create -n topo3d-mito python=3.9 -y
conda activate topo3d-mito
# install PyTorch matching your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -e .
```

### 3. Topology-loss dependencies (heads up — non-trivial)

The topological loss requires:

- **`gudhi`** — `pip install gudhi` (or `conda install -c conda-forge gudhi`)
- **`cripser`** — `pip install cripser` (compiles C++, needs a working toolchain)
- **`pot`** (Python Optimal Transport) — `pip install pot` (or `conda install -c conda-forge pot`)

These are listed in `setup.py` so `pip install -e .` should pick them up, but `cripser` in particular may fail on environments without a C++ build toolchain. On those systems install them separately first.

---

## Repository layout

```
configs/
  MitoPre-train.yaml          # baseline pre-training config
  new-folds/
    fold1/, fold2/, fold3/    # 3-fold cross-validation configs per loss variant
connectomics/                 # forked from pytorch_connectomics v1.0.0
  model/loss/
    topoloss.py               # TopoLossMSE2D, TopoLossMSE2D_slice, TopoLossMSE3D
    cldice.py, soft_skeleton.py  # soft clDice loss
    criterion.py              # extended loss registry
  engine/trainer.py           # per-component validation loss logging added
scripts/
  main.py                     # train / inference entry point (upstream)
  validate.py                 # Dice / clDice / Betti-0 / IoU / connectivity evaluation
  betti1.py                   # Betti-1 error
outputs/                      # released checkpoints (Git LFS)
  baseline/Pre-train/         # pre-trained model used as PRE_MODEL for fold runs
  fold{1,2,3}/{Topo3D,Toposlice,clDice,woTopoloss}/
```

### Loss variants

| Folder name   | Config name                       | Loss combination                                |
|---------------|-----------------------------------|-------------------------------------------------|
| `woTopoloss`  | `Mito3D-GTE.yaml`                 | WeightedBCE + DiceLoss                          |
| `Topo3D`      | `Mito3D-Topo_3D_BCE.yaml`         | WeightedBCE + DiceLoss + TopoLossMSE3D          |
| `Toposlice`   | `Mito3D-Topo_slice_BCE.yaml`      | WeightedBCE + DiceLoss + TopoLossMSE2D_slice    |
| `clDice`      | `Mito3D-cldice.yaml`              | WeightedBCE + DiceLoss + SoftCLDice             |

---

## Training

Pre-training (baseline):

```bash
python scripts/main.py --config-file configs/MitoPre-train.yaml
```

Fine-tuning on a fold (uses the pre-trained checkpoint via `MODEL.PRE_MODEL`):

```bash
python scripts/main.py --config-file configs/new-folds/fold1/Mito3D-Topo_3D_BCE.yaml
```

Applying CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/main.py --config-file configs/new-folds/fold1/Mito3D-Topo_3D_BCE.yaml
```

---

## Inference

```bash
python scripts/main.py \
  --config-file configs/new-folds/fold1/Mito3D-Topo_3D_BCE.yaml \
  --inference \
  --checkpoint outputs/fold1/Topo3D/checkpoint_best.pth.tar
```

---

## Evaluation

`scripts/validate.py` writes an Excel report with Dice / soft-Dice / clDice / Betti-0 / IoU and a connectivity-sampling accuracy.

```bash
# TODO: document exact invocation (args and inputs validate.py expects)
python scripts/validate.py ...
```

`scripts/betti1.py` computes cube-wise Betti-1 error.

```bash
# TODO: document exact invocation
python scripts/betti1.py ...
```

---

## Citation

<!-- TODO: paper / report citation -->

If you use this code, please also cite the underlying PyTorch Connectomics framework:

```
@article{lin2021pytorch,
  title   = {PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
  author  = {Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
  journal = {arXiv preprint arXiv:2112.05754},
  year    = {2021}
}
```

---

## License

MIT — see [LICENSE](LICENSE). Inherited from the upstream PyTorch Connectomics project.

## Acknowledgments

- [PyTorch Connectomics](https://github.com/zudi-lin/pytorch_connectomics) — base framework.
- clDice — Shit *et al.*, *clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021.
- Topological loss — Hu *et al.*, *Topology-Preserving Deep Image Segmentation*, NeurIPS 2019.
