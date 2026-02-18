[![DOI](https://zenodo.org/badge/380315103.svg)](https://zenodo.org/badge/latestdoi/380315103)
# RgtNet:using synthetic datasets to train an end-to-end CNN for 3-D RGT(Relative Geologic Time) estimation
**This is a [Pytorch](https://pytorch.org/) version of RgtNet for 3-D RGT(Relative Geologic Time) estimation**

## Attribution

This repository is a fork/adaptation of the original code shared by the authors of the article:

**Deep learning for simultaneously interpreting 3D seismic horizons and faults by estimating a relative geologic time volume**

Article link: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021JB021882

Original code and release record: https://zenodo.org/records/5090317

All credit for the original method, paper, and initial codebase belongs to the original authors.

## Getting Started with Example Model for RGT estimation

If you would just like to try out a pretrained example model, then you can download the [pretrained model](https://pan.baidu.com/s/1SDTRIc4yggoQYlFPPFa0dQ) [neuc] and use the [demo.ipynb](https://github.com/zfbi/rgtNet/blob/main/demo.ipynb) script to run a demo (example data can be downloaded from [here](https://drive.google.com/drive/folders/12waSUwNHRwdo4g-Ag_xXH1pCNKf2i1Js?usp=sharing)).

### Requirments

```
python>=3.6
torch>=1.0.0
torchvision
torchsummary
natsort
numpy
pillow
plotly
pyparsing
scipy
scikit-image
sklearn
tqdm
```
Install all dependent libraries:
```bash
pip install -r requirements.txt
```

### Dataset

**To train our CNN network, we automatically created 400 pairs of synthetic seismic and corresponding RGT volumes, which were shown 
to be sufficient to train a good RGT estimation network.** 

**Original training data (400 pairs) can be found here: https://doi.org/10.5281/zenodo.4536561**

After extraction, this repository expects the following structure:

```text
datasets/
	syn/
		seis/                 # seismic input volumes (.dat)
		rgt/                  # RGT label volumes (.dat)
		fault/                # fault labels (.dat), optional for current training script
	syn_training/
		seis -> ../syn/seis   # symlink to syn/seis
		rgt  -> ../syn/rgt    # symlink to syn/rgt
	syn_testing/
		seis -> ../syn/seis   # symlink to syn/seis
		rgt  -> ../syn/rgt    # symlink to syn/rgt
```

Additional folders for real field data workflow:

```text
data/
	realSeismic/            # put your real input SEG-Y files (*.segy)
	realSeismicOutputs/     # place exported model results as SEG-Y (*.segy)
```

Notes:
- The dataloader used by `train.py` and `infer.py` reads `seis/` and `rgt/` files by matching file names.
- Current inference code writes binary `.dat` volumes under `sessions/<session_name>_Test/bin/`.
- If you export to SEG-Y, store those exported files in `data/realSeismicOutputs/`.

### Training

Run train.sh to start training a new RgtNet model by using the synthetic dataset
```bash
sh train.sh
```

### Retraining Step-by-Step

1. Install dependencies:
```bash
pip install -r requirments.txt
```

2. Verify dataset folders exist:
```bash
ls datasets/syn/seis | head
ls datasets/syn/rgt  | head
```

3. (Optional) adjust training parameters in `train.sh`:
- `--dataset_size` and `--dataset_size_val`
- `--batch_size`
- `--shape 256 256 128`
- `--lr`, `--nepochs`, and `--loss_type`

4. Start retraining:
```bash
sh train.sh
```

5. Track outputs generated during training:
- checkpoints: `sessions/<session_name>_Train/checkpoint/`
- logs/history: `sessions/<session_name>_Train/history/`
- validation images: `sessions/<session_name>_Train/picture/`

6. Run inference with a trained model checkpoint:
```bash
sh infer.sh
```

7. Inference outputs are saved to:
- binary predictions: `sessions/<session_name>_Test/bin/pred/`
- binary copied inputs: `sessions/<session_name>_Test/bin/seis/`
- figures: `sessions/<session_name>_Test/picture/`

8. (Optional) convert output `.dat` to `.segy` with your preferred seismic toolchain and store exported files in:
```text
data/realSeismicOutputs/
```

### Notebook Workflows (Current)

This repository now includes two independent notebook training pipelines:

- `train_rgt_only.ipynb`: trains only seismic ➜ RGT.
- `train_rgt_fault.ipynb`: trains a multitask model for seismic ➜ (RGT + fault).

Both notebooks:
- use `datasets/syn` as data root,
- create session outputs under `sessions/<session_name>_Train/`,
- save checkpoints/history/figures,
- include GPU-first setup and mixed-precision support.

### Current Training Parameters (Notebook Defaults)

#### RGT-only (`train_rgt_only.ipynb`)
- shape: `(256, 256, 128)`
- batch size: `1`
- epochs: `400`
- learning rate: `8e-4`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau(factor=0.5, patience=2)`
- loss: `MSE` (default; `SSIM` also supported)
- model backbone: `net3d.model` with `input_channels=1`, `encoder_channels=512`, `decoder_channels=16`

#### Joint RGT+Fault (`train_rgt_fault.ipynb`)
- shape: `(256, 256, 128)`
- batch size: `1`
- epochs: `200`
- learning rate: `8e-4`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau(factor=0.5, patience=2)`
- RGT loss: `MSE` (default; `SSIM` optional)
- Fault loss: `BCEWithLogitsLoss` (with estimated `pos_weight`)
- joint objective: `lambda_rgt * rgt_loss + lambda_fault * fault_loss`
- model: shared `net3d` RGT branch + fault head (`Conv3d` stack)

### Current Project Files and Purpose

- `train.py`, `train.sh`: original script-based RGT training entry points.
- `infer.py`, `infer.sh`: script-based inference entry points.
- `demo.ipynb`: demo inference notebook with dataset-path and checkpoint discovery improvements.
- `train_rgt_only.ipynb`: standalone RGT-only training workflow.
- `train_rgt_fault.ipynb`: standalone multitask training workflow.
- `lossf/loss.py`: includes both `ssim3DLoss` and `mse3DLoss`.
- `data/realSeismic/` and `data/realSeismicOutputs/`: placeholders for SEG-Y input/output workflows.

### Recent Work Completed in This Repo

- dataset extraction and structure alignment under `datasets/syn/{seis,rgt,fault}`
- README setup/retraining documentation refresh
- `.gitignore` expanded for datasets, sessions, checkpoints, archives, and binary artifacts
- demo notebook fixes for dataset root and checkpoint lookup
- creation of independent RGT-only and RGT+fault notebooks
- memory-focused notebook updates:
	- AMP API updated to `torch.amp.autocast(...)`
	- reduced memory defaults (`batch_size=1`, fewer workers)
	- streaming epoch metrics (no full-epoch tensor accumulation)
	- explicit CUDA OOM handling messages

### Validation & Application
Run infer.sh to start applying a new RgtNet model to the synthetic or field seismic data
```bash
sh infer.sh
```

## License

This extension to the Pytorch library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
