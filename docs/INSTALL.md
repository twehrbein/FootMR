# Install

## Environment

```bash
git clone git@github.com:twehrbein/FootMR.git --recursive --depth 1 --shallow-submodules --no-tags
cd FootMR

conda create -y -n footmr python=3.10
conda activate footmr
pip install -r requirements.txt
pip install -e .
```

### Optional: DPVO (slower but more accurate world-space reconstructions)
Follow DPVO installation instructions of [GVHMR](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md#optional-dpvo-not-recommended-if-you-want-fast-inference-speed).

### Optional: Sapiens (slower but more accurate 2D foot detections)
```bash
cd third-party/sapiens/engine
pip install -e . -v --no-build-isolation
cd ../cv
pip install -e . -v --no-build-isolation
pip install -r requirements/optional.txt
cd ../pretrain
pip install -e . -v --no-build-isolation
cd ../pose
pip install -e . -v --no-build-isolation
```

## Data

- Follow [GVHMR](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md#inputs--outputs) to prepare required checkpoints and training and testing data

- Download all data from [here](https://cloud.tnt.uni-hannover.de/index.php/s/tpLX3F6Mz4FqHaD)

- If needed, download Sapiens checkpoint: sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth from [here](https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_host/pose/checkpoints/sapiens_2b) 

The final folder structure should be like this:

```bash
inputs/checkpoints/
├── body_models/smpl/
│   └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
├── dpvo/
│   └── dpvo.pth
├── footmr/
│   └── footmr_checkpoint.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── sapiens/
│   └── sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth
├── vitpose/
│   ├── vitpose-h-multi-coco.pth
│   └── vitpose-h-wholebody.pth
└── yolo/
    └── yolov8x.pt

inputs/
├── 3DPW/hmr4d_support/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── MOOF/hmr4d_support/
├── MOYO/hmr4d_support/
└── RICH/hmr4d_support/
```
