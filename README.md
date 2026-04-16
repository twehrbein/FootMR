# Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture

[![Website](https://img.shields.io/badge/Website-FootMR-orange)](https://twehrbein.github.io/footmr-website/) [![arXiv](https://img.shields.io/badge/arXiv-2603.09681-red)](https://arxiv.org/abs/2603.09681)

> **Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture** <br>
> [Tom Wehrbein](https://www.tnt.uni-hannover.de/en/staff/wehrbein/) and [Bodo Rosenhahn](https://www.tnt.uni-hannover.de/en/staff/rosenhahn/) <br>
> 2026 International Conference on 3D Vision (3DV)

## Abstract

<img src="./assets/teaser.jpg"/>

State-of-the-art methods can recover accurate overall 3D human body motion from in-the-wild videos. However, they often fail to capture fine-grained articulations, especially in the feet, which are critical for applications such as gait analysis and animation. This limitation results from training datasets with inaccurate foot annotations and limited foot motion diversity. We address this gap with FootMR, a Foot Motion Refinement method that refines foot motion estimated by an existing human recovery model through lifting 2D foot keypoint sequences to 3D. By avoiding direct image input, FootMR circumvents inaccurate image–3D annotation pairs and can instead leverage large-scale motion capture data. To resolve ambiguities of 2D-to-3D lifting, FootMR incorporates knee and foot motion as context and predicts only residual foot motion. Generalization to extreme foot poses is further improved by representing joints in global rather than parent-relative rotations and applying extensive data augmentation. To support evaluation of foot motion reconstruction, we introduce MOOF, a 2D dataset of complex foot movements. Experiments on MOOF, MOYO, and RICH show that FootMR outperforms state-of-the-art methods, reducing ankle joint angle error on MOYO by up to 30% over the best video-based approach.

## TODOs

- [x] Project page
- [x] arXiv paper
- [X] Code release
- [ ] MOOF dataset release (coming soon)

## Setup

Please see [installation](docs/INSTALL.md) for details.


## Quick Start

### Demo

We provide a demo script that takes a video as input and reconstructs the most prominent person in the scene.

* Use `-s` to skip visual odometry if the camera is static.
* Use `--use_sapiens` for more accurate foot reconstruction (at the cost of heavily increased runtime).

```bash
python tools/demo.py --video {PATH_TO_VIDEO}

# run inference on provided example video:
python tools/demo.py --video docs/example_video/stepdance.mp4
```

### Reproducing Results

#### 1. Evaluation

To reproduce the results on **MOOF**, **MOYO**, and **RICH** in a single run, execute:

```bash
python tools/train.py \
    global/task=footmr/test_feet \
    exp=footmr/mixed/mixed \
    ckpt_path=inputs/checkpoints/footmr/footmr_checkpoint.ckpt
```

#### 2. Training

To train the model from scratch, run:

```bash
python tools/train.py exp=footmr/mixed/mixed exp_name_var={EXP_NAME}
```


## Citation

If you find this work useful, please cite our paper and the original [GVHMR](https://github.com/zju3dv/GVHMR) work:

```bibtex
@InProceedings{wehrbein26footmr,
    author    = {Wehrbein, Tom and Rosenhahn, Bodo},
    title     = {Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture},
    booktitle = {International Conference on 3D Vision (3DV)},
    year      = {2026},
}
```

```bibtex
@InProceedings{shen2024gvhmr,
    author    = {Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
    title     = {World-Grounded Human Motion Recovery via Gravity-View Coordinates},
    booktitle = {SIGGRAPH Asia Conference Proceedings},
    year      = {2024},
}
```