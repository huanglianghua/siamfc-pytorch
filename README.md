# SiamFC - PyTorch

A clean PyTorch implementation of SiamFC tracker described in paper [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html). The code is evaluated on 7 tracking datasets ([OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS](http://ci2cv.net/nfs/index.html) and [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)), using the [GOT-10k toolkit](https://github.com/got-10k/toolkit).

## Performance

### GOT-10k

| Dataset | AO    | SR<sub>0.50</sub> | SR<sub>0.75</sub> |
|:------- |:-----:|:-----------------:|:-----------------:|
| GOT-10k | 0.355 | 0.390             | 0.118             |

The scores are comparable with state-of-the-art results on [GOT-10k leaderboard](http://got-10k.aitestunion.com/leaderboard).

### OTB / UAV123 / DTB70 / TColor128 / NfS

| Dataset       | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| OTB2013       | 0.589            | 0.781            |
| OTB2015       | 0.578            | 0.765            |
| UAV123        | 0.523            | 0.731            |
| UAV20L        | 0.423            | 0.572            |
| DTB70         | 0.493            | 0.731            |
| TColor128     | 0.510            | 0.691            |
| NfS (30 fps)  | -                | -                |
| NfS (240 fps) | 0.520            | 0.624            |

### VOT2018

| Dataset       | Accuracy    | Robustness (unnormalized) |
|:-----------   |:-----------:|:-------------------------:|
| VOT2018       | 0.502       | 37.25                     |

## Dependencies

Install PyTorch, opencv-python and GOT-10k toolkit:

```bash
pip install torch opencv-python got10k
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 7 popular tracking datasets.

## Running the tracker

In the root directory of `siamfc-pytorch`:

1. Download pretrained `model.pth` from [Baidu Yun](https://pan.baidu.com/s/1TT7ebFho63Lw2D7CXLqwjQ) or [Google Drive](https://drive.google.com/open?id=1Qu5K8bQhRAiexKdnwzs39lOko3uWxEKm), and put the file under `pretrained/siamfc`.

2. Create a symbolic link `data` to your datasets folder (e.g., `data/OTB`, `data/UAV123`, `data/GOT-10k`):

```
ln -s ./data /path/to/your/data/folder
```

3. Run:

```
python test.py
```

By default, the tracking experiments will be executed and evaluated over all 7 datasets. Comment lines in `run_tracker.py` as you wish if you need to skip some experiments.

## Training the tracker

1. Assume the GOT-10k dataset is located at `data/GOT-10K`.

2. Run:

```
python train.py
```
