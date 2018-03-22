# SiamFC - PyTorch
PyTorch port of the tracking method described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).

(The code structure as well as many functions are directly borrowed from [siamfc-tf](https://github.com/torrvision/siamfc-tf))

In particular, it is the improved version presented as baseline in [*End-to-end representation learning for Correlation Filter based tracking*](https://www.robots.ox.ac.uk/~luca/cfnet.html), which achieves state-of-the-art performance at high framerate. The other methods presented in the paper (similar performance, shallower network) haven't been ported yet.

**Note1**: results should be similar (i.e. slightly better or worse) than the MatConvNet implementation. However, for direct comparison please refer to the precomputed results available in the project pages or to the original code, which you can find pinned in [Bertinetto's Github](https://github.com/bertinetto).

**Note2**: at the moment this code only allows to use a pretrained net in forward mode.

## Preparations for running the code
1) Install opencv-python:
`pip install opencv-python`
1) Install PyTorch:
follow the instructions at [PyTorch website](http://pytorch.org/).
1) Install SciPy:
`pip install scipy`
1) Clone the repository
`git clone https://github.com/huanglianghua/siamfc-pytorch.git`
1) `cd siamfc-pytorch`
1) `mkdir pretrained data`
1) Download the [pretrained networks](https://drive.google.com/file/d/0B7Awq_aAemXQZ3JTc2l6TTZlQVE/view) in `pretrained` and unzip the archive (we will only use `baseline-conv5_e55.mat`)
1) Download [VOT dataset](http://www.votchallenge.net/vot2016/dataset.html) in `data` and unzip the archive.


## Running the tracker
1) Set `video` from `parameters.evaluation` to `"all"` or to a specific sequence (e.g. `"ball1"`)
1) See if you are happy with the default parameters in `parameters/hyperparameters.json`
1) Optionally enable visualization in `parameters/run.json`
1) Call the main script (within an active virtualenv session)
`python run_tracker_evaluation.py`

## References
If you find this work useful, please consider citing

↓ [Original method] ↓
```
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850--865},
  year={2016}
}
```
↓ [Improved method and evaluation] ↓
```
@article{valmadre2017end,
  title={End-to-end representation learning for Correlation Filter based tracking},
  author={Valmadre, Jack and Bertinetto, Luca and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip HS},
  journal={arXiv preprint arXiv:1704.06036},
  year={2017}
}
```

## License
This code can be freely used for personal, academic, or educational purposes.
Please contact us for commercial use.
