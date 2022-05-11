# Inpainting Error Maximization

Author implementation of [Inpainting Error Maximization](https://arxiv.org/abs/2012.07287).

## Requirements
The code has been developed and tested with Python 3.8.8 and PyTorch 1.8.0.

## Running IEM

The repo will run IEM on the Flowers dataset by default, so you should download and prepare a folder with the dataset images beforehand.

To run IEM:

```
python main.py PATH_TO_DATASET
```

PATH_TO_DATASET should point to a folder with setid.mat and subdirectories 'jpg' and 'segmin'.

You can also change settings, such as the kernel size (--kernel-size) and the number of convolutions (--reps) for the Gaussian filter, through command-line arguments.

With the default settings (--kernel-size 11 --reps 2 --sigma 5.0), you should get the following output, which yields an IoU of 76.9 with ~77s runtime on a nVidia 1080ti.

```
Batch 1/1
	Iter   0: InpError 0.848 IoU 0.505 DICE 0.655
	Iter   1: InpError 0.887 IoU 0.520 DICE 0.667
	...
	Iter 148: InpError 1.435 IoU 0.769 DICE 0.846
	Iter 149: InpError 1.435 IoU 0.769 DICE 0.846
IEM finished in 77.1 seconds
```

There is also a --scale-factor argument that can be used to speed up IEM by running the inpainter on a downsampled version of the image. This was not used for any of the results in the paper but often a similar performance can be achieved in less time by using that extra feature. For example,
```
python main.py PATH_TO_DATASET --sigma 2.5 --kernel-size 7 --scale-factor 2
```
will make IEM perform 2x downsampling on images before the inpainting procedure (sigma was set to 2.5 instead of 5.0 and kernel-size to 7 instead of 11 to compensate downscaling). This will yield 77.1 IoU (actually improving the performance) with a running time of ~49s compared to ~77s without downsampling.
```
Batch 1/1
	Iter   0: InpError 0.823 IoU 0.509 DICE 0.658
	Iter   1: InpError 0.879 IoU 0.522 DICE 0.668
	...
	Iter 148: InpError 1.439 IoU 0.771 DICE 0.848
	Iter 149: InpError 1.439 IoU 0.771 DICE 0.848
IEM finished in 49.0 seconds
```

## Citation
```
@inproceedings{
savarese2021iem,
  title={Information-Theoretic Segmentation by Inpainting Error Maximization},
  author={Savarese, Pedro and Kim, Sunnie SY and Maire, Michael and Shakhnarovich, Greg and McAllester, David},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
