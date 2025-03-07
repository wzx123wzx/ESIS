# Efficient Superpixel-based Seamline Detection for Large-scale Image Stitching
A novel method for stitchig large-scale high resolution images with robust registration and efficient seamline detection.
You can find our paper [here](https://ieeexplore.ieee.org/document/10912483).
![image](workflow.png)

# About Registration
We have implemented an efficient linear affine registration optimization algorithm, 
but due to space limitations, 
it is not described in this paper.
It's an important pre-step for our seamline detection method,
please refer to [here](https://github.com/wzx123wzx/ESIS-Registration-Part) for more details about our registration optimization algorithm.

And you can also use other algorithms to generate aligned images as input data to test our seamline detection method.

# Dataset details
We evaluate our algorithm on 5 image datasets, which are all captured by Phantom 3 Advanced drone with resolution of 4000 × 3000 and available in Dronemapper website[6]. We also propose an efficient matching image pair selection method. The image number is shown following.

<table style="width:100%">
  <tr>
    <th>Dataset</th>
    <th>Number of images</th>
  </tr>
   </tr>
  <tr>
    <td>Gregg</td>
    <td>187</td>
  </tr>
  <tr>
    <td>Golf Course</td>
    <td>664</td>
  </tr>
  <tr>
    <td>4thAveReservoir</td>
    <td>82</td>
  </tr>
  <tr>
    <td>AdobeButtes1</td>
    <td>160</td>
  </tr>
  <tr>
    <td>AdobeButtes2</td>
    <td>371</td>
  </tr>
</table>

# Result details
Due to space limitation,
we only show our results of Golf Course dataset in the paper.
Here, we provide our detailed results on 5 test datasets.
Please refer to [Our_results](Our_results).

# How to test our code
## Data
You can use UAV image dataset from Dronemapper website [at this link](https://dronemapper.com/sample_data/) or your own dataset with GPS information to test our code.

## Requirements and Installation
Our code is implemented on a linux system.
In order to run our code, you need to make sure that you have all the required python packages and enough RAM. 
You can find the list of packages we installed on our conda environment at [this text file](requirements.txt). 
It is very important to install the same versions of some of these packages in order for the code to run.

## Running
Set "images_path" in the [main.py](Code/main.py) with your dataset path (make sure that there are only images in your folder and the image has GPS information).
Then, set your configuration in the [constant.py](Code/constant.py) and run [main.py](Code/main.py).

## Fast to test our code

1. modify the "images_path" with [test_data](test_data), "slic_num_pixels_per_superpixel" with 1000 in [constant.py](Code/constant.py)

2. run [main.py](Code/main.py)

And then you'll get the stitching results of a small dataset (4 images) in 20s.

# Cite our paper
If you find this method and the paper interesting and useful for your research, please cite our paper using 
```
@ARTICLE{Wang2025GRSL,
  author={Wang, Zhongxing and Fu, Zhizhong and Xu, Jin},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Efficient Superpixel-based Seamline Detection for Large-scale Image Stitching}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Optimization;Image stitching;Image segmentation;Image color analysis;Complexity theory;Minimization;Autonomous aerial vehicles;Remote sensing;Labeling;Costs;Multiple image stitching;seamline detection;superpixel segmentation;multi-label optimization;unmanned aerial vehicle (UAV) image},
  doi={10.1109/LGRS.2025.3548266}}
```

# Reference

1. “Dronemapper,” accessed: June. 12, 2023. [Online]. Available: https://dronemapper.com/sample_data/

