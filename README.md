# Efficient Superpixel-based Large-scale Image Stitching
A novel method for stitchig large-scale high resolution images with robust registration and efficient graph-cut.
You can find our paper [here](https://github.com/).
If you find this method and the paper interesting and useful for your research, please cite us using the following bibliography.
This README contains instructions on how to get the data that were used in the paper, install dependencies, and run our code.
# Data
You can use UAV image dataset from Dronemapper website [at this link](https://dronemapper.com/sample_data/) to test our code.
# Requirements and Installation
Our code is implemented on a linux system with 236 GB of RAM.
In order to run our code, you need to make sure that you have all the required python packages and enough RAM. 
You can find the list of packages we installed on our conda environment at [this text file](requirements.txt). 
It is very important to install the same versions of some of these packages in order for the code to run.
# Running
Set "images_path" in the [Main.py](Main.py) with your dataset path (make sure that there are only images in your folder and the image has GPS information).
Then, set your configuration in the [Settings_manager.py](Settings_manager.py) and run our [Main.py](Main.py).
![mosaic](https://github.com/wzx123wzx/Large-scale-image-stitching/assets/71632237/2064404f-2007-434b-9dc3-7d1d8021090c)
