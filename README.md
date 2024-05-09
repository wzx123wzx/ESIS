# Efficient Superpixel-based Large-scale Image Stitching
A novel method for stitchig large-scale high resolution images while being robust to drift and inconsistencies. 
# Data
You can use UAV image dataset from Dronemapper website [at this link](https://dronemapper.com/sample_data/) to test our code.
# Requirements and Installation
Currently, there are no installation scripts for this repo. 
In order to run our code, you need to make sure that you have all the required python packages and files, and set correct configurations. 
You can find the list of packages we installed on our conda environment at [this text file](requirements.txt). 
It is very important to install the same versions of some of these packages in order for the code to run.
# Configurations
You can change our settings in the [Settings_manager.py](Settings_manager.py).
# Running
Set "images_path" in the [MAIN.py](MAIN.py) with your dataset path.
Make sure that your image have GPS information.
