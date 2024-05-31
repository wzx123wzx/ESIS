# Efficient Superpixel-based Large-scale Image Stitching
A novel method for stitchig large-scale high resolution images with robust registration and efficient graph-cut.
You can find our paper [here](https://github.com/).
![workflow](https://github.com/wzx123wzx/Large-scale-image-stitching/assets/71632237/eb7bb039-954f-4ab0-b63e-c6176684c721)

If you find this method and the paper interesting and useful for your research, please cite us using 
```
@article{zarei2021megastitch,
  title={MegaStitch: Robust Large Scale Image Stitching},
  author={Zarei, Ariyan and Gonzalez, Emmanuel and Merchant, Nirav and Pauli, Duke and Lyons, Eric and Barnard, Kobus},
  year={2021},
  publisher={TechRxiv}
}

```
This README contains instructions on how to get the data that were used in the paper, install dependencies, and run our code.

# Performance
 <table style="width:100%">
  <tr>
    <th>Dataset</th>
    <th>Number of images</th>
    <th>Number of matching image pair</th>
  </tr>
   </tr>
  <tr>
    <td>Gregg</td>
    <td>187</td>
    <td>338</td>
  </tr>
  <tr>
    <td>Golf Course</td>
    <td>664</td>
    <td>1424</td>
  </tr>
  <tr>
    <td>4thAveReservoir</td>
    <td>82</td>
    <td>158</td>
  </tr>
  <tr>
    <td>AdobeButtes1</td>
    <td>160</td>
    <td>310</td>
  </tr>
  <tr>
    <td>AdobeButtes2</td>
    <td>371</td>
    <td>748</td>
  </tr>
</table>

:::: table*
::: center
               Measure                          Methods              Datasets                                                  
  ---------------------------------- ------------------------------ ----------- ------------- ----------------- -------------- --------------
                 3-7                                                 Gregg1_2    Golf Course   4thAveReservoir   AdobeButtes1   AdobeButtes2
             RMSE(pixels)                        MGRAPH                7.57         3.81            2.60             2.27           5.91
                                               MegaStitch              2.31         1.65            1.94             1.52           1.51
                                                  Ours               **2.23**     **1.52**        **1.62**         **1.41**       **1.47**
   Transformation Optimization Time              MGRAPH               16m43s      5h13m25s          3m14s           16m14s        1h28m58s
                                               MegaStitch               21s         6m30s            3s               9s           1m59s
                                                  Ours                **7s**      **4m4s**         **1s**           **4s**        **31s**
       Optimization Term Number                  MGRAPH                21246        88268           9812            20534          46547
                                           MegaStitch affine           28826       118194           13426           27708          62650
                                      MegaStitch bundle adjustment     14410        59094           6710            13851          31322
                                                  Ours               **13520**    **56960**       **6320**        **12400**      **29920**
:::
::::



# Data
You can use UAV image dataset from Dronemapper website [at this link](https://dronemapper.com/sample_data/) or your own dataset with GPS information to test our code.

# Requirements and Installation
Our code is implemented on a linux system.
In order to run our code, you need to make sure that you have all the required python packages and enough RAM. 
You can find the list of packages we installed on our conda environment at [this text file](requirements.txt). 
It is very important to install the same versions of some of these packages in order for the code to run.

# Running
Set "images_path" in the [Main.py](py/Main.py) with your dataset path (make sure that there are only images in your folder and the image has GPS information).
Then, set your configuration in the [Settings_manager.py](py/Settings_manager.py) and run our [Main.py](py/Main.py).
