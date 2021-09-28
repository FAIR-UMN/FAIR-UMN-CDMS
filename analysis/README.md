# Results Analysis




## Overview

This folder contains Python code/Jupyter notebooks for analyzing the prediction results. It is straight-forward and simple, including:


- **1_statistical_info.ipynb**: 1) summarizing the RMSE for training/validation/test set; 2) getting statistical information (mean and standard deviation) for these predictions. 

- **2_visualization.ipynb**: visualize the histogram of predictions made by the deep neural network models.

  

To use this Jupyter notebook, you only need to set three hyper-parameters:

-  *layer_num* : including all *layer_num* you have set up when training the deep neural network models.
-  *hidden_dim*: including all *hidden_dim* you have set up when training the deep neural network models.
-  *held_out_positions*： the positions that are held out for test.
- *root_dir*: the directory where the prediction results are located. By default, it is `src/DNN_Regression/deepnn_results/test_results`.

After running *1_statistical_info.ipynb*， you will get the result in `stat_results/stat_info.csv`.



For more details about the neural network model we used and its performance results, please check our [complementary document](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf). 



