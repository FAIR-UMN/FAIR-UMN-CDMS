# DNN Models

## Overview 

This folder contains Python code/Jupyter notebooks for deep neural network models which are located in `DNN_Regression`, including:


- **model_files**: 1) *nn_model.py* defines the neural network model; 2) *util.py* includes functions such as loading dataset, loss function, and visualization.

- **train_deep_regression.ipynb**: the main Jupyter notebook in this folder. It runs and tests our DNN model. It will call the backbone functions located in *model_files*.

  




## The Deep Neural Network Model

We formulate the problem as a [regression problem](https://wiki2.org/en/Regression_analysis) in which we want the DNN model to learn a function that can map the features to its corresponding positions/targets. The detailed design of our DNN model can be found in the [complementary document](https://github.com/ml-deepai/FAIR-UMN/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf). In here, we merely briefly describe the important hyper-parameters we used in training the DNN models. 

- *normalization_type*:  the method used to normalize the dataset; it should be selected from { 'StandardScaler', 'MinMaxScaler'}. The details for these normalization methods can be found [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing).
- *layer_num*: the hidden layers for the DNN model; it can be any  positive integer number. But please keep in mind that a very large *layer_num* means a more complicated neural network which may need longer training time and at the same time may increase the risk of overfitting. Therefore, it should be properly set according to the complicity of tasks. In our demo example, it is set to be 2.
-  *input_dim*: the number of input features. In our demo example, it is set to be 19.
-  *hidden_dim*: the number of neurons in each hidden layer; a large number means a large and complicated model. It should be set properly according to the complicity of tasks as well as the *layer_num*. In our demo example, it is set to be 32.
-  *output_dim*: the number of neuron is the output layer. In our demo example, it is set to be 1.
- *trn_batch_size*: the batch size for training set.
- *val_batch_size* : the batch size for validation set.
- *tst_batch_size* : the batch size for test set (HOS).
- *gpu_id*： the ID for the GPU device. If you do not have a GPU device, then it can be set to be any number and it will be ignored and use CPU directly.
- *learning_rate*: the learning rate to update neural network's weights (by default we set it to be 0.001).
- *max_epoch*: the total training epochs (we set it to be 500).
- *check_freq*: the frequency to validate the learned model (we set it to be 1).



*Note*: it is very easy to modify the code to obtain deep neural network models with different complexity levels. To achieve such goal, you can simply modify the *layer_num* and *hidden_dim*. 



For more details about the neural network model we used and its performance results, please check our [complementary document](https://github.com/ml-deepai/FAIR-UMN/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf). 



