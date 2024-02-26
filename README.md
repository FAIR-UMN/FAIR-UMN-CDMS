# FAIR-UMN-CDMS: Identifying Interaction Location in SuperCDMS Detectors



## Overview

This github repository contains the code for analyzing and modeling the CDMS data. The code is written in Python. In most cases, we provide a Jupyter notebook with inline descriptions while in some cases (e.g., the model and its  auxiliary functions) we provide a Jupyter notebook and python scripts. 

Besides this github repository, we have a complementary document that provides background for this problem, describes the purposes of this project,  and introduces the dataset and neural network models in more detail. The document can be [downloaded here](https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf). 

We also have a project web-page which can be accessed via [this link](https://fair-umn.github.io/FAIR-UMN-CDMS/).



This repository includes the following folders and/or files. For *folders*, we provide a respective *Readme* document inside. The structure of folders is shown below: 

```
.
├── data                                  /* All you need to preprocess the dataset.
│   └── reduced_dataset
│       └── reduced_data.csv              /* Dataset with reduced set of features
│   └── full_dataset
│       └── full_data.csv                 /* Dataset with full 85 features
|   ├── raw_txt                           /* Original dataset in the txt format.
|   ├── processed_csv                     /* Extracted features in the csv format.
|   ├── dnn_dataset                       /* Training/validation/test data for deep neural network models.
|   ├── feature_analysis                  /* Results/figures for visualizing the distribution of features (before/after normalization).
|   ├── 1_txt2csv.ipynb                   /* Jupyter notebook to extract features from original dataset (in the txt format).
|   ├── 2_prepare_dataset.ipynb           /* Jupyter notebook to get training/validation/test dataset.             
|   ├── 3_features_analysis.ipynb         /* Jupyter notebook to visualize the distribution of features (before/after normalization).     
|   └──  README.md 
|
├── Exercises
│   ├── 01-Intro2FAIR.ipynb               /* Introduction to F(indability)A(ccessability)I(nteroperability)R(eproducibility) principles
│   ├── 02-FAIRCheck-MNIST.ipynb          /* Example of using FAIR standards on the MNIST dataset
│   ├── 03-FAIRCheck-CDMS.ipynb           /* Application of FAIR principles to CDMS dataset
│   ├── 04a-CDMS-LR.ipynb                 /* Linear Regression using `scikit` (part I) 
│   ├── 04b-CDMS-LR-FAIR.ipynb            /* Linear Regression Model (part II)
│   ├── 05a-CDMS-PCA.ipynb                /* Using Principal Component Analysis
│   ├── 07a-CDMS_NNRegressor.ipynb        /* Using Multilayer Perceptrons (part I)
│   ├── 07b-CDMS_NNRegressor-FAIR.ipynb   /* Using Multilayer Perceptrons (part I)
│   ├── 08a-CDMS_NNVAE.ipynb              /* Introduction to Variational Autoencoders (NNVAE) (part I)
│   ├── 08b-CDMS_NNVAE-FAIR.ipynb         /* NNVAE (part II)
│   └── 09-CNN-CDMS-FAIR.ipynb            /* Convolutional Neural Networks (CNNs)
|
├── src                                   /* All you need to train the deep neural network (DNN) models.
|   ├── DNN_Regression                      
|   |   ├── model_files                   /* A folder that hosts the backbone of deep neural network models (e.g., model definition and its auxiliary functions).
|   |   ├── train_deep_regression.ipynb   /* Jupyter notebook to train and test the deep neural network models. 
|   └── README.md 
|
|
├── analysis                              /* All you need to analyze the results generated by the DNN model.
|   ├── hist_results                      /* The histogram of predictions.
|   ├── stat_results                      /* The statistical information of predictions (e.g., the mean and standard deviation).
|   ├── 1_statistical_info.ipynb          /* Jupyter notebook to get the statistical information of predictions.         
|   ├── 2_visualization.ipynb             /* Jupyter notebook to generate the histogram of predictions.   
|   └── README.md  
|
|
└── doc                                   /* Documents for our project.
|
|
└── fair_gpu.yml                          /* The YML file to create a GPU execution environment.
|
|
└── fair_cpu.yml                          /* The YML file to create a CPU execution environment.
|
|
└── LICENSE                               /* The MIT LICENSE.


```


Below, we provide more details about each folder.

- **data**. This folder contains several Jupyter notebooks for preprocessing the dataset and preparing the dataset for use with the neural network models.

- **src**. This folder contains the core code for neural network models, including the model architecture, auxiliary functions, and a Jupyter notebook to train and validate the model.

- **analysis**. This folder contains several Jupyter notebooks that are used to extract the predictions made by the neural network models, format these predictions into a neat csv file, and analyze these predictions (e.g., generating figures and obtaining statistics such as mean and standard deviations of predictions).


## Get Started

We provided two options for users to set up the execution environment: 
- we provide the envrionment YML file so that one can set up the execution environment with it directly;
- we provide the detailed steps and commands to install each required package. 

Before starting, be sure to have the [git](https://git-scm.com/) and [Anaconda3](https://www.anaconda.com/products/individual) installed (alternatively, you can also use [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) instead of[Anaconda3](https://docs.anaconda.com/free/anaconda/install/linux/), which has been tested by us and works well for our demo).

### Set up from the YML file

1. Get and clone the github repository:

   `git clone https://github.com/FAIR-UMN/FAIR-UMN-CDMS/`

2. Switch to `FAIR-UMN-CDMS` :

   `cd XXX/FAIR-UMN-CDMS`  (*Note*: `XXX` here indicates the upper directory of `FAIR-UMN`. For example, if you clone `FAIR-UMN-CDMS` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use [Anaconda3](https://www.anaconda.com/products/individual-d)):

   `conda deactivate`

4. Create a new conda environment with the YML file (choose GPU or CPU version according to your computational resources):

    GPU version run: `conda env create -f fair_gpu.yml`
   
    CPU version run: `conda env create -f fair_cpu.yml`

5.  Activate conda environment:
    
    `conda activate fair_gpu` (If you choose the GPU version in Step4)
    
    `conda activate fair_cpu` (If you choose the CPU version in Step4)

6. You are now ready to explore the codes/models! Please remember to follow this order: *data*->*src*->*analysis*



### Set up from the source

1. Get and clone the github repository:

   `git clone https://github.com/FAIR-UMN/FAIR-UMN-CDMS/`

2. Switch to `FAIR-UMN-CDMS` :

   `cd XXX/FAIR-UMN-CDMS`  (*Note*: `XXX` here indicates the upper directory of `FAIR-UMN-CDMS`. For example, if you clone `FAIR-UMN-CDMS` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use [Anaconda3](https://www.anaconda.com/products/individual-d)):

   `conda deactivate`

4. Create a new conda environment:

   `conda create -n fair_umn python=3.6`

5.  Activate conda environment:
    
    `conda activate fair_umn`

6. Install Pytorch (choose GPU or CPU version according to your computational resources):

   GPU version run: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
   
   CPU version run: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   
7. Install scikit-learn/pandas/matplotlib/numpy/seaborn/tqdm/Jupyter notebook

   ```
   pip install scikit-learn
   pip install pandas
   pip install matplotlib
   pip install numpy
   pip install seaborn
   pip install tqdm
   pip install notebook
   ```
   
8. You are now ready to explore the codes/models! Please remember to follow this order: *data*->*src*->*analysis*

   
*Note*: 
1) To install Anaconda, please follow its [official guideline](https://docs.anaconda.com/anaconda/user-guide/getting-started/). For example, to install Anaconda3 on Linux, check [here](https://docs.anaconda.com/anaconda/install/linux/); to install Anaconda3 on Windows, check [here](https://docs.anaconda.com/anaconda/install/windows/); and to install Anaconda3 on macOS, check [here](https://docs.anaconda.com/anaconda/install/mac-os/).
3) We test our model on Ubuntu, Windows, and macOS.


## Additional FAIR Exercies

Different types of ML approaches to this regression problem can be found in a folder labeled `Exercises`. This folder primarily containes Jupyter notebooks which contains detailed explanations for each method. The notebooks were originally authored by Avik Roy ([original repo](https://github.com/yorkiva/FAIR-Exercises)), which have now been merged with the FAIR-UMN-CDMS repository. To run the notebooks, one would have to setup a conda environment using the above instructions. 


## Convolutional Neural Networks

Convolution Neural Networks can also used to solve this regression problem. The code was originally developed by Aidan Chambers ([original repo](https://github.com/aidan-dc)). This code uses `Tensorflow` libraries instead of pytorch. To setup an environment on linux, one can use `fair_tf_cpu.yml` instead of `fair_cpu.yml` using the instruction mentioned above. In case one is using Mac M1 (ARM 64 Arch), `fair_tf_m1.yml` can be used to setup the environment.

To build and test CNN, one can use the notebook titled `09-CNN-CDMS-FAIR.yml`. This notebook contains instructions to process the dataset, bbuild CNN models and run the training.


## Support or Contact

If you need any help, please feel free to contact us!  

Bhargav Joshi (joshib@umn.edu)
