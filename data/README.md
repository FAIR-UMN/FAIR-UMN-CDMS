# Data Preprocessing

## Overview

This folder contains Python code/Jupyter notebooks for data preprocessing, including:


- **1_txt2csv.ipynb**: extract features and targets (positions) from the raw txt dataset.

- **2_prepare_dataset.ipynb**: 1) obtain a held-out subset; 2) split the rest to training/validation/test set.

- **3_features_analysis.ipynb**: visualize features before/after normalization.

*Note*: 

(1) To use these Jupyter notebooks, you should: first, place these 3 Jupyter notebooks in a folder named `data`; second, create a folder named `raw_txt` and put your raw dataset into it.

(2) When use these Jupyter notebooks, the suggested order should be: *1_txt2csv.ipynb* -> *2_prepare_dataset.ipynb* -> *3_features_analysis.ipynb*.  

(3) *3_features_analysis.ipynb* is mainly for the use of interests, it is not necessary to run it. However, you must run *1_txt2csv.ipynb* and *2_prepare_dataset.ipynb* correctly before moving to build neural network models.

(4) Each Jupyter notebook has detailed inline comments/descriptions for its important functions/steps.



## The Dataset 

In total, we have 13 different positions (targets) in our dataset,of which the details are summarized in Table 1. For each sample, we extract 19 informative features (see our [complementary document](https://www.overleaf.com/project/60db4e3f24f55f3c4f43e993) for details), including:

- **P[B,C,D,F]start**, the time at which the pulse rises to 20% of its peak with respect to the Channel A;

- **P[A,B,C,D,F]rise**, the time it takes for a pulse to rise from 20% to 50% of its peak;

- **P[A,B,C,D,F]width**, the width (in seconds) of the pulse at 80% of the pulse height;

- **P[A,B,C,D,F]fall**, the time it takes for a pulse to fall from 40% to 20% of its peak.



​															 Table 1: The positions/targets of our dataset and the corresponding sample sizes.

| Position | Sample number |
| :------: | :-----------: |
|   0.0    |      924      |
|  -3.969  |      500      |
|  -9.988  |      613      |
| -12.502  |      395      |
| -17.992  |      357      |
| -19.700  |      376      |
| -21.034  |      747      |
| -24.077  |      567      |
| -29.500  |      634      |
| -36.116  |      560      |
| -39.400  |      386      |
| -41.010  |      606      |
| -41.900  |      486      |



In order to see  how the machine learning models work on newly samples with never-seen positions, we divide the dataset into two subsets:

- *Held-out subset (HOS)*: it contains data samples whose positions are -12.502, -29.5, and -41.9; so it has 1515 data samples.
- *Model-learning subset (MLS)*: it contains the rest data samples; so it has 5636 data samples.

To obtain *HOS* and *MLS*, please see the details in *2_prepare_dataset.ipynb*.



*Note*: Here, as a demo example, we put those samples whose positions are {-12.502, -29.5, and -41.9} into our *HOS* as a newly test set. But you are encouraged to explore other settings (e.g., you may want to put less or more samples into *HOS*) and to do so, you only need to modify *2_prepare_dataset.ipynb*.



## The Problems

Given the dataset we have described above, the problems are:

- learning a neural network model with the given (features, positions/targets) pairs in *MLS*;

- testing the generality of the learned model on *HOS*. 

  

For more details about the problems and the neural network model we used, please check our [complementary document](https://github.com/ml-deepai/FAIR-UMN/blob/main/doc/FAIR%20Document%20-%20Identifying%20Interaction%20Location%20in%20SuperCDMS%20Detectors.pdf) and the Jupyter notebooks in [src](https://github.com/ml-deepai/FAIR-UMN/tree/main/src). 



