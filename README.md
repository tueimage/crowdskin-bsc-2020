# crowdskin-bsc-2020 Max Joosten
This repository contains the code used for my bachelor end project.
The goal of this project was to expand on the work of Raumanns et al. (https://arxiv.org/abs/2004.14745) by training more deep convolutional 
models and ensembling them. 

The dataset that was used was the isic 2017 image challenge dataset with 2000 images. 70% 
of the dataset was used as a training set (1400 images), 17.5% of  the  dataset  was  used  as  a  validation  set  
(350  images) and 12.5% of the dataset was used as a test set (250 images).


### folder structure

├───data: contains excel files with annotations  
├───models: contains all model and preprocessing scripts  
│   ├───5_procedural_classification: contains colab notebooks for training baseline model  
│   ├───5_procedural_multitask: contains colab notebooks for training vgg16      
│   ├───5_procedural_multitask_efficientnet: contains colab notebooks for training EfficientNetB1   
│   ├───5_procedural_multitask_inception: contains colab notebooks for training baseline InceptionV3    
│   ├───5_procedural_multitask_resnet: contains colab notebooks for training ResNetV2    
│     
├───reports: contains csv files with auc for every combination of seed, model and ensemble   
├───Visualisation and misc: containing code for visualisation and experimental code to open and start 5 colab tabs   
└───weights: folder containing h5 weight and json model files used for ensemble predictions

### Manual for running code
The project is intended to run locally for some files and on google colab for other files. To start copy over the 
repository to google drive to make it available to google colab. This folder was synchronised with a computer 
where the local code would be run. Download the dataset to your local computer and resize it using lean_dataset.py. 
change IMAGE_DATA_PATH to the folder conataining the ISIC images. A h5 file will be generated with the resized images 
in the same folder. Upload this file to somewhere in your google drive.

Experiments are run using the files in the 5_procedural... folder. Each folder contains 5 variants of the same model 
type with different seeds (for crossvalidation). These files are colab notebooks where 1 notebook trains 1 seed of 1 
model. It is also possible to run these notebooks on the local computer.
(when a powerful enough GPU is available (colab pro uses NVIDIA Tesla P100 GPU's with 16GB of memory)). 
The IMAGE_DATA_PATH needs to be set to the folder containing the h5 file.

The script first loads the annotation data and image file list (divided into asymmetry, border and color) 
using the get_data script. The generate_data.py script generates augmented batches of batch size BATCH_SIZE used for 
training and testing the model. The flag GENERATE_ALTERNATIVE == True was used (GENERATE_ALTERNATIVE == True uses 
experimental image generators not used in generating the results) These batches contain the images and annotation data. 
The model is then built, fitted and tested on the test dataset. The AUC is then reported in reports as a csv. after 
this the model is saved as h5 weight files and json model files, in the folder weights. 

After this the ensemble.ipynb is run in google colab to generate the ensembles. The last line of code in this notebook
defines which ensemble function is used. the functions auc_score_ensemble_multi_ABC() and auc_score_ensemble_multi_model()
were used for model and feature ensembles respectively. Results are saved in the reports folder. Grad-cams are made 
during the model predictions.

Visualisations were made using visualisation.py on the local computer. plot_aucs() generated boxplots based on the 
names_of_runs list. this list contains the names of the report files before _aucs in the filename. plot_acc_loss()
can plot accuracy and loss curves for the training process, however it is better to use tensorboard in the colab notebooks
for this.
