# crowdskin-bsc-2020 Thaomy Tran

> Hello, on this page you will find the codes and files that I have used for my Bachelor final project Medical Image Analysis (8Z423)
This research was build from the work of Raumanns et al. [see article](https://arxiv.org/pdf/2004.14745.pdf)

> The goal of my research was to compare different transfer learning models (VGG-16, VGG-19, InceptionV3 and ResNet50) to see what impact it has on the performance of multi-task learning  of skin lesions classifications
and to investigate which models may be combined to optimize the latter performance. 

> The baseline model is **without** annotated data
> The multi-task model is **with** annotated data

# Materials

-Models: [Baseline model](models/Baseline.ipynb)
-Models: [Multi-task model](models/Multitask model.ipynb)
In order to run the baseline and multi-task model, the codes in the ["models"](models) folder, [annotated data](annotated data) and [skin lesions dataset](skin lesion dataset/all_images.h5)  are needed
-Reports : [VGG-16](VGG16)
-Reports : [VGG19](VGG19)
-Reports : [InceptionV3](inceptionV3)
-Reports : [ResNet50](ResNet50)

files for visualisation can be found [here](Visualisation)

The codes and files to compare the different networks and features with the true data can be found [here](Codes)

All the wrong predicted labels compared to the true data for feature per network can be found [here](annotated data/wrong predicted labels.xlsx)

