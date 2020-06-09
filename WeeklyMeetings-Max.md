# Weekly meetings

Copy/paste and fill in the template below each week (most recent date on top, template at the bottom), commit and push your changes BEFORE coming to the weekly meeting with your supervisor.    
### Date: 09/06/2020

#### What did you achieve this week?
* written materials and methods and part of results
* tried the other ensemble variant were the mean of all models is taken
* made visualisation of model architecture
* made final visualization of results
#### What did you struggle with?


#### What would you like to work on next week?
* Discussion
* make final code with a manual for the github repository

#### Where do you need help?
Where do i write things that have been tried but did not work out (adding more heads to the multitask model)? In the discussion?


#### Any other topic

#

### Date: 02/06/2020

#### What did you achieve this week?
* finished all model training and ensembling
* visualised results
* started visualising model architecture
* report writing
#### What did you struggle with?


#### What would you like to work on next week?
* finish materials and methods and results of the report
* make final code with a manual for the github repository

#### Where do you need help?



#### Any other topic

#

### Date: 26/05/2020

#### What did you achieve this week?
* implemented 3 extra models(resnet50v2, inceptionv3, efficientnetB1)
* made plans for ensembles (ensembling per annotation type or per model type)
* report writing
* implemented Gradient-CAM as a visualisation tool (what is the model looking at)
#### What did you struggle with?


#### What would you like to work on next week?
* finish ensembling of extra model models (ensemble of different models per annotation which are again ensembled)
* further report work on materials and methods and results
* maybe add more data (depending on goal of report)

#### Where do you need help?



#### Any other topic

#

### Date: 19/05/2020

#### What did you achieve this week?
* implemented ensemble learning
* implemented weighted ensemble learning (using random search optimization on validation set to optimize weights)
* implemented multitask learning using multiple classes(pairs of 2 and pairs of 3)
* experimented with different multitask models 
* experimented with adding dropout
#### What did you struggle with?
* Most models seem to overfit quite quickly (probably due to the size of the dataset)
* Multitask learning on multiple classes does not work well (one of the features overfits thus results are worse than
procedural classification)

#### What would you like to work on next week?
* continue writing report 
* Add more crowdsourced data
* Found that ensembling works best when combining different types of models, so i want to implement more types of models
that are added to the ensemble. (resnet, inception as a start)
* Make an ensemble of ensembles of different models if there is time left.

#### Where do you need help?



#### Any other topic

#

### Date: 12/05/2020

#### What did you achieve this week?
* Started on writing introduction
* replicated results for procedural classification and multitask classification
* Tried white balancing as a preprocessing step
* read literature on multitask and enemble learning
#### What did you struggle with?


#### What would you like to work on next week?
* continue writing introduction 
* make code for ensemble model
* try different combinations of multitask models (combinations of features)
* Implement code to load annotations that are not yet used

#### Where do you need help?
* Class weights do now work using keras but i have seen some strange behaviour for the multitask model:
    'conv_base.trainable = False' should result in only the last dense layer being trainable. 
    However with the provided code the whole model is trainable. The results produced by the model are in line
    with results from the paper.



#### Any other topic

#

### Date: 05/05/2020

#### What did you achieve this week?
* Read literature
* Started with mockup of report on overleaf
* fixed saving of smaller dataset 
* replicated results for procedural classification
* Tried implementing class weight by multiplying with sample weights (not available in keras for multiclass models)
* Tried changing image preprocessing by using the albumentations library
* Made script to start multiple colab notebooks in parallel
* Made script for visualising results
#### What did you struggle with?

* have not yet been able to replicate results for the multi task model (Probably due to class_weights not working)
* multi task model seems to overfit

#### What would you like to work on next week?
* start writing introduction and problem statement 
* replicate multitask results
* Writing code for multitask model with more features at once
* Implement code to load annotations that are not yet used
* If there is time left: write code for ensemble model

#### Where do you need help?
* Is class_weights critical for reaching the test auc scores in the multitask model?


#### Any other topic

#

### Date: 28/04/2020

#### What did you achieve this week?
* Read literature

* Got experiment working on Google colab
* Added tensorboard
* Tested multiprocessing when importing data
* Some testing with TPU's
* Profiled current model to look for improvements
* Loaded dataset to ram on google colab (because of slow disk access)
* Made alternative data generators using keras Sequence
* made new dataset which is already resized saved in hdf5 format
    * These changes reduced training time per epoch to around 20s for procedural and 90s for multitask

#### What did you struggle with?
* A bug in tensorflow 2.1 which stopped the multitask model from training
* Difference between tf.keras and keras

#### What would you like to work on next week?
* Several options:
    * look at combining three models trained on ABC features as an ensamble
    * does increasing size of dataset give a better result
    * improving model itself (different model, adding regularisation)
    * training on crowd sourced features and fine tune on diagnosis (from paper)

#### Where do you need help?
* Did not understand why there are 100 batches (so 2000 images) in one epoch when the training set is only
1400 images large. Same applies to validation set.
* using tf.keras results in error: 
ValueError: `class_weight` is only supported for Models with a single output.
when using procedural_multitask (maybe i could just use keras?)

#### Any other topic
* What are the deliverables? (report presentation code ect.)

#
### Date: 11/06/2018

#### What did you achieve this week?
* Did an experiment
* Wrote a section of report

#### What did you struggle with?
* 

#### What would you like to work on next week?
* Extend the experiment to dataset X

#### Where do you need help?
* Not sure how to do Y


#### Any other topic
* Another question