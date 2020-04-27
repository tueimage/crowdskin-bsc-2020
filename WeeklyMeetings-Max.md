# Weekly meetings

Copy/paste and fill in the template below each week (most recent date on top, template at the bottom), commit and push your changes BEFORE coming to the weekly meeting with your supervisor.    

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