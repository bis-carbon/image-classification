# ModelZoo implementation in Keras
   Code by: Bisrat Zerihun

## Description of Code
This project works for vgg16, vgg19, ResNet50 and InceotionV3 models. It includes the following modules:

 0: run_platform.py, This module is used to run any of the modules listed below. Set up the appropriate file paths and then uncomment the module you want to run.

 1: embeddings.py, This module creates embedding from the any given layer of a model.
    This module also attaches labels and sprite image in to the embedding.

    You can vizualize the out put by running tensorboar as "tensorboard --logdir /path/to/summary_dir "

 2: readtxt.py, This module is used to cluster images based on the out put of tensorboard T-SNE.
    After visualizing the embedding on tensorboard, select T-SNE clustered images and save them.
    The saved "state.txt" file contains list of image names which belong to same class. Feed the path of this saved file to this module.

 3: sprite.py, this module creates sprite image.

 4: finetune.py, this module is used to fine tune the last layer of a pre-trained model. 

    Image data set feed to this module is structured as follows:

       MAIN_DIR:
          name_of_class1_DIR:
             images.....
          name_of_class2_DIR:
             images.....
          name_of_class3_DIR:
             images.....
          ...

 5: classify.py, run this module to make prediction using user specified model.
 
 6: generic_fun.py, This is a base class which implements fit, compile and predict.
 
 7: set_model.py, This is a base class which is used to import and set up a user specified model.
