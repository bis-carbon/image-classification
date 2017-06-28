# Inception-V3 implementation in TensorFlow
   by: Bisrat Zerihun

## Description of Code
This project includes the following functionalities:

 0: run_inception.py, This module is used to run any of the modules listed below. Set up the appropriate file paths and then uncomment the module you want to run.

 1: add_embedding.py, This module creates embedding from the out put of one of the layers of the pretrained model.
    The default layer used in this code is the "pool_3" layer. This module attaches labels and sprite image in to the embedding.
    In addition to creating embedding, this module also prints out prediction and save the predictions into html file.
    To get accurate word predicitons use the last layer "softmax" instade of "pool_3.
    For better feature extraction use "pool_3" layer.

    This module out puts Embedding. You can see the out put by running tensorboard as "tensorboard --logdir path/to/summary_dir "

 2: readtxt.py, This module is used to cluster images based on the out put of tensorboard T-SNE.
    After visualizing the embedding on tensorboard, select T-SNE clustered images and save them.
    The saved "state.txt" file contains list of image names which belong to the same class. Feed the path of this saved file to this module.

 3: sprite.py, this module creates sprite image.

 4: retrain.py, this module is used to fine tune the last layer of the pre-trained inception V3 model. The last layer is renamed into "final_result" after re-training.

    This module out puts the following:
      Scalars
      Graph
      Distributions
      Histograms

    Image data set feed to this module is structured as follows:

       MAIN_DIR:
          name_of_class1_DIR:
             images.....
          name_of_class2_DIR:
             images.....
          name_of_class3_DIR:
             images.....
          ...

 5: classify.py, run this module to make prediction after re-training the model on a given data set.


### For detailed understanding of inception-V3 please visit

     https://github.com/tensorflow/models/tree/master/inception
