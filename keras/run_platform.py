from classify import classify
from finetune import fine_tune
from embeddings import feature_extract, _embedding
import numpy as np



'''runs classify'''


classifier = classify()
#classifier.model_weight= 'fine_tuned.h5' # Path to user defined model weight
#classifier.model_json = 'fine_tuned.json' # path to user defined model structure
classifier.model_name = 'ResNet50' # name of a model in keras module 'ResNet50', 'vgg16', 'vgg19' and 'InceptionV3'
classifier.pred('dog1.jpg') # image path or path to a directory of multiple images


'''runs fine tune '''
'''
re_train = fine_tune()
re_train.model_name = 'ResNet50' # model name to be fine tuned
re_train.training_dir = '/home/bisrat/flowers-data/raw-data/train/' # path to training directory
re_train.validation_dir = '/home/bisrat/flowers-data/raw-data/validation/' # path to validation directory
re_train.nb_epoch = 1 #number of epoch
re_train.retrain()
'''

'''runs embedding'''
'''
data_dir = '/home/bisrat/vacation_test/' # path to image directory
log_dir = '/home/bisrat/startupml/inception_V3TF/modelzoo/log' # path to save embeddings
embed = _embedding()
#embed.sprite_img(data_dir) # if you want to include sprite image uncomment
embed.model_name = 'ResNet50' # model name
embed.create_embedding(data_dir, LOG_DIR=log_dir, _sprite= True, _metadata=True, layer_nam='fc1000') #if you dont want to include metadata or sprite image set _sprtie and _metadata to False
'''
