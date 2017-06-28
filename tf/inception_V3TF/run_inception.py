from add_embedding import _embedding
from retrain import retrain
from classify import run_inference_on_image
from sprite import create_sprite
from readtxt import read_imgname


image_dir="/home/bisrat/startupml/inception_bc/data" # path to image files
output_dir="/home/bisrat/startupml/inception_bc/retrainedmodel" # path to save retrained model
summary_dir="/home/bisrat/startupml/inception_bc/retrainedmodel"  # path to summary directory
model_dir = "/home/bisrat/startupml/tensorflow_models/inception/inception/model" # path to model
imagepath = "100500.jpg" # path to an image
CLUS_PATH = "/home/bisrat/startupml/inception_bc/data/class1/" # Path where clustered images are saved. Change the name "class1" accordingly for each cluster
TXT_PATH = "/home/bisrat/startupml/inception_bc/retrainedmodel/state.txt" # Path where .txt file from tensorboard is saved

'''Uncomment the function you want to use and leave commented the rest'''


'''create embedding using inception V3 pretrained model. This function also prints out predictions of the pre-trained model and saves the prediction in html format'''
_embedding(model_dir,image_dir, summary_dir)

''' Cluster images based on the out of Tensorboard T-SNE'''
#read_imgname(image_dir, CLUS_PATH, TXT_PATH)

'''create sprite image'''
#create_sprite(image_dir)

'''retrain model. "image_dir" should contain direcrories with names similar to class names. In each directory there should be images more than 20'''
#retrain(image_dir, output_dir, summary_dir, model_dir )

'''classify images after you re-train inception on a given data set'''
#run_inference_on_image(imagepath, output_dir)
