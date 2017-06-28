from keras.applications.vgg19 import VGG19, decode_predictions as vgg19_decoder
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions as vgg16_decoder
from keras.applications.inception_v3 import InceptionV3,  decode_predictions as InceptionV3_decoder
from keras.applications.resnet50 import ResNet50, decode_predictions as ResNet50_decoder
from keras.models import Model
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import os, os.path
from tensorflow.contrib.tensorboard.plugins import projector
from sprite import sprite
from set_model import dnn_model

class feature_extract(dnn_model):
    def __init__(self):
        super(feature_extract, self).__init__()
        self.feature_tensor = None
        self._decode = False
        self.layer_name = None
        self.layer_index = None

    def select_layer(self, layer_na = None, layer_in = None):
        """selects layer using layer index or layer name and builds model

        Args:
            layer_na: layer name
            layer_in: layer index
        """

        if self.model_name is None:
            self.load_model()
        else:
            self.set_model()

        if layer_na is not None:
            self.model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_na).output)
        else:
            self.model = Model(inputs=self.model.input, outputs=self.model[layer_in].output)

    def _features(self, img_path):
        """extracts features from a given image

        Args:
            img_path: path to an image
        """
        img = image.load_img(img_path, target_size=self.im_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        self._predict(x, self.model_name)

class _embedding(feature_extract):
    """Creates embedding """

    def __init__(self):
        super(_embedding, self).__init__()

    def accumulate_tensor(self, pred_tens):
        """appends tensor outputs of a given layer for each image

         Args:
             pred_tens: tensor to be converted into embedding
         """
        self.prediction_tensor.append(pred_tens.tolist())

    def sprite_img(self, image_directory):
        """generates image sprite
        Args:
            image_directory: path to image directory

        """
        sp = sprite()
        sp.create_sprite(image_directory)

    def create_embedding(self, DATA_DIR, LOG_DIR = '', _sprite = False, _metadata = False, layer_nam = None, layer_ind = None):
        """create embedding and saves it to model.ckpt
        Args:
            DATA_DIR: path to directory of images
            LOG_DIR: path to a directory where log files are saved
            _sprite: If true adds sprite image into embedding
            _metadata: If true adds metadata into embedding
            layer_nam: name of a layer
            layer_ind: index of a layer
        """

        self.select_layer(layer_na = layer_nam, layer_in= layer_ind)
        self.num_pred = 1
        j = 0
        self._decode = False
        valid_images = [".jpg"]
        if _metadata == True:
            print 'Creating metadata...'
            self._decode = True
            metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
            metadata_file.write('Name\tClass\n')

        for f in os.listdir(DATA_DIR):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            print f
            self._features(os.path.join(DATA_DIR, f))
            self.prediction = np.squeeze(self.prediction.flatten())

            self.accumulate_tensor(self.prediction)
            if _metadata == True:
                j += 1
                metadata_file.write('%06d\t%s\n' % (j, self.class_label[0][0][1]))
        if _metadata == True:
            metadata_file.close()

        embedding_var = tf.Variable(np.squeeze(self.prediction_tensor), name='embedding')

        with tf.Session() as sess:
            # The embedding variable, which needs to be stored
            sess.run(embedding_var.initializer)
            summary_writer = tf.summary.FileWriter(LOG_DIR)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # attach metadata to embedding
            # Comment out if you dont have metadata
            if _metadata == True:
                embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')  # join log directory with the name of the meta data

            # attach sprite image to embedding
            # Comment out if you don't have sprite image
            if _sprite == True:
                embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')  # join log directory with the name of the sprite image
                embedding.sprite.single_image_dim.extend([263, 320])

            projector.visualize_embeddings(summary_writer, config)
            saver = tf.train.Saver([embedding_var])
            saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1)  # out put embedding check point