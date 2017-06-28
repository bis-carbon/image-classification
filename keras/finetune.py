from keras.applications.vgg19 import VGG19, decode_predictions as vgg19_decoder
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions as vgg16_decoder
from keras.applications.inception_v3 import InceptionV3,  decode_predictions as InceptionV3_decoder
from keras.applications.resnet50 import ResNet50, decode_predictions as ResNet50_decoder
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Activation, BatchNormalization, Embedding
from keras.preprocessing.image import ImageDataGenerator
from set_model import dnn_model
import os
class fine_tune(dnn_model):
    """Re-trains last layer of a given model"""

    def __init__(self, train_data_dir = None, validation_data_dir = None, nb_epoch = 1 ):
        super(fine_tune, self).__init__()
        self.training_dir = train_data_dir
        self.validation_dir = validation_data_dir
        self.train_generator = None
        self.validation_generator = None
        self.nb_epoch = nb_epoch
        self.nb_classes = 1
        self.im_width = self.im_size[0]
        self.im_height = self.im_size[1]
        self.model = None

    def class_index(self, path = None):
        """writes class names in to .txt file. Will be used for decoding purpose during prediction.

        Args:
          Path: NxHxW[x3] tensor containing the images.
        """
        i=0
        dict = {}
        class_list = open(path, 'w')
        dir_lis = os.listdir(self.training_dir)
        dir_lis.sort()
        for directories in dir_lis:
            dict[i]=directories
            i += 1
        class_list.write(str(dict))
        class_list.close()


    def process_data(self):
        '''Process data in to training and validation'''

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=self.im_size,
            batch_size=self._batch_size,
            class_mode='categorical'
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.training_dir,
            target_size=self.im_size,
            batch_size=self._batch_size,
            class_mode='categorical'
        )
        self.nb_classes = self.train_generator.num_class
        self.sample_epoch = self.train_generator.samples
        self.nb_validation_samples = self.validation_generator.samples

    def design_layers(self):
        """Sets up the last layers of a given model and make the model ready for fine tuning
           Returns:
               predictions: returns predicted values for a given image

                        """
        if self.model_name == "vgg16":
            x = self.base_model.output
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.nb_classes, activation='sigmoid', name='prediction')(x)
            return predictions
        elif self.model_name == "vgg19":
            x = self.base_model.output
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.nb_classes, activation='sigmoid', name='prediction')(x)
            return predictions
        elif self.model_name == "InceptionV3":
            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(self.nb_classes, activation='softmax')(x)
            return predictions
        elif self.model_name == "ResNet50":
            x = self.base_model.output
            x = Flatten(name='flatten')(x)
            x = BatchNormalization(axis=1, name='batch_norm')(x)
            x = Dense(self.nb_classes, name='fc1000', init='lecun_uniform', W_regularizer=l2(0.01))(x)
            predictions = Activation("softmax", name='bfc1000')(x)
            return predictions
        else:
            '''Tobe: fine tune a user defined model'''
            return None


    def retrain(self, trn=None, labels=None,  val=None, val_labels=None, class_list_path = 'class_list.txt'):
        """Retrains the last layer of a given model

                Args:
                  trn: Training dataset
                  labels: Training dataset classes
                  val: validation dataset
                  val_labels: validation dataset labels
                  class_list_path: path to class_list.txt

                """

        if self.training_dir is not None:
            self.process_data()
            self.class_index(path=class_list_path)

        if self.model_name is None:
            self.load_model() #Tobe: remove the last layer of a user input model
            print "fine tuning user defind model is under construction"
            return None

        else:
            self.include_top_layer = False
            self.set_model()
            self.base_model = self.model

        for layer in self.base_model.layers:
            layer.trainable = False

        predictions = self.design_layers()

        self.model = Model(input=self.base_model.input, output=predictions)
        for layer in self.model.layers:
            print layer.name
            print layer.trainable

        # compile the model
        self._compile()

        # Train model
        if self.train_generator is not None:
            self._fit_gen(self.train_generator, self.validation_generator)
        else:
            self._fit(trn, labels,  val, val_labels)

        #save model
        self.save_model()

    def train(self):
        """Tobe: Trains whole model from scratch"""
        if self.model_path is not  None:
            self.load_model() # Tobe: initialize model to random weights
        else:
            #self.include_top_layer = True '''by default'''
            self.dataset = None
            self.set_model()

        for layer in self.model.layers:
            layer.trainable = True

        # compile the model
        self._compile()
        # Train model
        self._fit_gen(self.train_generator, self.validation_generator)
        # save model
        self.save_model()
