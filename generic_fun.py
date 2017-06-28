from keras.applications.vgg19 import VGG19, decode_predictions as vgg19_decoder
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions as vgg16_decoder
from keras.applications.inception_v3 import InceptionV3,  decode_predictions as InceptionV3_decoder
from keras.applications.resnet50 import ResNet50, decode_predictions as ResNet50_decoder
import json
import numpy as np
from keras import optimizers


class _generic(object):
    """implements fit, compile, fit_generator, predict, decode"""
    def __init__(self):
        self.history = None
        self.prediction = np.array([])
        self.prediction_tensor = []
        self.nb_validation_samples = 10
        self.nb_epoch = 50
        self.base_model = None
        self.model = None
        self.layer_feature = None
        self._dnn = None
        self.sample_epoch = 128
        self._batch_size = 16
        self.model_path = None
        self._decode = True
        self.model_json = 'fine_tuned.json'
        self.model_weight = 'fine_tuned.h5'
        self.num_pred = 3
        self.class_label = None
        self.class_list_txt = None


    def _fit_gen(self, train_generator, validation_generator):
        """Train a model
        Args:
            train_generator: Processed training data
            validation_generator: Processed validation data

        """
        self.history = self.model.fit_generator(train_generator,
                                                nb_epoch=self.nb_epoch,
                                                samples_per_epoch=self.sample_epoch,
                                                validation_data=validation_generator,
                                                nb_val_samples=self.nb_validation_samples)

    def _predict(self, x, model_name = None, class_list_path ='class_list.txt'):
        """predict based on user specified model
        Args:
            x:processed image tensor
            model_name: name of a model
            class_list_path: path to class_list.txt

        Returns:
            prints prediction result

        """
        self.prediction = self.model.predict(x)

        if self._decode == True:
            """Decode prediction in to classes"""
            if model_name == "vgg16":
                self.class_label = vgg16_decoder(self.prediction, top=self.num_pred )
                print('Predicted:', self.class_label)
            elif model_name == "vgg19":
                self.class_label = vgg19_decoder(self.prediction, top=self.num_pred )
                print('Predicted:', self.class_label )
            elif model_name == "InceptionV3":
                self.class_label = InceptionV3_decoder(self.prediction, top=self.num_pred )
                print('Predicted:', self.class_label )
            elif model_name == "ResNet50":
                self.class_label = ResNet50_decoder(self.prediction, top=self.num_pred )
                print('Predicted:',self.class_label )
            else:
                print 'class index : ', self.prediction.argmax(axis=-1)
                self.decode_class(ind=self.prediction.argmax(axis=-1), class_list_path=class_list_path)


    def _fit(self, trn, labels,  val, val_labels):
        """Train a model

                       Args:
                         trn: Training dataset
                         labels: Training dataset classes
                         val: validation dataset
                         val_labels: validation dataset labels

                       """

        '''Tobe: tested'''
        self.model.fit(trn, labels, nb_epoch=self.nb_epoch, validation_data=(val, val_labels), batch_size=self._batch_size)

    def _compile(self):
        """Compile a model"""
        self.model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    def decode_class(self, ind = None, class_list_path = None ):
        """Decodes predictions in to classes
        Args:
            ind:prediction index
            class_list_path: path to a file with list of classes
        Returns:
            Prints classes after decoding.

        """
        try:
            text_file = (open(class_list_path, "r"))
            lines = eval(str(text_file.readlines()))
            lines = lines[0]
            lines = eval(lines)
            print lines[(ind[0])]
        except:
            return None


