from keras.applications.vgg19 import VGG19, decode_predictions as vgg19_decoder
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions as vgg16_decoder
from keras.applications.inception_v3 import InceptionV3,  decode_predictions as InceptionV3_decoder
from keras.applications.resnet50 import ResNet50, decode_predictions as ResNet50_decoder
from keras.models import model_from_json
from generic_fun import _generic
class dnn_model(_generic):

    def __init__(self, model_name=None, dataset="imagenet"):
        super(dnn_model, self).__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.im_size = (224, 224)
        self.include_top_layer = True
        self.nb_channels = 3

    def set_model(self):
        """select and load a model"""
        if self.model_name == 'vgg16':
            self.model = VGG16(weights=self.dataset, include_top=self.include_top_layer, input_shape = (self.im_size[0], self.im_size[1], self.nb_channels))
        elif self.model_name == 'vgg19':
            self.model = VGG19(weights=self.dataset, include_top=self.include_top_layer, input_shape = (self.im_size[0], self.im_size[1], self.nb_channels))
        elif self.model_name == 'InceptionV3':
            self.im_size = (299, 299)
            self.model = InceptionV3(weights=self.dataset, include_top=self.include_top_layer, input_shape=(self.im_size[0], self.im_size[1], self.nb_channels))
        elif self.model_name == 'ResNet50':
            self.model = ResNet50(weights=self.dataset, include_top=self.include_top_layer,  input_shape = (self.im_size[0], self.im_size[1], self.nb_channels))


    def save_model(self, model_path=''):
        """saves model to a given path
        Args:
            model_path: path to a given model.

        """
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path+ self.model_json, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_path+self.model_weight)
        print("Model Saved to disk")

    def load_model(self, weight = True):
        """Loads model from a given path
        Args:
            weight: if true loads weight into a given model
        """

        # load json and create model
        json_file = open(self.model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        if weight == True:
            # load weights into new model
            self.model.load_weights(self.model_weight)
            print(" Model Loaded from disk")

