from keras.applications.vgg19 import VGG19, decode_predictions as vgg19_decoder
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions as vgg16_decoder
from keras.applications.inception_v3 import InceptionV3,  decode_predictions as InceptionV3_decoder
from keras.applications.resnet50 import ResNet50, decode_predictions as ResNet50_decoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import os
from keras.preprocessing import image
from set_model import dnn_model
class classify(dnn_model):
    """Classifies images"""

    def __init__(self):
        super(classify, self).__init__()

    def pred(self, image_path):
        """classify image

        Args:
            image_path: path to image file or directory

        """

        if self.model_name is None:
            print "Loading weight..."
            #self.im_size = (224, 224)
            self.load_model()
            self._compile()
        else:
            self.set_model()
        im_path = []
        valid_images = [".jpg", ".png"]
        if os.path.isdir(image_path):
            for f in os.listdir(image_path):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                im_path.append(os.path.join(image_path, f))
        else:
            im_path.append(image_path)
        for ims in im_path:
            img = image.load_img(ims, target_size=self.im_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            self._predict(x, self.model_name)



