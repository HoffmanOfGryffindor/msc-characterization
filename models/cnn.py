import numpy as np
from tensorflow import keras

class CNNModel(keras.Model): 
    def __init__(self, num_classes: int = 5, filter_dim: tuple = (3,3), data_dim: tuple = (10,10), regression: bool = False):
        super(CNNModel, self).__init__()
        self.filter_dim = filter_dim
        self.data_dim = data_dim
        self.regression = regression
        if regression:
            self.final_layer_size, self.final_layer_activation = (1, 'linear')
        else:
            self.final_layer_size, self.final_layer_activation = (num_classes, 'softmax')
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, filter_dim, activation='relu', input_shape=(data_dim[1:])),
            keras.layers.MaxPooling2D((5, 5)),
            keras.layers.Conv2D(64, filter_dim, activation='relu'),
            keras.layers.MaxPooling2D((5, 5)),
            keras.layers.Conv2D(64, filter_dim, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.final_layer_size, activation=self.final_layer_activation)
        ])
        
    def call(self, x: np.array) -> np.array:
        model_data = self.model(x)  
        return model_data

class MobileNet(keras.model):
    def __init__(self, input_shape: list, num_classes: int  = 5):
        super(ResNet, self).__init__()
        self.mobile = keras.Sequential()
        self.backbone = keras.applications.mobilenet.MobileNet(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            classes=num_classes,

        )
        self.mobile.add(self.backbone)
        self.mobile.add(keras.layers.Flatten())
        self.mobile.add(keras.layers.Dense(512, activation='relu'))
        self.mobile.add(keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, x: np.array) -> np.array:
        model_data = self.mobile(x)  
        return model_data


class ResNet(keras.model):
    def __init__(self, input_shape: list, num_classes: int  = 5):
        super(ResNet, self).__init__()
        self.resnet = keras.Sequential()
        self.backbone = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            classes=num_classes,

        )
        self.resnet.add(self.backbone)
        self.resnet.add(keras.layers.Flatten())
        self.resnet.add(keras.layers.Dense(512, activation='relu'))
        self.resnet.add(keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, x: np.array) -> np.array:
        model_data = self.resnet(x)  
        return model_data
