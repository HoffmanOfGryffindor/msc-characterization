import numpy as np
from tensorflow import keras

class CNNModel(keras.Model): 
    def __init__(self, config, shape):
        super(CNNModel, self).__init__()
        self.data_dim = tuple(shape)
        self.filter_dim = tuple(config['model']['filter_dim'])
        self.regression = config['model']['output'] != "classification"
        self.num_classes = config['data']['num_classes']
        if self.regression:
            self.final_layer_size, self.final_layer_activation = (1, 'linear')
        else:
            self.final_layer_size, self.final_layer_activation = (self.num_classes, 'softmax')
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, self.filter_dim, activation='relu', input_shape=(self.data_dim)),
            keras.layers.MaxPooling2D((5, 5)),
            keras.layers.Conv2D(64, self.filter_dim, activation='relu'),
            keras.layers.MaxPooling2D((5, 5)),
            keras.layers.Conv2D(64, self.filter_dim, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.final_layer_size, activation=self.final_layer_activation)
        ])
        
    def call(self, x: np.array) -> np.array:
        model_data = self.model(x)  
        return model_data

class MobileNet(keras.Model):
    def __init__(self, config: dict, shape: list):
        super(MobileNet, self).__init__()
        self.mobile = keras.Sequential()
        self.backbone = keras.applications.mobilenet.MobileNet(
            include_top=False,
            weights='imagenet',
            input_shape=shape,
            classes=config['data']['num_classes'],

        )
        self.mobile.add(self.backbone)
        self.mobile.add(keras.layers.Flatten())
        self.mobile.add(keras.layers.Dense(512, activation='relu'))
        self.mobile.add(keras.layers.Dense(config['data']['num_classes'], activation='softmax'))

    def call(self, x: np.array) -> np.array:
        model_data = self.mobile(x)  
        return model_data


class ResNet(keras.Model):
    def __init__(self, config: dict, shape: list):
        super(ResNet, self).__init__()
        self.resnet = keras.Sequential()
        self.backbone = keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=shape,
            classes=config['data']['num_classes'],

        )
        self.resnet.add(self.backbone)
        self.resnet.add(keras.layers.Flatten())
        self.resnet.add(keras.layers.Dense(512, activation='relu'))
        self.resnet.add(keras.layers.Dense(config['data']['num_classes'], activation='softmax'))

    def call(self, x: np.array) -> np.array:
        model_data = self.resnet(x)  
        return model_data
