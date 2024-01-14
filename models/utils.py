import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from scikitplot.metrics import plot_confusion_matrix


def plt_confusion_matrix(x, y, model, encoder, title, neural_network=True):
    predict = model.predict(x)
    if neural_network:
        highest_pred = np.argmax(predict, axis=1)

    cm = plot_confusion_matrix(encoder.inverse_transform(y), encoder.inverse_transform(highest_pred),title=title)
    plt.show()


class Data:
    def __init__(self, config):
        super(Data).__init__()
        self.img = config['data']['img']
        self.data_path = config['data']['path']
        self.file_type = config['data']['file_type']
        self.model_output = config['model']['output']

        if self.model_output == 'classfication': 
            self.encoder = LabelEncoder()
        else:
            self.encoder = None
        
        # generate data dictionary
        self.data = self._load_data()
        self._process_images()


    def _load_data(self):
        data = {}
        for file in ['x_train', 'y_train', 'x_test', 'y_test']:
            data[file] = np.load(os.path.join(self.data_path, file + self.file_type))
            
            if file == ['y_train', 'y_test']:
                data[file] = self._string_to_int(data[file])

            if self.encoder:
                data[file] = self.encoder.fit_transform(data[file])

        for x, y in [('x_train', 'y_train'), ('x_test', 'y_test')]:
            data[x], data[y] = shuffle(data[x], data[y])
        
        return data
    
    def _string_to_int(self, labels):
        for label in range(len(labels)):
            if labels[label] == "contrl" or labels[label] == "control":
                labels[label] = 0
                continue
            if labels[label] == '5d':
                labels[label] = 120
                continue
            else:
                labels[label] = int(labels[label].split('h')[0])
        return labels
    
    def _process_images(self):
        
        for x in ['x_train', 'x_test']:
            imgs = self.data[x]

            if self.img == 'nucleus':
                imgs = imgs[:,:,:,0]
            elif self.img == 'actin':
                imgs = imgs[:,:,:,1]
            elif self.img == 'all':
                continue
            else:
                raise Exception('Not a recognizable image type, must be "nucleus", "actin" or "all"')
             
            imgs = imgs[:,:,:, np.newaxis]
            imgs = np.concatenate((imgs, imgs, imgs), axis=3)
            
            self.data[x] = imgs

        
    

    
