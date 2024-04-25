import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# from scikitplot.metrics import plot_confusion_matrix


class ModelTrainer:
    """ModelTrainer will digest the type of data you will need and train a model with segmented mesenchymal stem cells"""

    def __init__(self, config):
        """Initializes the ModelTrainer class
        :config: dictionary of key values for traininig
        """
        super().__init__()
        self.leaf_size = 14
        self.img = config["data"]["img"]
        self.data_path = config["data"]["path"]
        self.file_type = config["data"]["file_type"]
        self.model_name = config["model"]["name"]
        self.model_output = config["model"]["output"]
        self.epochs = config["model"]["epochs"]
        self.batch_size = config["model"]["batch_size"]
        self.label_map = config["labels"] if "labels" in config.keys() else None

        # creates the label encoder to transform labels from time
        if self.model_output == "classification":
            self.encoder = LabelEncoder()
        else:
            self.encoder = None

    def condition_data(self):
        """loads and processes data into a dictionary"""
        # generate data dictionary
        self.data = self._load_data()
        self._process_images()
        self.shape = self.data["x_train"].shape[1:]

    def _load_data(self):
        """load_data is the sub-function to condition_data that interprets the .npz files
        and generates a dictionary

        :return: dictionary containing data for x_train, y_train, x_test, y_test
        """
        data = {}
        for file in ["x_train", "y_train", "x_test", "y_test"]:
            data[file] = np.load(os.path.join(self.data_path, file + self.file_type))

            if file in ["y_train", "y_test"]:
                data[file] = self._string_to_int(data[file])

                if self.encoder:
                    data[file] = self.encoder.fit_transform(data[file])

        for x, y in [("x_train", "y_train"), ("x_test", "y_test")]:
            data[x], data[y] = shuffle(data[x], data[y])

        return data

    def _string_to_int(self, labels: np.ndarray) -> np.ndarray:
        """converts a string value to integer mapped from the config file

        :labels: np.ndarray containing raw values
        :return: np.ndarray with mapped values to discrete labels
        """

        for idx, label in enumerate(labels):
            labels[idx] = self.label_map[label]

        return labels

    def _process_images(self):
        """ indicate whether the image is using nucleus, actin or both based on the channel it is stored in
        """

        for x in ["x_train", "x_test"]:
            imgs = self.data[x]

            if self.img == "nucleus":
                imgs = imgs[:, :, :, 0]
            elif self.img == "actin":
                imgs = imgs[:, :, :, 1]
            elif self.img == "all":
                continue
            else:
                raise Exception(
                    'Not a recognizable image type, must be "nucleus", "actin" or "all"'
                )

            imgs = imgs[:, :, :, np.newaxis]

            if self.model_name == "autoencoder":
                self.data[x] = imgs
                continue

            imgs = np.concatenate((imgs, imgs, imgs), axis=3)
            self.data[x] = imgs

    def train(self, model):

        if self.model_name == "autoencoder":
            model.compile(optimizer=optimizers.Adam())

            full_data = np.concatenate(
                (self.data["x_train"], self.data["x_test"]), axis=0
            )
            model.fit(full_data, epochs=self.epochs, batch_size=self.batch_size)

        else:
            model.compile(
                optimizer=optimizers.Adam(),
                loss=(
                    "sparse_categorical_crossentropy"
                    if self.model_output == "classification"
                    else "mse"
                ),
                metrics=["accuracy"],
            )

            model = model.fit(
                self.data["x_train"],
                self.data["y_train"],
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.data["x_test"], self.data["y_test"]),
            )


# def plt_confusion_matrix(x, y, model, encoder, title, neural_network=True):
#     predict = model.predict(x)
#     if neural_network:
#         highest_pred = np.argmax(predict, axis=1)

#     cm = plot_confusion_matrix(encoder.inverse_transform(y), encoder.inverse_transform(highest_pred),title=title)
#     plt.show()
