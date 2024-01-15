import numpy as np
import yaml
from argparse import ArgumentParser
from models.utils import ModelTrainer
from models.cnn import CNNModel, MobileNet
from models.autoencoder import VariationalAutoencoder

MODELS = {
    "small": CNNModel,
    "mobilenet": MobileNet,
    "autoencoder": VariationalAutoencoder
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'rb') as file:
        config = yaml.safe_load(file)

    trainer = ModelTrainer(config)
    trainer.condition_data()
    
    model = MODELS.get(config["model"]["name"])(config=config, shape=trainer.shape)
    trainer.train(model)


