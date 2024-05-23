# Image-based discrimination of the early stages of mesenchymal stem cell differentiation

### The repository contains all the code necessary to reproduce the results in the paper
![image1](https://github.com/HoffmanOfGryffindor/msc-characterization/assets/68908581/896e3d9e-a7e1-416e-a5a5-a429ab4a8ba4 "Diagram")

## Running a model

### To run inference populate the data path in one of the config file:

```yaml
data:
  path: /path/to/imagery
```

### Once you have populated the path, choose which model you would like to run:

```yaml
model:
  name: autoencoder # names are small, mobilenet and autoencoder
  output: classification # choices are classification or regression - if autoencoder, this value does not matter
```

### Now that you have populated the config you can now run the following command to train a model:

```bash
python main.py --config /path/to/config.yaml
```
