# cnn-fcc-pytorch
A learning project where we attempt to learn the PyTorch library. The purpose is to learn how to implement standard neural network models, such as a FCC and a CNN. 

# Model
The models are applied on the well known [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) dataset. The models are pretty minimalist and they don't provide outstanding results : this is not the purpose. The CNN model has a better performance. 

# Packages
```
numpy
```
```
torch
```
```
torchvision==0.3.0
```
```
matplotlib.pyplot
```
```
time
```

# Results
It tooke approximately 10m to train and test the CNN, with a GeForce 2060x GPU, for only 100 epochs.

With untuned hyperparameters we obtain a final precision on the validation set of approximately 90%. 

<img src="https://github.com/edhhan/cnn-fcc-pytorch/blob/main/results/acc_CNN.png" width="500" height="300">
<img src="https://github.com/edhhan/cnn-fcc-pytorch/blob/main/results/loss_CNN.png" width="500" height="300">



# Author
Edward H-Hannan
