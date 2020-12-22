import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable

from CNN_class import CNN
from FCC_class import FCC



# FashionMNIST dataset
data_MINST = datasets.FashionMNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

train_data, val_data = torch.utils.data.random_split(data_MINST, [50000, 10000])

batch_size = 100
test_batch_size = 100

# Training set
train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size, shuffle=True)

# Validation set
validation_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=batch_size, shuffle=True)

# Training
def train(nn_model, train_loader, optimizer, GPU):
    """
    :param model: nn model defined in a X_class.py
    :param train_load: data format from torchvision
    :param GPU: boolean variable that initialize some variable on the GPU if accessible, otherwise on CPU
    """
    nn_model.train()
    loss_training = 0

    for train_index, (data, target) in enumerate(train_loader):

        if not GPU:
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()

        optimizer.zero_grad()
        out_values = nn_model(data)
        loss = F.nll_loss(out_values, target)
        loss_training = loss_training + loss.item()
        loss.backward()
        optimizer.step()

    return nn_model


# Accuracy
def get_accuracy(nn_model, loader, GPU):
    nn_model.eval()
    loss_validation = 0
    nb_correct = 0

    for data, target in loader:
        if not GPU:
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()

        out_values = nn_model(data)
        loss_validation += F.nll_loss(out_values, target, size_average=False).item()
        prediction = out_values.data.max(1, keepdim=True)[1]
        nb_correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    return nb_correct.item() * 100 / len(loader.dataset), loss_validation/len(loader.dataset)


# Testing function
def testing(nn_model, nb_epoch, lr, GPU=False):
    best_precision = 0
    optimizer = optim.Adam(nn_model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    training_losses = []
    accuracies = []
    validation_losses = []

    for epoch in range(1, nb_epoch + 1):

        # Training
        nn_model = train(nn_model, train_loader, optimizer, GPU)

        # Accuracy on training set
        train_precision, loss_training = get_accuracy(nn_model, train_loader, GPU)
        training_losses.append(loss_training)

        # Accuracy on validation set
        precision, loss_validation = get_accuracy(nn_model, validation_loader, GPU)
        validation_losses.append(loss_validation)
        accuracies.append(precision)
        if precision > best_precision:
            best_precision = precision
            best_model = nn_model

        # Scheduler
        scheduler.step((loss_validation))


    plt.figure()
    plt.plot(range(1, nb_epoch+1), training_losses, '--b', label='Training')
    plt.plot(range(1, nb_epoch+1), validation_losses, '--r', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Negative log likelihood')

    plt.legend()

    plt.figure()
    plt.plot(range(1, nb_epoch+1), accuracies)
    plt.ylabel('Precision (%)')
    plt.xlabel('Epoch')

    return best_model, best_precision


########
# Main #
########

#Hyperparameters
lr = 0.0001  # Note : scheduler implemented in testing() function
nb_epoch = 100
GPU = True

#-Model loop
best_precision = 0
losses = []

#for model in [FCC(), CNN()]:
for model in [CNN()]:

    if GPU:
        model.cuda()
    
    t = time.time()

    # Training, etc.
    model, precision = testing(model, nb_epoch, lr, GPU)

    # Final precision on trained model with validation dataset
    print('Precision Finale: {}'.format(get_accuracy(model, validation_loader, GPU)))

    elapsed = time.time() - t 

    print('temps dapprentissage : {}'.format(elapsed))

plt.show()
