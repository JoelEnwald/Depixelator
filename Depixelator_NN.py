import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from skimage.transform import radon, rescale, iradon
from skimage.metrics import structural_similarity as ssim
# Memory usage monitoring
import tracemalloc
from matplotlib.colors import LogNorm
import cv2
from torch.nn.modules.utils import _pair, _quadruple

# An attempt at training a neural network to learn the pixel mappings I wanted.
# It didn't work as I wanted so it was ditched.


# I think I need to take 3x3 and 5x5 medians of the target image,
# and then let the NN combine the three images as it sees fit.

HEIGHT = 192
WIDTH = 384
BATCH_SIZE = 1
KERN_SIZE = 3
PADDING = (KERN_SIZE-1)//2
EPOCHS = 200
VISUALIZE = False

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class DepixelinatorSmall(nn.Module):
    def __init__(self):
        super(DepixelinatorSmall, self).__init__()
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=(1,1), padding=0)
        self.conv3x3 = nn.Conv2d(1, 1, kernel_size=(3,3), padding=1)
        self.conv5x5 = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2)
        self.meanpool3x3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.meanpool5x5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.medianpool3x3 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        self.medianpool5x5 = MedianPool2d(kernel_size=5, stride=1, padding=2)
        self.medianpool7x7 = MedianPool2d(kernel_size=7, stride=1, padding=3)
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool5x5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.convimsize = nn.Conv2d(1, 1, kernel_size=(192, 384), padding=0)
        self.depthconv = nn.Conv2d(3, 1, kernel_size=(1, 1), padding=(0, 0))
        self.epoch_losses = []

    def forward(self, input):
        # Add channel dimension?
        input = input.view(-1, 1, input.shape[1], input.shape[2])
        #conv1x1 = self.conv1x1(input)
        #conv3x3 = self.conv3x3(input)
        #conv5x5 = self.conv5x5(input)
        #means3x3 = self.meanpool3x3(input)
        #means5x5 = self.meanpool5x5(input)
        medians3x3 = self.medianpool3x3(input)
        medians5x5 = self.medianpool5x5(input)
        #medians7x7 = self.medianpool7x7(input)
        #maxes3x3 = self.maxpool3x3(input)
        #maxes5x5 = self.maxpool5x5(input)
        #convimage = torch.multiply(torch.ones((192, 384), device='cuda'), self.convimsize(input))
        #input_combined = torch.cat((input, conv1x1, conv3x3, conv5x5, means3x3, means5x5, medians3x3, medians5x5, medians7x7, maxes3x3, maxes5x5), dim=1)
        input_combined = torch.cat((input, medians3x3, medians5x5), dim=1)
        output = self.depthconv(input_combined)
        return output

    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (input, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            input = input.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])

            # Average loss in one batch of data.
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            batch_loss = loss_fun(output, target)
            # Make epoch loss into something that doesn't save the computational graph
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.conv1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. This doesn't actually weigh all data instances evenly,
        # due to the last epoch not necessarily being full-sized. So the average loss is not quite accurate!
        epoch_loss = epoch_loss / (np.ceil(len(inputs.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = 0
        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        data_counter = 0
        for input, target in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])
            # Calculate loss
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            # Make the loss into something that doesn't save the computational graph
            val_loss += float(loss_fun(output, target))

            # Write down the outputs
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output.cpu().detach().numpy()
            data_counter += 1

        val_loss = float(val_loss) / len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs

class LookupLearner(nn.Module):
    def __init__(self):
        super(LookupLearner, self).__init__()
        self.n_filters = 500
        self.conv5x5 = nn.Conv2d(1, self.n_filters, kernel_size=(5,5), padding=2, bias=False)
        self.outputvals = torch.nn.Parameter(torch.rand(size=(self.n_filters,)))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.epoch_losses = []

    def forward(self, input):
        # Add channel dimension?
        input = input.view(-1, 1, input.shape[1], input.shape[2])
        # Calculate the correlation coefficient between the filters and the image patches
        # Currently not numerically stable.
        Exy = torch.div(self.conv5x5(input), 25)
        Ex = self.avgpool(input)
        Ey = torch.mean(self.conv5x5.weight, dim=(2,3))
        ExEy = torch.mul(Ex, Ey)

        # Get the weights for each output value
        t2 = torch.softmax(t1, dim=1)
        t3 = self.outputvals.repeat((1, HEIGHT, WIDTH, 1))
        t3 = torch.transpose(t3, dim0=2, dim1=3)
        t3 = torch.transpose(t3, dim0=1, dim1=2)
        # Multiply the weights by the output values
        output = torch.sum(torch.multiply(t2, t3), dim=1)
        # Add channel dimension?
        output = output.view(1, 1, output.shape[1], output.shape[2])

        # Get the index of the maximum value
        #ind_max = torch.argmax(t1, dim=1, keepdim=True)
        # Pick the corresponding output value from outputvals
        #output = self.outputvals[ind_max]
        
        return output

    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE)

        optimizer = torch.optim.SGD(self.parameters(), lr=0.2, momentum=0.9)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (input, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            input = input.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])

            # Average loss in one batch of data.
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            batch_loss = loss_fun(output, target)
            # Make epoch loss into something that doesn't save the computational graph
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.conv1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. This doesn't actually weigh all data instances evenly,
        # due to the last epoch not necessarily being full-sized. So the average loss is not quite accurate!
        epoch_loss = epoch_loss / (np.ceil(len(inputs.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = 0
        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        data_counter = 0
        for input, target in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])
            # Calculate loss
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            # Make the loss into something that doesn't save the computational graph
            val_loss += float(loss_fun(output, target))

            # Write down the outputs
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output.cpu().detach().numpy()
            data_counter += 1

        val_loss = float(val_loss) / len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs

class DepixelinatorBig(nn.Module):
    def __init__(self):
        super(DepixelinatorBig, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(7, 25, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(33, 125, kernel_size=(3,3), padding=1)
        self.median1 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        self.median2 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        self.median3 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        self.depthconv = nn.Conv2d(159, 1, kernel_size=(1, 1), padding=(0, 0))
        self.epoch_losses = []

    def forward(self, input):
        # Add channel dimension?
        input = input.view(-1, 1, input.shape[1], input.shape[2])
        #conv1x1 = self.conv1x1(input)
        #conv3x3 = self.conv3x3(input)
        hallo1 = F.relu(self.conv1(input))
        hallo2 = self.median1(input)
        hidden1 = torch.cat((F.relu(self.conv1(input)), self.median1(input), input), dim=1)
        hidden2 = torch.cat((F.relu(self.conv2(hidden1)), self.median2(hidden1), input), dim=1)
        hidden3 = torch.cat((F.relu(self.conv3(hidden2)), self.median3(hidden2), input), dim=1)
        output = F.relu(self.depthconv(hidden3))
        return output

    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (input, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            input = input.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])

            # Average loss in one batch of data.
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            batch_loss = loss_fun(output, target)
            # Make epoch loss into something that doesn't save the computational graph
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.conv1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. This doesn't actually weigh all data instances evenly,
        # due to the last epoch not necessarily being full-sized. So the average loss is not quite accurate!
        epoch_loss = epoch_loss / (np.ceil(len(inputs.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = 0
        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        data_counter = 0
        for input, target in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = self(input)
            output = output.view(1, output.shape[2], output.shape[3])
            # Calculate loss
            #loss_fun = nn.L1Loss()
            loss_fun = nn.MSELoss()
            # Make the loss into something that doesn't save the computational graph
            val_loss += float(loss_fun(output, target))

            # Write down the outputs
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output.cpu().detach().numpy()
            data_counter += 1

        val_loss = float(val_loss) / len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs


testimg_input = cv2.imread("test_image2_easy_resized.png")
testimg_target = cv2.imread("test_image2_easy_target.png")
testimg_input = cv2.cvtColor(testimg_input, cv2.COLOR_BGR2GRAY) / 255
testimg_target = cv2.cvtColor(testimg_target, cv2.COLOR_BGR2GRAY) / 255

x_train = np.reshape(testimg_input, (1, testimg_input.shape[0], testimg_input.shape[1]))
x_valid = np.reshape(testimg_input, (1, testimg_input.shape[0], testimg_input.shape[1]))
y_train = np.reshape(testimg_target, (1, testimg_target.shape[0], testimg_target.shape[1]))
y_valid = np.reshape(testimg_target, (1, testimg_target.shape[0], testimg_target.shape[1]))

# Put the x and y parts together as training and validation data
train_data = list((x_train, y_train)); valid_data = list((x_valid, y_valid))


if torch.cuda.is_available():
    print('Using GPU!')
    device = torch.device('cuda')
else:
    print('Using CPU!')
    device = torch.device('cpu')

# Create a new network
model = LookupLearner()
model = model.to(device)

val_losses = []
accuracies = []
for epoch in range(1, EPOCHS + 1):
    train_losses = model.train_with_data(device, train_data, epoch=epoch, log_interval=10)
    val_loss, outputs = model.validate(device, valid_data)
    val_losses.append(val_loss)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(outputs.reshape((outputs.shape[1], outputs.shape[2])))
plt.subplot(1, 2, 2)
plt.imshow(testimg_target)
plt.show()

# Plot errors
plt.figure()
plt.semilogy()
plt.plot(list(range(1, len(train_losses)+1)), train_losses, label="Training losses")
plt.plot(list(range(1, len(val_losses)+1)), val_losses, label="Validation losses")
plt.legend()
plt.show()

