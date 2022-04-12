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

# CAN THE NETWORK LEARN THE 3X3 KERNEL??
# MAYBE I CAN DO THIS, USING MIN OR MAX POOLING?
# Doesn't seem like it because convolution layers always sum the multiplied values
# I need to either implement my own learning algorithm, or modify Conv2d so that it takes the median
# instead of a sum.
HEIGHT = 192
WIDTH = 384
BATCH_SIZE = 1
KERN_SIZE = 3
PADDING = 1
EPOCHS = 1000
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

class Depixelinator(nn.Module):
    def __init__(self):
        super(Depixelinator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(KERN_SIZE, KERN_SIZE), padding=(PADDING, PADDING))
        #self.conv2 = nn.Conv2d(1, 1, kernel_size=(KERN_SIZE, KERN_SIZE), padding=(PADDING, PADDING))
        self.epoch_losses = []

    def forward(self, input):
        # Add channel dimension?
        input = input.view(-1, 1, input.shape[1], input.shape[2])
        h1 = self.conv1(input)
        output = F.avg_pool2d(h1, kernel_size=(3,3), stride=(1,1), padding=((1,1)))
        return h1

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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0025)
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


def test_compare_shiftadd_deconvolver(model, device, test_data):
    # Set network to evaluation mode
    model.eval()

    N_VALID = len(test_data[0])

    # Convert the numpy arrays to Tensors
    inputs = torch.tensor(test_data[0], dtype=torch.float)
    targets = torch.tensor(test_data[1], dtype=torch.float)

    # Calculate different errors for the validation set
    # Convert the tensors to TensorDataSet
    test_combined = data_utils.TensorDataset(inputs, targets)
    # Get a batch-based iterator for the dataset
    test_loader = data_utils.DataLoader(test_combined, batch_size=BATCH_SIZE, shuffle=False)

    loss_L1 = 0
    loss_L12 = 0
    loss_MSE = 0
    loss_SSIM = 0
    loss_PSNR = 0
    data_counter = 0
    for input, target in test_loader:
        # Feed input to model, get output
        input = input.to(device)
        output = model(input, device)
        target_array = test_data[1][data_counter, :, :]
        output_array = np.squeeze(output.cpu().detach().numpy())

        loss_L1 += np.mean(np.abs(output_array - target_array))
        loss_L12 += np.mean(np.sqrt((output_array - target_array) ** 2 + 1) - 1)
        loss_MSE += np.mean((output_array - target_array) ** 2)
        loss_SSIM += ssim(output_array, target_array)
        loss_PSNR += cv2.PSNR(output_array, np.float32(target_array))

        data_counter += 1
    loss_L1 /= data_counter
    loss_L12 /= data_counter
    loss_MSE /= data_counter
    loss_SSIM /= data_counter
    loss_PSNR /= data_counter
    print("L1 loss: " + str(loss_L1))
    print("L1.5 loss: " + str(loss_L12))
    print("MSE loss: " + str(loss_MSE))
    print("SSIM index: " + str(loss_SSIM))
    print("PSNR: " + str(loss_PSNR))

    # Run the first 3 data instances through the network, and plot the results
    plt.figure()
    for i in range(0, 3):
        input = inputs[i][:][:]
        target = targets[i][:][:]

        # Copy data and target to right device and make the size correct
        input = input.to(device).view(1, 1, input.shape[0], input.shape[1])
        target = target.to(device).view(1, target.shape[0], target.shape[1])

        output = model(input, device)

        # Modify for visualization
        output = output.view(HEIGHT, WIDTH)
        target = target.view(HEIGHT, WIDTH)
        input = input.view(HEIGHT, WIDTH)

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        input = input.cpu().detach().numpy()

        # Make a strictly positive version of the output for logarithmic visualisation
        output_visual = output
        output_visual[output == 0] = 0.01

        plt.subplot(3, 3, 3 * i + 1)
        plt.imshow(input)
        plt.subplot(3, 3, 3 * i + 2)
        plt.imshow(output_visual, norm=LogNorm(vmin=0.01, vmax=20))
        plt.subplot(3, 3, 3 * i + 3)
        plt.imshow(target, norm=LogNorm(vmin=0.01, vmax=20))
    plt.show()


testimg_input = cv2.imread("test_image2_resized.png")
testimg_target = cv2.imread("test_image2_resized_blurred.png")
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
model = Depixelinator()
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

test_compare_shiftadd_deconvolver(model, device, valid_data)

# Plot errors
plt.figure()
plt.plot(list(range(1, len(train_losses)+1)), train_losses, label="Training losses")
plt.plot(list(range(1, len(val_losses)+1)), val_losses, label="Validation losses")
plt.legend()
plt.show()

