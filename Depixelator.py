

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import LogNorm
import cv2
import random
import sklearn_extra.cluster._k_medoids as KMedoids
from scipy import stats
from scipy import spatial
import os
import time


# 'Smart' image plotting function. Random but deterministic.
# Sets the limits so that roughly 95% of the data is between vmin and vmax
def smartimshow(img, cmap=None):
    # sample 200 values from the image, deterministically
    random.seed(1234321)
    xs = np.random.choice(np.arange(0, len(img)), size=200, replace=True)
    ys = np.random.choice(np.arange(0, len(img[0])), size=200, replace=True)
    img_sampled = img[xs, ys]
    img_sampled = np.sort(img_sampled)
    vmin = img_sampled[5]
    vmax = img_sampled[195]
    plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

# 'img' is the input image
# 'imgtype' defines whether the image is grayscale or color. The process differs for the two types,
# though they could probably be combined. Pick 'RGB' or 'GRAY'.
# 'amount' chooses the amount of smoothing. Pick 'low' or 'high'.
# 'bias' is used to make the process work correctly for images that are mainly
# dark forms on a light background, or light forms on a dark background. Bias of 0 corresponds to
# dark on light (default), and 1 to light on dark.
def depixelate(img, imgtype, amount, bias=0):
    # Method for finding the biased medoid of an image patch (for color pixels)
    # Can be biased towards dark or light
    def biasedmedoid_RGB(impatch, bias=None):
        # Make the central value (0,0,0), to bias the median towards dark
        values = np.copy(impatch)
        if bias == 0:
            values[1, 1, :] = np.zeros(3)
        elif bias == 1:
            values[1,1,:] = np.ones(3)
        # Find the value that minimizes the average L1 distance to all the other values (including the central 0 value)
        values = values.reshape((values.shape[0] * values.shape[1], 3))
        distmat = spatial.distance_matrix(values, values, p=1)
        distsums = np.sum(distmat, axis=0)
        # Find the index of the pixel with the minimum distance
        minind = np.argmin(distsums)
        # Get the new value, convert the index to 3x3 coords
        new_value = impatch[int(np.floor(minind / 3)), (minind % 3), :]
        return new_value

    # Method for finding a medoid within given values
    def medoid_RGB(values):
        # Find the value that minimizes the average L1 distance to all the other values (including the central 0 value)
        distmat = spatial.distance_matrix(values, values, p=1)
        distsums = np.sum(distmat, axis=0)
        # Find the index of the pixel with the minimum distance
        minind = np.argmin(distsums)
        # Get the new value, convert the index to 3x3 coords
        new_value = values[minind]
        return new_value

    def LUTInterpGRAY(img_test, lookuptables, lookupvalues, amount, bias=0):
        height = img_test.shape[0]
        width = img_test.shape[1]
        new_img = np.ones(img_test.shape)
        # Start processing from where the filter can fully overlap with the image
        for i in range(3, height - 3):
            print("Processing row " + str(i))
            for j in range(3, width - 3):
                impatch = img_test[i - 3:i + 3 + 1, j - 3:j + 3 + 1]
                # if np.std(impatch) > 0.3:
                #    hallo = 5
                # Check if impatch contains only one value
                if np.std(impatch) < 1e-9:
                    new_value = impatch[0][0]
                # Never change the central pixel of a 3x3 megapixel
                elif i % 3 == 1 and j % 3 == 1:
                    new_value = impatch[(7 - 1) // 2, (7 - 1) // 2]
                # if we are looking at one of the corners of a 3x3 area, use the biased median
                elif (i % 3 == 0 or i % 3 == 2) and (j % 3 == 0 or j % 3 == 2):
                    values = img_test[i - 1:i + 2, j - 1:j + 2].flatten()
                    # Make the central value small or big, to bias the values
                    # (the original central value should not affect the result)
                    if bias == 0:
                        values[4] = 0
                    elif bias == 1:
                        values[4] = 1
                    # Take the biased median
                    new_value = np.sort(values)[4]
                # If we are looking at the edges of a superpixel, use LUT's on high smoothing setting,
                # and biased median on low smoothing setting.
                else:
                    if amount == 'high':
                        # Divide the impatch values into 9 different groups. Then try all possible 2-cluster clusterings on them
                        impatchgroupreps = np.zeros((3, 3))
                        for k in range(0, 3):
                            for l in range(0, 3):
                                impatchgroupreps[k, l] = impatch[3 * k, 3 * l]
                        impatchgroupreps = impatchgroupreps.flatten()
                        # Remove duplicates from the different values. Also sorts them.
                        impatchgroupreps = np.unique(impatchgroupreps)
                        impatchgroupreps = np.sort(impatchgroupreps)
                        minclusterloss = 1e9
                        # a is always smaller than b
                        a_best = None
                        b_best = None
                        for k in range(0, len(impatchgroupreps)):
                            for l in range(k + 1, len(impatchgroupreps)):
                                # Pick the clustering centers
                                a = impatchgroupreps[k]
                                b = impatchgroupreps[l]
                                # Calculate the minimal clustering loss with these centers
                                clusterloss = np.sum(
                                    np.min(np.stack((np.abs(impatch - a), np.abs(impatch - b)), axis=2), axis=2))
                                if clusterloss < minclusterloss:
                                    minclusterloss = clusterloss
                                    a_best = a
                                    b_best = b
                        # Discretize impatch with the cluster centers
                        labels = np.zeros((7, 7))
                        try:
                            # Where b is closer than a, set values to 1. Elsewhere, they stay 0
                            labels[np.abs(impatch - b_best) < np.abs(impatch - a_best)] = 1
                        except:
                            hallo = 5
                        # Find the closest LUT (should be an exact match)
                        diffs = np.sum(np.abs(labels - lookuptables), axis=(1, 2))
                        diffs_sorted = np.sort(diffs)
                        # Get the indices of the lowest values
                        mininds = np.where(diffs == diffs_sorted[0])[0]
                        closesttable = lookuptables[mininds[0]]
                        closesttablevalue = lookupvalues[mininds[0]]

                        if b < a:
                            print("SOMETHING IS WRONG! b < a!")
                        # Version just taking the lower or higher cluster center. DOES NOT WORK.
                        # THE APPROACHES CAN BE COMBINED. CREATE MASK. IF THE CURRENT PIXEL IS INSIDE THE MASK,
                        # DON'T CHANGE THE VALUE. IF IT'S OUTSIDE, PICK THE CLUSTER CENTER CHOSEN BY LUT
                        # if closesttablevalue == 0:
                        #    new_value = a_best
                        # else:
                        #    new_value = b_best

                        # The version using a mask
                        # Get mask based on the lookup values and tables of the min indices
                        mask = (closesttablevalue == closesttable)
                        # If target pixel falls inside the mask, don't change the pixel value
                        # THIS IS WHAT MAKES IT DIFFERENT FROM JUST PICKING THE CLUSTER CENTER!
                        if mask[(7 - 1) // 2, (7 - 1) // 2] == 1:
                            new_value = impatch[(7 - 1) // 2, (7 - 1) // 2]
                        else:
                            # Potential new values
                            potvalues = impatch[mask == 1]
                            # Take the median value in the mask area, biased towards dark if needed
                            potvalues = np.sort(potvalues)
                            # Choose biased median in the mask area
                            # (biased only in the case of even number of values)
                            #new_value = potvalues[int((len(potvalues) - 1) / 2)]
                            # Choose median (unbiased) in the mask area
                            new_value = potvalues[int(len(potvalues) / 2)]
                    elif amount == 'low':
                        values = img_test[i - 1:i + 2, j - 1:j + 2].flatten()
                        # Make the central value small or big, to bias the values
                        # (the original central value should not affect the result)
                        if bias == 0:
                            values[4] = 0
                        elif bias == 1:
                            values[4] = 1
                        # Take the biased median
                        new_value = np.sort(values)[4]
                new_img[i, j] = new_value
        return new_img

    # 'amount' defines how much smoothing we want. 'low' or 'high'
    def LUTInterpRGB(img_test, lookuptables, lookupvalues, amount, bias=0):
        height = img_test.shape[0]
        width = img_test.shape[1]
        new_img = np.ones(img_test.shape)
        # Start processing from where the filter can fully overlap with the image
        for i in range(3, height - 3):
            print("Processing row " + str(i))
            for j in range(3, width - 3):
                impatch = img_test[i - 3:i + 3 + 1, j - 3:j + 3 + 1, :]
                # Check if impatch contains only one value (in each channel)
                if np.sum(np.std(impatch, axis=(0,1))) < 1e-9:
                    new_value = impatch[0, 0, :]
                # Never change the central pixel of a 3x3 superpixel
                elif i % 3 == 1 and j % 3 == 1:
                    new_value = impatch[(7 - 1) // 2, (7 - 1) // 2, :]
                # if we are looking at one of the corners of a 3x3 area, use the 3x3 "biased median"
                # Find the biased medoid (L1 minimizer)
                elif (i % 3 == 0 or i % 3 == 2) and (j % 3 == 0 or j % 3 == 2):
                    values_smallpatch = img_test[i - 1:i + 2, j - 1:j + 2, :]
                    new_value = biasedmedoid_RGB(values_smallpatch, bias=bias)
                # The case where we are looking at an edge of a superpixel. If we want a lot of
                # smoothing, use the LUT's. Otherwise, use the biased medoid again
                else:
                    if amount == 'high':
                        # Divide the impatch values into 9 different groups (the original pixel values).
                        # Then try all possible 2-clusterings on them to find
                        # the two-valued LUT that is the closest match in terms of absolute difference
                        impatchgroupreps = np.zeros((3, 3, 3))
                        for k in range(0, 3):
                            for l in range(0, 3):
                                impatchgroupreps[k, l, :] = impatch[3 * k, 3 * l, :]
                        impatchgroupreps = impatchgroupreps.reshape(9, 3)
                        # Go through all possible 2-clusterings.
                        # Remove duplicates from the different values. NOT SURE THIS WORKS CORRECTLY
                        impatchgroupreps = np.unique(impatchgroupreps, axis=0)
                        minclusterloss = 1e9
                        a_best = None
                        b_best = None
                        for k in range(0, len(impatchgroupreps)):
                            for l in range(k + 1, len(impatchgroupreps)):
                                # Pick the clustering centers
                                a = impatchgroupreps[k,:]
                                b = impatchgroupreps[l,:]
                                # Calculate the minimal clustering loss with these centers
                                clusterloss = np.sum(
                                    np.min(np.stack((np.abs(impatch - a), np.abs(impatch - b)), axis=3), axis=3))
                                if clusterloss < minclusterloss:
                                    minclusterloss = clusterloss
                                    a_best = a
                                    b_best = b
                        # Discretize impatch using the cluster centers
                        labels = np.zeros((7, 7))
                        if np.linalg.norm(b_best) > np.linalg.norm(a_best):
                            larger = b_best; smaller = a_best
                        else:
                            larger = a_best; smaller = b_best
                        try:
                            # Set the values closer to the larger cluster center to 1. Elsewhere, they stay 0
                            labels[np.linalg.norm(impatch - larger, ord=2, axis=2) < np.linalg.norm(impatch - smaller, ord=2, axis=2)] = 1
                        except:
                            hallo = 5
                        # Find the closest LUT (should be an exact match)
                        diffs = np.sum(np.abs(labels - lookuptables), axis=(1, 2))
                        diffs_sorted = np.sort(diffs)
                        # Get the indices of the lowest values
                        mininds = np.where(diffs == diffs_sorted[0])[0]
                        closesttable = lookuptables[mininds[0]]
                        closesttablevalue = lookupvalues[mininds[0]]

                        # Get a mask based on the lookup values and tables of the min indices
                        mask = (closesttablevalue == closesttable)
                        # If target pixel falls inside the mask, don't change the pixel value
                        # THIS IS WHAT MAKES IT DIFFERENT FROM JUST PICKING THE CLUSTER CENTER!
                        if mask[(7 - 1) // 2, (7 - 1) // 2] == 1:
                            new_value = impatch[(7 - 1) // 2, (7 - 1) // 2, :]
                        else:
                            # Potential new values
                            potvalues = impatch[mask == 1]
                            # Choose medoid in the mask area (is bias needed?)
                            new_value = medoid_RGB(potvalues)
                    else:
                        values_smallpatch = img_test[i - 1:i + 2, j - 1:j + 2, :]
                        new_value = biasedmedoid_RGB(values_smallpatch, bias=bias)

                # new_value = scipy.stats.mode(impatch[mask == 1])[0][0]
                new_img[i, j] = new_value
        return new_img

    # First Upscale the image by 3, using nearest neighbour scaling
    img_test = cv2.resize(img, (3*img.shape[1], 3*img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Then pad the image with 3 rows and columns on either side. At the end we will crop the result
    img_test = cv2.copyMakeBorder(img_test, 3, 3, 3, 3, borderType=cv2.BORDER_REPLICATE)

    # If the required amount of smoothing is high, assemble the LUT's
    if amount == 'high':
        # Assemble the LUTs using De Bruijn sheet
        img_input = cv2.imread("3x3 De Bruijn sheet resized.png")
        img_target = cv2.imread("3x3 De Bruijn sheet target A.png")

        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) / 255
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY) / 255

        height = img_input.shape[0]
        width = img_input.shape[1]

        lookuptables = []
        lookupvalues = []
        #lookuptables = {}
        # tobytes() returns a raw python byte string for the array, which is hashable
        #lookuptables[np.zeros((7, 7)).tobytes()] = 0
        # Need to initialize the lists with single values
        lookuptables.append(np.zeros((7, 7)))
        lookupvalues.append(0)
        LUTamounts = np.zeros((7*7+1, 2))
        # Gather the lookup tables. This could be done just once, and then the results saved somewhere.
        for i in range((7-1)//2, height-(7-1)//2):
            for j in range((7-1)//2, width-(7-1)//2):
                table = img_input[i-3:i+3+1, j-3:j+3+1]
                value_desired = img_target[i,j]
                # If the table has not been saved, add the table and the desired value
                if not (lookuptables == table).all(axis=(1,2)).any():
                    lookuptables.append(table)

                    lookupvalues.append(value_desired)

                # If table is in the list, check that the desired value is consistent
                else:
                    tableind = np.nonzero((lookuptables == table).all(axis=(1,2)))[0][0]
                    # Check that the desired value is consistent
                    if lookupvalues[tableind] != value_desired:
                        print(tableind)
                        print(lookuptables[tableind])
                        print(lookupvalues[tableind])
                        print(value_desired)
                        print("Lookup table corresponds to multiple values!")
                        print("Problem location: (" + str(i) + ", " + str(j) + ")")
                # Check which sums are present for which desired values
                if np.abs((i % 3) - (j % 3)) == 1:
                    LUTamounts[int(np.sum(table)), int(value_desired)] = 1
    else:
        lookuptables = None; lookupvalues = None

    # Color image
    if imgtype == "RGB":
        new_img = LUTInterpRGB(img_test, lookuptables, lookupvalues, amount=amount, bias=bias)

    # Grayscale image
    elif imgtype == "GRAY":
        new_img = LUTInterpGRAY(img_test, lookuptables, lookupvalues, amount=amount, bias=bias)

    # For B&W images we have an exact match for every image patch, and finding them can be a very fast process.
    # It isn't currently though, since we currently have a LUT search implementation
    # instead of using something like a dictionary
    elif imgtype == 'BW':
        new_img = np.copy(img_test)
        height = img_test.shape[0]
        width = img_test.shape[1]
        for i in range(3, height-3):
            print("Row " + str(i))
            for j in range(3, width-3):
                impatch = img_test[i-3:i+3+1, j-3:j+3+1]
                # I need the index of a table. Why not use a dictionary instead?
                tableind = np.nonzero((lookuptables == impatch).all(axis=(1,2)))[0][0]
                new_img[i,j] = lookupvalues[tableind]

    # At the end crop the new image
    new_img = new_img[3:new_img.shape[0]-3, 3:new_img.shape[1]-3]
    return new_img

imgtype = "RGB"
# amount of smoothing
amount = 'high'
bias = 0

imgfolder = "Images"

#imgname = "Beautifly.png"
#imgname = "9pkmn.png"
#imgname = "Punch_mech.png"
#imgname = "Rhydon.png"
#imgname = "3BW.png"
#imgname = "test_image3.png"
#imgname = "Burst_Laser_II.png"
#imgname = "Transcendence.png"
#imgname = "test_image2.png
imgname = "test_image2_color.png"

img_test = cv2.imread(os.path.join(imgfolder, imgname))

if imgtype == "BW" or imgtype == "GRAY":
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)/255
elif imgtype == "RGB":
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)/255

new_img = depixelate(img_test, imgtype=imgtype, amount=amount, bias=bias)

plt.rcParams['figure.figsize'] = (16,8)
plt.figure()
if imgtype == "BW" or imgtype == "GRAY":
    plt.imshow(new_img, cmap='gray', vmin=0, vmax=1)
elif imgtype == 'RGB':
    plt.imshow(new_img)
plt.show()

if imgtype == "BW" or imgtype == "GRAY":
    # vmin and vmax are needed to save the image correctly
    plt.imsave(os.path.join(imgfolder, "img_LUTprocessed.png"), new_img, cmap='gray', vmin=0, vmax=1)
elif imgtype == "RGB":
    plt.imsave(os.path.join(imgfolder, "img_LUTprocessed.png"), new_img)


