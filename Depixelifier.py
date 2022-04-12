import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# Original resizing algorithm is:-
# - upscale image 3x with nearest neighbour
# - "Blur" the image using biased median blurring with 3x3 kernel
# Alternative algorithm:
# - Upscale image 2x with NN, blur with unbiased median 3x3 kernel
# The algorithm doesn't add any detail, but makes the images less pixelated
# TEST WITH MULTIPLE DIFFERENT IMAGES AND COMPARE TO NN, BILINEAR AND BICUBIC UPSCALING!
# Take an image, scale it to half by bilinear sampling, then test different upscaling algorithms
# Train a tiny neural network using the test images! Create target upscaled images!

#img = cv2.imread("profilepic.png")
#img = cv2.imread("9pkmn.png")
#img = cv2.imread("Ice Wind - Day.png")
img = cv2.imread("test_image2.png")
#img = cv2.imread("Vista.png")
#img = cv2.imread("Resized_img.png")
#img = cv2.imread("3x5pixel_font.png")
#img = cv2.imread("1FRLG.png")
#img = cv2.imread("9PT.png")
#img = cv2.imread("Transcendence.png")
#img = cv2.imread("22DP.png")
#img = cv2.imread("137HGSS.png")
#img = cv2.imread("Elvin City.png")
#img = cv2.imread("Jungle Waterfall - Morning.png")
#img = cv2.imread("3BW.png")
#img = cv2.imread("test_pixel.png")
# Make the image grayscale + scale values to [0,1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255

scale = 3
# Size of the filter
a = 3
FILTER = "median_biased"

img_prev = np.copy(img)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')


# This biases towards dark if the image is bright, and vice versa
hallo = np.mean(img)
if np.mean(img) > 0.5:
    bias = -1
else:
    bias = 0

for t in range(0, 1):
    new_size = (int(scale * img_prev.shape[1]), int(scale * img_prev.shape[0]))
    img_prev_resized = cv2.resize(img_prev, dsize=new_size, interpolation=cv2.INTER_NEAREST)
    img_new = np.ones(img_prev_resized.shape)
    # No need to look at edges of the image
    for i in range(2, img_prev_resized.shape[0]-2):
        for j in range(2, img_prev_resized.shape[1]-2):
            filter_values = []
            for k in range(int(-(a-1)/2),int((a-1)/2)+1):
                for l in range(int(-(a - 1) / 2), int((a - 1) / 2) + 1):
                    filter_values.append(img_prev_resized[i+k, j+l])
            if FILTER == "median":
                filter_values.sort()
                new_value = filter_values[int((a * a - 1) / 2)]
            elif FILTER == "median_biased":
                # Don't consider central pixel
                filter_values.pop(int(a * a / 2))
                filter_values.sort()
                new_value = filter_values[int((a * a - 1) / 2) + bias]
            # This is biased towards changing values
            elif FILTER == "median_centralchangebiased":
                centerpixel_value = filter_values[int(a * a / 2)]
                # Don't consider central pixel
                filter_values.pop(int(a*a/2))
                filter_values.sort()
                if centerpixel_value > 0.5:
                    bias = -1
                else:
                    bias = 0
                new_value = filter_values[int((a*a-1)/2) + bias]
            # This looks at the local environment, and biases the new value away from the mean
            elif FILTER == "median_localenvbiased":
                mean_env_value = 0
                # This looks for the mean value in the local environment, and biases the filter to choose a value towards 0.5
                pixelcount = 0
                i_3 = int(i / 3)
                j_3 = int(j / 3)
                # Look at a 5x5 environment
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        if 0 <= i_3+k < img.shape[0] and 0 <= j_3+l < img.shape[1]:
                            mean_env_value += img[i_3+k, j_3+l]
                            pixelcount += 1
                mean_env_value /= pixelcount
                if mean_env_value > 0.5:
                    bias = -1
                elif mean_env_value < 0.5:
                    bias = 0
                else:
                    bias = 0
                # Don't consider central pixel
                filter_values.pop(int(a * a / 2))
                filter_values.sort()
                new_value = filter_values[int(a * a / 2) + bias]
                hallo = 5
            # scale must be set to 5 for this to work!
            if FILTER == "median_biased_3x3plus5x5":
                # Don't consider central pixel
                filter_values.pop(int(a * a / 2))
                # Repeat 3x3 filter values (minus centre)
                for k in range(6, 9):
                    filter_values.append(filter_values[k])
                for k in range(11, 13):
                    filter_values.append(filter_values[k])
                for k in range(15, 18):
                    filter_values.append(filter_values[k])
                filter_values.sort()
                new_value = filter_values[int(len(filter_values) / 2) + bias]
            img_new[i, j] = new_value
        print(i)
    img_prev = img_new

cv2.imwrite("img_processed.png", img_new*255)

plt.subplot(1, 2, 2)
plt.imshow(img_new, cmap='gray')
plt.show()
print("Mean of img is " + str(np.mean(img)))
print("Mean of img_new is " + str(np.mean(img_new)))
hallo = 5
