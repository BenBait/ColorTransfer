import scipy
from scipy import stats
import cvxpy as cp
import math
import numpy as np
from PIL import Image
import color_space
from color_space import RGBtoLAB, LABtoRGB

# experiment with these parameters, they significantly change the artifacts
# in the output image
NUM_ROTATIONS = 40 # MAX 40 because we are loading from a set of rotations
#NUM_ROTATIONS = 30 # Uncomment if you swap im1 and im2 to recreate image
                    # in the idt transfer quad image in the paper
NUM_BINS      = 400

im1_name = 'trail.jpg'
im2_name = 'beauty.jpg'

im1 = Image.open(im1_name).convert('RGB')
im1 = im1.resize((256, 256))

im2 = Image.open(im2_name).convert('RGB')
im2 = im2.resize((256, 256))

# Convert images to numpy arrays and split into channels
u_arr = np.array(im1)
u_arr = u_arr.reshape(-1, u_arr.shape[-1])
# Convert to lab space; I experimented this and the results look better
# than just using RGB
u_l, u_a, u_b = RGBtoLAB(u_arr.T[0], u_arr.T[1], u_arr.T[2])
u_arr = np.stack((u_l, u_a, u_b), axis=1)

v_arr = np.array(im2)
v_arr = v_arr.reshape(-1, v_arr.shape[-1])
v_l, v_a, v_b = RGBtoLAB(v_arr.T[0], v_arr.T[1], v_arr.T[2])
v_arr = np.stack((v_l, v_a, v_b), axis=1)

# Load the rotation matrices
# These rotation matrices work best for some reason, I tried several and saved
# the results. Each R is in SO(3).
Rs = np.load("Rs.npy")
# RR is for remapping the values of u with the new transformed channel
RR = np.zeros(u_arr.T.shape)
# Iterate through each rotation
for i in range(0, NUM_ROTATIONS):
    R = Rs[i]
    # Rotate images to a different coordinate space
    RU = np.dot(R, u_arr.T)
    RV = np.dot(R, v_arr.T)
    # iterate through each channel
    for j in range(3):
        # construct cdf of each image; this is the marginals
        # for matching the distribution functions
        abs_range=[min(RU[j] + RV[j]), max(RU[j] + RV[j])]
        cdf_u = (np.histogram(RU[j], bins=NUM_BINS, range=abs_range)[0]).cumsum()
        cdf_v = (np.histogram(RV[j], bins=NUM_BINS, range=abs_range)[0]).cumsum()
        # now normalize to between 0 and 1
        np.divide(cdf_u, max(cdf_u))
        np.divide(cdf_v, max(cdf_v))
        # get the transport map and remap the values of U
        RR[j] = np.interp(RU[j], np.histogram_bin_edges(RU[j], bins=NUM_BINS, range=abs_range)[1:],
                np.interp(cdf_u, cdf_v, np.histogram_bin_edges(RU[j], bins=NUM_BINS, range=abs_range)[1:]))
    # re-project, like in a Radon transformation
    w = np.linalg.solve(R, (RR - RU))
    w += u_arr.T

w_l, w_a, w_b = LABtoRGB(w[0], w[1], w[2])
w = np.stack((w_l, w_a, w_b), axis=1)
w = w.reshape((256, 256, 3))

new_image = Image.fromarray(np.uint8(w))
im1.show()
im2.show()
new_image.show()
