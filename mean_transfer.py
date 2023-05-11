import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from color_space import RGBtoLAB, LABtoRGB

im1_name = 'trail.jpg'
im2_name = 'beauty.jpg'

# For the paper, we swap im1_name with im2_name to see
# how both directions work
im1 = Image.open(im1_name).convert('RGB')
im1 = im1.resize((256, 256))

im2 = Image.open(im2_name).convert('RGB')
im2 = im2.resize((256, 256))

# Convert images to numpy arrays and split into channels
u_arr = np.array(im1)
u_r, u_g, u_b = np.split(u_arr, 3, axis=2)
v_arr = np.array(im2)
v_r, v_g, v_b = np.split(v_arr, 3, axis=2)

# Convert RGB channels to LAB color space
u_l, u_a, u_b = RGBtoLAB(u_r, u_g, u_b)
v_l, v_a, v_b = RGBtoLAB(v_r, v_g, v_b)

# Calculate mean and standard deviation of each channel
u_mean_l, u_std_l = np.mean(u_l), np.std(u_l)
u_mean_a, u_std_a = np.mean(u_a), np.std(u_a)
u_mean_b, u_std_b = np.mean(u_b), np.std(u_b)
v_mean_l, v_std_l = np.mean(v_l), np.std(v_l)
v_mean_a, v_std_a = np.mean(v_a), np.std(v_a)
v_mean_b, v_std_b = np.mean(v_b), np.std(v_b)

# Implement color transfer algorithm for each channel
w_l = ((u_l - u_mean_l) / u_std_l) * v_std_l + v_mean_l
w_a = ((u_a - u_mean_a) / u_std_a) * v_std_a + v_mean_a
w_b = ((u_b - u_mean_b) / u_std_b) * v_std_b + v_mean_b

# Convert LAB channels back to RGB color space
w_r, w_g, w_b = LABtoRGB(w_l, w_a, w_b)
w = np.concatenate([w_r, w_g, w_b], axis=2)

# Create PIL image from numpy array and return
new_image = Image.fromarray(np.uint8(w))
im1.show()
im2.show()
new_image.show()
