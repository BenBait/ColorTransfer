# The Linear Monge-Kantorovitch Transfer (LMKT) Algorithm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import sqrtm

im1_name = 'trail.jpg'
im2_name = 'beauty.jpg'

# For the paper, we swap im1_name with im2_name to see
# how both directions work
im1 = Image.open(im1_name).convert('RGB')
im1 = im1.resize((256, 256))

im2 = Image.open(im2_name).convert('RGB')
im2 = im2.resize((256, 256))

# get the normalized RGB pixel data
pixels1 = np.divide(np.array(im1.getdata()), 255.0)
pixels2 = np.divide(np.array(im2.getdata()), 255.0)

# get the covariance matrices 
cov1 = np.cov(pixels1.T)
cov2 = np.cov(pixels2.T)

# construct the terms needed for the closed form lmkt
T_inner = sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1))
T = np.linalg.inv(sqrtm(cov1)) @ T_inner @ np.linalg.inv(sqrtm(cov1))
transform1 = ((pixels1 - pixels1.mean(axis=0)) @ T) + pixels2.mean(axis=0)

# normalize values between 0 and 1 (which will be between 0 and 255)
transform1 = transform1.clip(min=0, max=1)

new_image = Image.fromarray((transform1.reshape((256, 256, 3)) * 255).astype(np.uint8))

im1.show()
im2.show()
new_image.show()
