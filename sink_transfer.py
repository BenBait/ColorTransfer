# Sinkhorn transfer relying heavily on external libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ot

im1_name = 'trail.jpg'
im2_name = 'beauty.jpg'

im1 = Image.open(im1_name).convert('RGB')
im1 = im1.resize((256, 256))

im2 = Image.open(im2_name).convert('RGB')
im2 = im2.resize((256, 256))

# get the normalized images
p1 = np.divide(np.array(im1.getdata()), 255.0)
p2 = np.divide(np.array(im2.getdata()), 255.0)

# sample 500 pixels from each image to avoid computation of gigantic cost matrix
sample_num = 500
idx1 = np.random.randint(p1.shape[0], size=(sample_num))
idx2 = np.random.randint(p2.shape[0], size=(sample_num))
t1 = p1[idx1, :]
t2 = p2[idx2, :]

# gamma is a regularization term for the entropy
# increasing this will make the image more gray/homogenous
gamma = 1e-2

# get the transformation model from the ot library
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=gamma, method="sinkhorn_log")
ot_sinkhorn.fit(Xs=t1, Xt=t2)
transp = ot_sinkhorn.transform(Xs=p1)
transp[transp<0] = 0
transp[transp>1] = 1

new_image = Image.fromarray((transp.reshape((256, 256, 3)) * 255).astype(np.uint8))

im1.show()
im2.show()
new_image.show()
