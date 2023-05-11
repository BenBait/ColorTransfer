import math
import numpy as np

# Adapted from Reinhard's paper and 
#https://stackoverflow.com/questions/32696138/converting-from-rgb-to-l%CE%B1%CE%B2-color-spaces-and-converting-it-back-to-rgb-using-open
# for some reason the matlab code in the link above does not use the direct
# RGB to LMS transformation, but this matrix is specified clearly in Reinhard et al
def RGBtoLAB(u_r, u_g, u_b):
    L = 0.3811*u_r+0.5783*u_g+0.0402*u_b
    M = 0.1967*u_r+0.7244*u_g+0.0782*u_b
    S = 0.0241*u_r+0.1288*u_g+0.8444*u_b

    L = np.where(L == 0.0, 1.0, L)
    M = np.where(M == 0.0, 1.0, M)
    S = np.where(S == 0.0, 1.0, S)

    u_l = (1.0 / math.sqrt(3.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (1.0000 * np.log10(S)))
    u_a = (1.0 / math.sqrt(6.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (-2.0000 * np.log10(S)))
    u_b = (1.0 / math.sqrt(2.0)) * ((1.0000 * np.log10(L)) + (-1.0000 * np.log10(M)) + (-0.0000 * np.log10(S)))

    return u_l, u_a, u_b

def LABtoRGB(u_l, u_a, u_b):
    _L = u_l*1.7321
    Alph = u_a*2.4495
    Beta = u_b*1.4142

    L = (0.33333*_L) + (0.16667 * Alph) + (0.50000 * Beta)
    M = (0.33333 * _L) + (0.16667 * Alph) + (-0.50000 * Beta)
    S = (0.33333 * _L) + (-0.33333 * Alph) + (0.00000 * Beta)

    L = np.power(10, L)
    M = np.power(10, M)
    S = np.power(10, S)

    L = np.where(L == 1.0, 0.0, L)
    M = np.where(M == 1.0, 0.0, M)
    S = np.where(S == 1.0, 0.0, S)

    u_r = 4.36226*L-3.58076*M+0.1193*S
    u_g = -1.2186*L+2.3809*M-0.1624*S
    u_b = 0.0497*L-0.2439*M+1.2045*S
    u_r[u_r<0] = 0
    u_r[u_r>255] = 255
    u_g[u_g<0] = 0
    u_g[u_g>255] = 255
    u_b[u_b<0] = 0
    u_b[u_b>255] = 255

    return u_r, u_g, u_b
