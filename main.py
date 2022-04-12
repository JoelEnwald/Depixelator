import numpy as np
import matplotlib.pyplot as plt

# Use some kind of sliding window setup, that goes through the image pixels one by one, and
# tries to fit first a constant function, then a line if the next point doesn't fit, then
# a parabola if the third point doesn't fit? If a more simple solution than an n-1:th degree polynomial
# is found, use that as long as possible, then have a discontinuation point?
# Goal is to minimize the number of parameters!

signal = [0,0,0,1,1,1,2,2,2,3,4,5,8,9,8,5,5,5,5,3,2,4,0,0,0]

plt.figure()
plt.scatter(range(0, len(signal)), signal)

poly_coeffs = np.zeros((len(signal), 3))
cutoffs = np.zeros(len(signal))
i = 0
j = 1
p = 0
tol = 1/256
while i < len(signal):
    # At least 2 same values, fit constant function
    if i + 1 < len(signal) and np.abs(signal[i+1] - signal[i]) < tol:
        l = 1
        while i+l < len(signal) and np.abs(signal[i+l] - signal[i+l-1]) < tol:
            l += 1
        poly_coeffs[p, 0] = signal[i]
        cutoffs[p] = i + l - 0.5
        p += 1
    # At least 3 values in a line, fit a line
    elif i+2 < len(signal) and np.abs((signal[i+2] - signal[i+1]) - (signal[i+1] - signal[i])) < tol:
        l = 2
        while i+l < len(signal) and np.abs((signal[i+l] - signal[i+l-1]) - (signal[i+l-1] - signal[i+l-2])) < tol:
            l += 1
        poly_coeffs[p, 1] = signal[i+1] - signal[i]
        poly_coeffs[p, 0] = signal[i] - poly_coeffs[p, 1]*i
        cutoffs[p] = i + l - 0.5
        p += 1
    i += 1

# Draw the polynomials
step = 0.1
x = np.arange(-0.5, len(signal) - 0.5 + step, step)
y = []
xk = -0.5
p = 0
c = 0
while xk <= np.max(x) + step/10:
    if xk >= cutoffs[c] and cutoffs[c] != 0:
        c += 1
        p += 1
    y.append(poly_coeffs[p,0] + poly_coeffs[p,1] * xk + poly_coeffs[p,2] * xk**2)
    xk += step
plt.plot(x, y)
plt.show()
hallo = 5

