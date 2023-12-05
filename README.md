This is a program meant for upscaling pixel art images by a factor of 3,
with a low or high amount of smoothing. The main function is depixelator.py.
I use De Bruijn sheets to encode the results I want, into look-up tables (LUT's).

WARNING: The high setting is currently extremely slow, probably due to how 
the LUT's are searched. Prepare to wait a while if used for big images.

The low setting can be used repeatedly on an image to make it even smoother,
though all lines will still be 0, 45 or 90 degree angles
