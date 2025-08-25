This is a program meant for upscaling pixel art images by a factor of 3,
with low or high amount of smoothing. The main function is depixelator.py.
I use De Bruijn sheets to encode the results I want into look-up tables (LUT's).

WARNING: The high setting is currently extremely slow, probably due to how 
the LUT's are searched. Prepare to wait a while if processing big images.

The low setting can be used repeatedly on an image to make it even smoother,
though all lines will still be at 0, 45 or 90 degree angles

Here are example outputs. The first two examples show the different degrees of smoothing possible.
![Example4](https://github.com/JoelEnwald/Depixelator/assets/6623412/6527181c-7e74-4835-83a8-07244abaf254)
![Example3](https://github.com/JoelEnwald/Depixelator/assets/6623412/bc7b3947-762a-473d-bf90-cc4115a6652f)
![Example1](https://github.com/JoelEnwald/Depixelator/assets/6623412/9a6e9924-159b-4f86-9e3f-8b0423b4d1ad)
![Example2](https://github.com/JoelEnwald/Depixelator/assets/6623412/4e664d29-9168-4ca8-a1a4-c5717b37b4f0)

This project was inspired by the excellent blog post here http://datagenetics.com/blog/december32013/index.html
