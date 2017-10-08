An OpenCV python to create 2 panoramas :

1.Using homographies and perspective warping on a common plane (3 images).
2.Using cylindrical warping (many images).
3.Using perspective warping and Laplacian Blending to stitch the images together
4.Using cylindrical warping and Laplacian Blending to stitch the images together

In both options:

Read the images from ./Source/ folder: input1.jpg, input2.jpg, input3.jpg
Calculate the transformation (homography for projective; affine for cylindrical) between each
Transform input2 and input3 to the plane of input1, since input1 is and produce output

Output Images: ./Result Folder

To run the program:

dbalani@dbalani:~/HW2-Panoramas$ python main.py 1 ./Source/input1.png ./Source/input2.png ./Source/input3.png ./Result/
dbalani@dbalani:~/HW2-Panoramas$ python main.py 2 ./Source/input1.png ./Source/input2.png ./Source/input3.png ./Result/
dbalani@dbalani:~/HW2-Panoramas$ python main.py 3 ./Source/input1.png ./Source/input2.png ./Source/input3.png ./Result/
dbalani@dbalani:~/HW2-Panoramas$ python main.py 4 ./Source/input1.png ./Source/input2.png ./Source/input3.png ./Result/




