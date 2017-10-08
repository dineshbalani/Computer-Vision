An OpenCV program to do the following things:

1.Perform Histogram Equalization on the given input image.
2.Perform Low-Pass, High-Pass and Deconvolution on the given input image.
3.Perform Laplacian Blending on the two input images (blend them together).

Input Images: /Source Folder - 
input1.jpg - Input for Task1
input2.png - Input for Task 2 (LPF AND HPF)
blurred2.exr - Input for Task 2 (Deconvolution)
input3A.jpg input3B.jpg - Input for Task 3 (Laplacian blending)

Output Images: /Result Folder

To run the program:

dbalani@dbalani:/HW1-Filters$ python main.py 1 input1.jpg /home/dbalani/HW1-Filters/
dbalani@dbalani:/HW1-Filters$ python main.py 2 input2.png blurred2.exr /home/dbalani/HW1-Filters/
dbalani@dbalani:/HW1-Filters$ python main.py 3 input3A.jpg input3B.jpg /home/dbalani/HW1-Filters/