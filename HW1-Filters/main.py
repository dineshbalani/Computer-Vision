
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ================ Histogram equalization ===========
# ===================================================

def custom_equalizeHist(channel):
   pass

def histogram_equalization(img_in):
   
   b, g, r = cv2.split(img_in)
   
   pixels = img_in.size
   pixelsb = b.size
   pixelsg = g.size
   pixelsr = r.size
 
   hb = cv2.calcHist([b],[0],None,[256],[0,256])
   hg = cv2.calcHist([g],[0],None,[256],[0,256])
   hr = cv2.calcHist([r],[0],None,[256],[0,256])
   
   cdfb = np.cumsum(hb)
   cdfb_n = cdfb * hb.max()/ cdfb.max()
   cdfg = np.cumsum(hg)
   cdfg_n = cdfg * hg.max()/ cdfg.max()
   cdfr = np.cumsum(hr)
   cdfr_n = cdfr * hr.max()/ cdfr.max()
   
   eqblue = np.round((cdfb-cdfb.min())*255/(pixelsb-1))   
   eqred = np.round((cdfr-cdfr.min())*255/(pixelsr-1))
   eqgreen = np.round((cdfg-cdfg.min())*255/(pixelsg-1))

   blue = eqblue[b]
   green = eqgreen[g]
   red = eqred[r]
   
   img_out = cv2.merge((blue, green, red))
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = "output1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ========== 2: Frequency domain filtering ==========
# ===================================================

def low_pass_filter(img_in):
   
   img = img_in
   dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
   dft_shift = np.fft.fftshift(dft)
   magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
   rows, cols = img.shape
   crow,ccol = rows/2 , cols/2
   
   mask = np.zeros((rows,cols,2),np.uint8)
   mask[int(crow)-10:int(crow)+10, int(ccol)-10:int(ccol)+10] = 1
   
   # apply mask and inverse DFT
   fshift = dft_shift*mask
   f_ishift = np.fft.ifftshift(fshift)
   img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
   img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
   plt.subplot(121),plt.imshow(img, cmap = 'gray')
   plt.title('Input Image'), plt.xticks([]), plt.yticks([])
   plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.show()
   img_out = img_back

   return True, img_out

def high_pass_filter(img_in):

   # high pass filter here
   img = img_in
   f = np.fft.fft2(img)
   fshift = np.fft.fftshift(f)
   magnitude_spectrum = 20*np.log(np.abs(fshift))
   plt.subplot(121),plt.imshow(img, cmap = 'gray')
   plt.title('Input Image'), plt.xticks([]), plt.yticks([])
   plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
   plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
   plt.show()

   rows, cols = img.shape
   crow,ccol = rows/2 , cols/2
   fshift[int(crow)-10:int(crow)+10, int(ccol)-10:int(ccol)+10] = 1
   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_back = np.abs(img_back)
   plt.subplot(131),plt.imshow(img, cmap = 'gray')
   plt.title('Input Image'), plt.xticks([]), plt.yticks([])
   plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
   plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
   #plt.subplot(133),plt.imshow(img_back)
   #plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
   plt.show()
   img_out = img_back
   return True, img_out

def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im),newsize)
    return np.fft.fftshift(dft)

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

 
def deconvolution(img_in):
   
   im=img_in
   
   # Write deconvolution codes here
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   imf = ft(im, (im.shape[0],im.shape[1])) # make sure sizes match
   gkf = ft(gk, (im.shape[0],im.shape[1])) # so we can multiple easily
   imconvf = imf / gkf

   # now for example we can reconstruct the blurred image from its FT
   img_out_f = ift(imconvf)
   img_out = np.array(img_out_f * 255, dtype = np.uint8) # Deconvolution result
   
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2],0);
   #input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread("blurred2.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = "output2LPF.png"
   output_name2 = "output2HPF.png"
   output_name3 = "output2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# =========  3: Laplacian pyramid blending ==========
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   A = img_in1
   B = img_in2
   A = A[:,:A.shape[0]]
   B = B[:A.shape[0],:A.shape[0]]
   xrange=range
   # generate Gaussian pyramid for A
   G = A.copy()
   gpA = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpA.append(G)
   # generate Gaussian pyramid for B
   G = B.copy()
   gpB = [G]
   for i in xrange(6):
       G = cv2.pyrDown(G)
       gpB.append(G)
   # generate Laplacian Pyramid for A
   lpA = [gpA[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpA[i])
       L = cv2.subtract(gpA[i-1],GE)
       lpA.append(L)
   # generate Laplacian Pyramid for B
   lpB = [gpB[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpB[i])
       L = cv2.subtract(gpB[i-1],GE)
       lpB.append(L)
   # Now add left and right halves of images in each level
   LS = []
   for la,lb in zip(lpA,lpB):
       rows,cols,dpt = la.shape
       ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
       LS.append(ls)
   # now reconstruct
   ls_ = LS[0]
   for i in xrange(1,6):
       ls_ = cv2.pyrUp(ls_)
       ls_ = cv2.add(ls_, LS[i])   
   
   img_out = ls_ # Blending result
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = "output3.png"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
         question_number = int(sys.argv[1])
             
         if (question_number == 1 and not(len(sys.argv) == 4)):
             help_message()
             sys.exit()
         if (question_number == 2 and not(len(sys.argv) == 5)):
            help_message()
            sys.exit()
         if (question_number == 3 and not(len(sys.argv) == 5)):
            help_message()
            sys.exit()
         if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
