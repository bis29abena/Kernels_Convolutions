# usage
# python main.py --image image_file

# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2 as cv


def convolution(image, kernel):
    # grab the spatial dimension of the image
    # along with the spatial dimensions of the kernel
    (ih, iw) = image.shape[:2]
    (kh, kw) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size are not reduced (width and height)
    pad = (kw - 1) // 2
    print(f"Pad: {pad}")
    image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    output = np.zeros((ih, iw), dtype="float32")

    # loop over the input image sliding the kernel across
    # each (x, y) - coordinates from left to right and top to down
    for y in np.arange(pad, ih + pad):
        for x in np.arange(pad, iw + pad):
            # extract the ROI of the image by extracting the
            # center region of the current (x, y) coordinates dimension.
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the element wise
            # multiplication between the ROI and the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x, y)
            # coordinates of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return output image
    return output


# construct the arument parse to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image", type=str)
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeblur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")

# construct the laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")

# construct the sobel x-axis kernel
sobelX = np.array(([-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]), dtype="int")

# construct the sobel y-axis kernel
sobelY = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]), dtype="int")

# construct a kernel bank, a list of kernels we are going
# to apply using both our custom "convolve" function and
# opencvs filter2D function

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeblur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY)
)

# load the image and converte it to grayscale
image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelname, kernel) in kernelBank:
    # apply the kernel to the grayscale image and using both
    # our custom "convole" function and opencvs filter2D
    # function
    print(f"INFO applying {kernelname}")
    convoleOutput = convolution(gray, kernel)
    opencvOutput = cv.filter2D(gray, -1, kernel)

    # show the output image
    cv.imshow("original", gray)
    cv.imshow(f"{kernelname} - Convole", convoleOutput)
    cv.imshow(f"{kernelname} - Opencv", opencvOutput)
    cv.waitKey(0)
    cv.destroyAllWindows()
