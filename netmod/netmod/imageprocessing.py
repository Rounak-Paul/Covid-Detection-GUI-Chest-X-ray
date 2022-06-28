import cv2
import numpy as np
import os

from vcolorpicker import getColor

#video to image frame by frame
def video2img(video,dir):
    # video = cv2.VideoCapture("8.mp4")
    i = 0
    os.chdir(dir)
    while True:
        ret, frame = video.read()
        cv2.imwrite("frame8_%d.jpg" % i, frame)
        i = i + 1

#grayscale
def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#color filter with ui
def colorFilterUI(img):
    low = np.zeros(3)
    high = np.array(list(getColor()))
    high = np.array([high[2],high[1],high[0]]) #OpenCV working on BGR format instead of RGB

    print(low,high)

    mask = cv2.inRange(img,low,high)
    result = cv2.bitwise_and(img,img,mask = mask)

    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting the Orginal image to Gray
    bw_bgr = cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR) # Converting the Gray image to BGR format
    result2 = cv2.bitwise_or(bw_bgr,result)
    return result2

    #color filter without UI
def colorFilter(img,rgb):
    low = np.zeros(3)
    high = rgb

    print(low,high)

    mask = cv2.inRange(img,low,high)
    result = cv2.bitwise_and(img,img,mask = mask)

    # bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting the Orginal image to Gray
    # bw_bgr = cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR) # Converting the Gray image to BGR format
    # result2 = cv2.bitwise_or(bw_bgr,result)
    return result

#image sharp
def sharp(image):
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp

#gaussian smooth
def smooth(image):
    gaussian = cv2.GaussianBlur(image, (3, 3), 0)
    return gaussian

#RGB layer extract
def layerExtract(img,n):
    blue, green, red = cv2.split(img)

    # We create a dummy 3D array
    blue_channel = np.zeros(img.shape, img.dtype)
    green_channel = np.zeros(img.shape, img.dtype)
    red_channel = np.zeros(img.shape, img.dtype)


    cv2.mixChannels([blue, green, red], [blue_channel], [0,0])
    cv2.mixChannels([blue, green, red], [green_channel], [1,1])
    cv2.mixChannels([blue, green, red], [red_channel], [2,2])

    if n == 0:
        return red_channel
    elif n == 1:
        return green_channel
    elif n ==2:
        return blue_channel
    else: 
        return

#histEqu
def histogramEqualization(img):
    blue, green, red = cv2.split(img)

    # Calculate histogram of each channel
    hist_blue = cv2.calcHist([blue], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
    hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])

    # Calculate the CDF for each histogram channel
    cdf_blue = hist_blue.cumsum()
    cdf_green = hist_green.cumsum()
    cdf_red = hist_red.cumsum()

    # Mask null values 
    cdf_blue_masked = np.ma.masked_equal(cdf_blue, 0)
    cdf_green_masked = np.ma.masked_equal(cdf_green, 0)
    cdf_red_masked = np.ma.masked_equal(cdf_red, 0)

    # Apply Equalization Formula to all none masked values: (y - ymin)*255 / (ymax - ymin)
    cdf_blue_masked = (cdf_blue_masked - cdf_blue_masked.min())*255 / (cdf_blue_masked.max() - cdf_blue_masked.min())
    cdf_green_masked = (cdf_green_masked - cdf_green_masked.min())*255 / (cdf_green_masked.max() - cdf_green_masked.min())
    cdf_red_masked = (cdf_red_masked - cdf_red_masked.min())*255 / (cdf_red_masked.max() - cdf_red_masked.min())

    cdf_final_b = np.ma.filled(cdf_blue_masked, 0).astype('uint8')
    cdf_final_g = np.ma.filled(cdf_green_masked, 0).astype('uint8')
    cdf_final_r = np.ma.filled(cdf_red_masked, 0).astype('uint8')

    # Merge all channels:
    blue_img = cdf_final_b[blue]
    green_img = cdf_final_g[green]
    red_img = cdf_final_r[red]

    # Final output, equalized image obtained from merging the respective channels 
    img = cv2.merge((blue_img, green_img, red_img))
    return img

#edge
def edge(img):
    return cv2.Laplacian(img, cv2.CV_8UC3)

#powerlaw
def powerlaw(img,n):
    # n is the value of gamma(user input)
    img = np.array(255*(img/255)**n,dtype='uint8')
    return img
