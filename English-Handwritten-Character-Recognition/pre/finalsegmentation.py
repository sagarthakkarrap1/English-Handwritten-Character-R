import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
from PIL import Image
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
from .hcr_final import predict


def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2  # get integer-valued resize of division

    kernel = np.zeros([kernelSize, kernelSize])  # kernel
    sigmaX = sigma  # scale factor for X dimension
    sigmaY = sigma * theta  # theta - multiplication factor = sigmaX/sigmaY

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def lineSegmentation(img, kernelSize=25, sigma=11, theta=7):

    img_tmp = np.transpose(prepareTextImg(img))# image to be segmented (un-normalized)
    img_tmp_norm = normalize(img_tmp)
    k = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img_tmp_norm, -1, k, borderType=cv2.BORDER_REPLICATE)
    img_tmp1 = normalize(imgFiltered)
    summ_pix = np.sum(img_tmp1, axis = 0)
    smoothed = smooth(summ_pix, int(img.shape[0]/11))
    # smoothed = smooth(summ_pix, 30)
    mins = np.array(argrelmin(smoothed, order=2))
    found_lines = transpose_lines(crop_text_to_lines(img_tmp, mins[0]))
    return found_lines

def prepareImg(img, height=500):
    """ Convert given image to grayscale image (if needed) and resize to desired height. """
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)
#     return(img)
def prepareTextImg(img):
    """ Convert given text image to grayscale image (if needed) and return it. """
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img)

def normalize(img):
    """ Normalize input image:
    img = (img[][]-mean)/ stddev
    using function: cv2.meanStdDev(src[, mean[, stddev[, mask]]]), returns: mean, stddev
    where: mean & stddev - numpy.ndarray[][] """
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def smooth(x, window_len=11, window='hanning'):
    """ Image smoothing is achieved by convolving the image with a low-pass filter kernel.
    Such low pass filters as: ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'] can be used
    It is useful for removing noise. It actually removes high frequency content
    (e.g: noise, edges) from the image resulting in edges being blurred when this is filter is applied."""
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y


def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    avg_diff=0
    for i, blank in enumerate(blanks):
        x2 = blank
        avg_diff=avg_diff+int(x2-x1)
#         print("x1=", x1, ", x2=", x2, ", Diff= ", x2-x1)
        x1 = blank
    avg_diff=float(avg_diff)/(2*len(blanks))
    x1=0
    for i, blank in enumerate(blanks):
        x2 = blank
        if x2-x1 >= int(avg_diff):
            line = text[:, x1:x2]
            lines.append(line)
        x1 = blank

#     print(avg_diff,len(blanks))
    
    return lines

def transpose_lines(lines):
    res = []
    for l in lines:
        line = np.transpose(l)
        if len(line)!=0:
            res.append(line)
    return res

def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if not orient in ['vertical', 'horizontal']:
        raise ValueError("Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'") 
    if orient == 'vertical': 
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    else:
            for i, l in enumerate(lines_arr):
                line = l
                plt.subplot(40, 1, i+1)  # A grid of 40 rows x 1 columns
                plt.axis('off')
                plt.title("Line #{0}".format(i))
                _ = plt.imshow(line, cmap='gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def save_lines(img_path):
    img = cv2.imread(img_path)
    img2=prepareImg(img)
    lines = lineSegmentation(img2)
    i=0
    print(len(lines))
    for line in range(len(lines)):
        line_path="media/lines/"+str(i)+".jpg"
        print(line_path)
        cv2.imwrite(line_path,lines[line])
        i=i+1

    i=0
    list=[]
    for line in range(len(lines)):
        line_path="media/lines/"+str(i)+".jpg"
        save_words(line_path,i,list)
        i=i+1
    print(list)
    return list

def wordSegmentation(img,i,kernelSize=21, sigma=11, theta=7, minArea=70):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

	Args:
		img: grayscale uint8 image of the text-line to be segmented.
		kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.

	Returns:
		List of tuples. Each tuple contains the bounding box and the image of the segmented word.
	"""

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    # The function applies an arbitrary linear filter to an image.
    # int ddepth (=-1) - desired depth of the destination image
    # anchor - indicates the relative position of a filtered point within the kernel;
    # default value (-1,-1) means that the anchor is at the kernel center.
    # borderType - pixel extrapolation method:
    # cv2.BORDER_REPLICATE -  The row or column at the very edge of the original is replicated to the extra border.
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    # threshold - If pixel value is greater than a threshold value, it is assigned one value, else it is assigned another value
    # img - source image, which should be a grayscale image.
    # Second argument is the threshold value which is used to classify the pixel values.
    # Third argument is the maxVal which represents the value to be given if pixel value is more than the threshold value.
    # Last - different styles of thresholding
    # Returns: threshold value computed, destination image
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
 
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
#         cv2.RETR_EXTERNAL or cv2.RETR_LIST - ???
        (components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        [x, y, w, h] = currBox
        res.append((x,y,w,h))
    res.sort(key=lambda tup:tup[0])
    pred_sen=""
    j=0
    for r in res:
        [x,y,w,h] = r
        b=8
        while y-b<0:
            b=b-5    
        currImg = img[y-b:y+h,x:x+w]

        if w<40 :
            save=cv2.resize(currImg,(70,50))
        elif w<140:
            save=cv2.resize(currImg,(150,50))
        else:
            save=cv2.resize(currImg,(250,50))
        word_path="media/words/"+str(i)+"_"+str(j)+".jpg"
        cv2.imwrite(word_path,save)
        pred_sen+=save_chars(word_path,i,j)
        j=j+1
    return pred_sen




def save_words(line_path,i,list): 
    print("save_words",i)
    img = prepareTextImg(cv2.imread(line_path))
    pred_sen=wordSegmentation(img,i)
    list.append(pred_sen)
    
    
import skimage.filters as filters
def charSegmentation(img,i,j, kernelSize=9, sigma=11, theta=7, minarea=350):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf"""

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)

    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    # findContours - The function retrieves contours from the binary image
    # First argument is source image, second is contour retrieval mode, third is contour approximation method.
    if cv2.__version__.startswith('3.'):
        
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        print("version")
#         cv2.RETR_EXTERNAL or cv2.RETR_LIST - ???
        (components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minarea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        [x, y, w, h] = currBox
        res.append((x,y,w,h))
    res.sort(key=lambda tup:tup[0])
    k=0
    p_word=""
    for r in res:
        [x,y,w,h] = r
        print(w,h)
        if w*h<5000 or h<=100:   
          print("hello")
          continue 
        if w<=100:
            print(y)
            currImg = img[:y+h+10,x:x+w+5]
        else:
            currImg = img[y:y+h+10,x:x+w+5]

        ret,currImg=cv2.threshold(currImg,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=Image.fromarray(currImg)
        if w<=100:
            print("h")
            image=image.resize((6,26), Image.ANTIALIAS)
            right = 10
            left = 10
            top = 1
            bottom = 1
        else:    
#         currImg=cv2.resize(currImg, (24,24),interpolation = cv2.INTER_AREA)
            image=image.resize((18,18), Image.ANTIALIAS)
            right = 5
            left = 5
            top = 5
            bottom = 5
        width, height = image.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(image.mode, (new_width, new_height), 255)
        result.paste(image, (left, top))
        char_path="media/chars/"+str(i)+"_"+str(j)+"_"+str(k)+".jpg"
        result.save(char_path)
        c=predict(char_path)
        
        p_word+=predict(char_path)
        k=k+1
    p_word+=" "
    # return list of words, sorted by x-coordinate
    return p_word


def save_chars(img_path,i,j): 
    img = prepareImg(cv2.imread(img_path),500)
    p_word=charSegmentation(img,i,j)
    return p_word

# constraints:
#For character segmentation printedline should not be visible- words should be written on plain paper



