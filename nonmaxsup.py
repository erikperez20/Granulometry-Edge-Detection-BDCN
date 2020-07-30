import cv2 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def skeletonize(img):


    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def thresholdxd(img, lowThreshold=10, highThreshold=140):
    
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(100)
    strong = np.int32(255)
    #print("weak weak: ", weak , strong)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak 
    
    
    return (res, weak, strong)

def hysteresis(image, weak, strong=255):
    
    
    img=image.copy()
    M, N = img.shape  
    img[0,:]=strong
    img[:,0]=strong
    img[M-1,:]=strong
    img[:,N-1]=strong
    for i in range(1, M-1):
        
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                        #print("strong")
                    else:
                        img[i, j] = 0
                        #print("0 xd")
                except IndexError as e:
                    pass
    
    
    #print ("img: ", img)
    return img

def LowerAndUpper(image):

    #image = Image.fromarray(img)
    img=image.copy()
    img = np.array(img)
    #plt.figure(1)
    #plt.imshow(img, 'gray')
    #plt.title('Grey-scale Map')

    # show histogaram
    bins = np.arange(256)
    hist, _ = np.histogram(img, np.hstack((bins, np.array([256]))))
    #print(hist)
    #plt.figure(2)
    #plt.bar(bins, hist)
    #plt.title('Histogram')

    # solve otsu threshold
    N = img.size
    hist_norm = hist / N
    #print(hist_norm)
    max_delta2 = 0
    
    
    for T in range(255):
        mu0 = 0
        mu1 = 0
        omega0 = np.sum(hist_norm[1:T+1])
        omega1 = 1-omega0
        for i in range(T+1):
            mu0 = mu0 + i * hist_norm[i]
        if omega0 != 0:
            mu0 = mu0 / omega0
        for i in range(T+1, 256):
            mu1 = mu1 + i * hist_norm[i]
        if omega1 != 0:
            mu1 = mu1 / omega1
        delta2 = omega0 * omega1 * (mu0-mu1)**2
        if max_delta2 < delta2:
            max_delta2 = delta2
            threshold = T

    #print('the otsu threshold is', threshold)
    ## image segmentation
    #img[img > threshold] = 255
    #img[img != 255] = 0
    #plt.figure(3)
    #plt.imshow(img,'gray')
    #plt.title('Segmentation Picture')
    #plt.imsave("blastingimg.png",img)
    #cv2.imwrite("img3.png", img)
    ## show all
    #plt.show()
    lower=threshold/2.5
    upper=threshold*1.2
    if upper>255:
        upper=threshold

    #print(lower," :lower/upper: ",upper," threshold: ", threshold)

    return lower,upper,threshold

def skeletonxd(img):

    #img=np.array(img, dtype=np.uint8)
    img=np.array(img, dtype=np.uint8)
    #cv2.imshow("imgbef", img)
    #img=cv2.bitwise_not(img)

    #cv2.imshow("img", img)
    #cv2.waitKey()

    # Read the image as a grayscale image
    #img = cv2.imread('A://testimg5.jpg', 0)
    ## Threshold the image
    #ret,img = cv2.threshold(img, 127, 255, 0)
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break

    # Displaying the final skeleton
    #cv2.imshow("Skeleton",skel)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return skel
def closeedges(dilated): 
    superkernel= np.ones((14,14),np.uint8)
    skeleton=skeletonxd(dilated)
    closinggraym=cv2.dilate(skeleton,kernel2,iterations = 25)
    closinggray2 = cv2.morphologyEx(closinggraym, cv2.MORPH_CLOSE, superkernel)
    return closinggray2