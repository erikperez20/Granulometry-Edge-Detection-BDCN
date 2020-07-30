"""
Script with necessary functions to do some filtering to images in the post process stage
"""
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from skimage import filters
import numpy as np
import argparse
import imutils
import cv2
import csv
import curves
import math
import nonmaxsup
import random
import time
import edge_detection_model
import os
import transform_images


# # Cambios
# import argparse


# parser = argparse.ArgumentParser('Image Processing')

# parser.add_argument('--img_file',type=str,help = 'image file')
# args = parser.parse_args()

# image_file = args.img_file


scale = 1 #pixels per meter #100.5/7.5 para 2.png(imagen de prueba 18/02/2020)
farscale = 3
closescale = 1

def midpoint(ptA, ptB):
	"""
	Calculates the midpoints between two points.
	"""
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


#################################################################################
###################        Apply filters Function        ########################
#################################################################################


def filters(image_file):
	###########   load the image, convert it to grayscale, and blur it slightly
	original1=cv2.imread(image_file,0)
	image = cv2.imread(image_file)
	image = np.array(image)
	original1 = np.array(original1)

	########    Apply gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	###########    Kernel Sizes
	kernel = np.ones((5,5),np.uint8)
	kernel1 = np.ones((3,3),np.uint8)
	kernel2 = np.ones((7,7), np.uint8)
	kernel3 = np.ones((1,1),np.uint8)
	kernel4 = np.zeros((5,5),np.uint8)

	###########    Elliptical, rectangular and cross kernels
	kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# Rectangular vertical
	kernel_rect_v = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
	# Rectangular horizontal
	kernel_rect_h = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
	# Rectangular diagonal 45
	kernel_diag = kernel4.copy()
	np.fill_diagonal(kernel_diag,1)
	# Rectangular antidiagonal 45
	kernel_antidiag = kernel4.copy()
	np.fill_diagonal(np.fliplr(kernel_antidiag),1)
    # Rectangular diagonal 45
	kernel_diag2 = kernel1.copy()
	np.fill_diagonal(kernel_diag2,1)
	# Rectangular antidiagonal 45
	kernel_antidiag2 = kernel1.copy()
	np.fill_diagonal(np.fliplr(kernel_antidiag2),1)

	########    Apply the first filter(median blur 5x5) after obtaining the segmented image
	median1 = cv2.medianBlur(gray, 5)
	########    Apply erosion and skeletonize the image
	erode1 = cv2.erode(median1, kernel, iterations = 1)
	skeleton1 = nonmaxsup.skeletonize(erode1)
	########   Apply an equalization function
	equ1 = cv2.equalizeHist(skeleton1)

    # New Filter
	b1=-10
	c1= 80
	bright1 = transform_images.apply_brightness_contrast(equ1,b1,c1)

	#######    Apply opening and closing
	closing1 = cv2.morphologyEx(bright1, cv2.MORPH_CLOSE , kernel1)

	# Dilatamos horizontalmente, verticalmente, diagonal y antidiagonal
	dilate1 = cv2.dilate(closing1, kernel_rect_v , iterations = 1 )
	dilate2 = cv2.dilate(dilate1 , kernel_rect_h ,iterations = 1)

    #New Filter
	b2 = -30
	c2 = 80
	bright2 = transform_images.apply_brightness_contrast(dilate2,b2,c2)
	dilate3 = cv2.dilate(bright2, kernel_diag , iterations = 1 )
	dilate4 = cv2.dilate(dilate3, kernel_antidiag, iterations = 1 )
    # New filter
	b2 = -10
	c2 = 80
	bright3 = transform_images.apply_brightness_contrast(dilate4,b2,c2)
	erode2 = cv2.erode(bright3 , kernel , iterations = 1)
	skeleton2 = nonmaxsup.skeletonize(erode2)
	dilate5 = cv2.dilate(skeleton2, kernel2 , iterations = 1)
	closing2 = cv2.morphologyEx(dilate5 , cv2.MORPH_CLOSE, kernel)
	dilate6 = cv2.dilate(closing2, kernel_diag2 , iterations = 1)
	dilate7 = cv2.dilate(dilate6, kernel_antidiag2 , iterations = 1)
	erode3 = cv2.erode(dilate7, kernel_ellipse , iterations = 1)
	ret, binary = cv2.threshold(erode3.copy(), 0, 255 , cv2.THRESH_BINARY)
	''' return binary image values only with 0 or 255 '''
	return binary

##############################################################################
##############################################################################

############      Contour detection and rock size determination     ##########

##############################################################################
##############################################################################


# file_resultados : file donde se almacenaran las imagenes de contornosm data y la grafica
# name_img: nombre de la imagen de referencia
def count_rocks(edgedimage , file_resultados , name_img ):
	"""
	Function to count every rock in the binary image after applying some filters to close contours because
	we have just sticked 64 images
	"""

	############ Far and close scale, those parameters will indicate the scale at the top of the image and at the bottom of the image
	farscale=1
	closescale=3


	MaxSizeForFines=30 # This is to get the percentage of fines in the image

	############ Reading and then applying some filters in order to close some countours because the image is divided in 64 parts

	image1=cv2.imread(str(edgedimage))

	image1 = np.float32(image1)
	kernelxd = np.ones((3,3),np.uint8)

	image=cv2.dilate(image1,kernelxd,iterations=1)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelxd)
	ret,edges=cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),50,255,cv2.THRESH_BINARY)

	row, column, depth = image1.shape
	blank_image = np.zeros((row,column,3), np.uint8)
	random.seed(4)

	############ Detecting countours in the binary image
	cnts= cv2.findContours(np.uint8(edges.copy()), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	dellist=[]
	cnts = cnts[0]

	# sort the contours from left-to-right and initialize the
	(cnts, _) = contours.sort_contours(cnts)
	SizeList=[] #creating the list for every size of every object
	WeightList=[] #creating the list for every wheight of every object
	Px=[] # creating the P(x) list
	TotalWeight=0 # This parameter will increase each time a rock has been detected

	count=0
	cy=1
	preminx=0
	preyforminx=0

	for c in cnts:

		meanValue , stand  = getContourStat(cnts , edges , count ) # Erasing a bit of sound in the image

		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		dim=dA
		area=(cv2.contourArea(c)) #measuring area

		if area>0: # If area exists

			############ Calculating the y value for the centroid of the contour
			M=cv2.moments(c)
			centroid_y=float(M['m01']/M['m00'])

			relation=(dA*dB*math.pi/4)/area #measuring the relation between actualarea and rectangular area of the contour
			filter=0.002 #2
			filter2=950 #95
			if dA<row/filter and dB<row/filter and dA<column/filter and dB<column/filter\
			and dA>row/filter2 and dA>column/filter2 and dB>row/filter2 and dB>column/filter2: # Filtering particles too small or too big
				if relation<100: # Filtering irregular sized rocks #1.34
					if any(i>100 for i in stand.flatten()) or any(j>100 for j in meanValue.flatten()): #100
						myname="hola"
					else:
						# compute the size of the object
						scale=CalculateScale(centroid_y, farscale,closescale,row)
						dimA = dA / scale
						dimB = dB / scale
						SizeList.append((4*area/math.pi)**(1/2))  #adding size to a list of sizes
						weight=(area/math.pi)**(3/2)*(4/3)*math.pi
						WeightList.append(weight) #adding the weight to the weightlist
						TotalWeight=TotalWeight+weight # increasing the totalweight
						blank_image=cv2.drawContours(blank_image,cnts,count,(random.randint(1,255),random.randint(1,255),random.randint(1,255)),-1)
		count=count+1

	SizeList.sort()
	numberofparticles=len(SizeList)
	print("there are {} rock particles ".format(numberofparticles))
	print(count-numberofparticles," rocks were eliminated")
	acumulatedweight=0

	############ Adding the accumulated weights to a list, with that we have the size list and acc. weight list needed to create the curves
	for rock in WeightList:
		acumulatedweight=acumulatedweight+rock
		Px.append(100*(acumulatedweight/TotalWeight))

	############ Getting the diverse parameters from every curve
	gXc,gP80,gerror,gfines , yGaudin , xGaudin = curves.GaudinSchuhmannCurve(Px,SizeList,MaxSizeForFines)
	n,rXc,rP80,rerror,rfines , yRosin , xRosin = curves.RosinRammlerCurve(Px,SizeList,MaxSizeForFines)
	sXc,sP80,serror,sfines , ySwebrec , xSwebrec = curves.SwebrecCurve(Px,SizeList,MaxSizeForFines,n)

	############ Plotting the curves to be given as a result
	fig, ax = plt.subplots(figsize = (11,9))

	ax.plot(SizeList , Px , linewidth = 2 , c=(0.1,0.2,0.4) , label = "Data")
	ax.scatter(SizeList , Px , c=(0.1,0.2,0.4) , s = 9)
	ax.plot(yGaudin,xGaudin, color="b", label="Gaudin Schuhmann",linewidth = 2)
	ax.plot(yRosin,xRosin, color="r", label="Rosin Rammler",linewidth = 2)
	ax.plot(ySwebrec,xSwebrec, color="g", label="Swebrec",linewidth = 2)

	plt.hist(SizeList, bins=12,alpha=0.6, ec="black",weights=100*np.ones(numberofparticles) / numberofparticles)

	ax.set_xlabel('x (cm)', fontsize = 'large')
	ax.set_ylabel('P(x) (%)', fontsize = 'large')
	ax.legend()

	############ Creating result files

	carpeta = os.path.join(file_resultados,name_img)
	edge_detection_model.make_dir(carpeta)

	############ Saving the graph

	graph_name = 'grafica_' + name_img + '.png'
	graph_name = os.path.join(carpeta, graph_name)
	fig.savefig(graph_name)

	############ Saving the contour image

	contours_file = 'contornos_'+ name_img + '.jpg'
	contours_file = os.path.join(carpeta,contours_file)
	cv2.imwrite(contours_file, blank_image)

	############ Saving the parameters in a .txt file

	info_file = 'data_' + name_img + '.txt'
	info_file = os.path.join(carpeta,info_file)
	f1 = open(info_file,"w+")

	f1.write("Curve name\t\tXc(cm)\t\t\tP80(cm)\t\tError(r**2)\t\tFines(%)" + '\n')
	f1.write("%s:\t%f\t%s\t%f\t%s" % ("Gaudin Schuhmann", gXc, float(gP80),  gerror,gfines) + '\n')
	f1.write("%s:\t%f\t%s\t%f\t%s" % ("Rosin Rammler    ", rXc, float(rP80), rerror,rfines) + '\n')
	f1.write("%s:\t%f\t%s\t%f\t%s" % ("Swebrec         ", sXc, sP80, serror,sfines) + '\n' )
	f1.write("*The error is based on the coefficient of determination and the closer to one, the less error it presents" + '\n')
	f1.write("*If there is a negative error, it indicates that the curve has a very high error" + '\n')
	f1.write('Number of Rocks: ' + str(numberofparticles))

def getContourStat(contour,image , idx ):
	"""
	Function necessary in a filter while counting contours
	"""
	mask = np.zeros(image.shape,dtype="uint8")
	cv2.drawContours(mask, contour , idx, 255, -1)
	mean,stddev = cv2.meanStdDev(image,mask=mask)

	return mean , stddev

def CalculateScale(y_value, farscale, closescale,row):
	"""
	Calculates the scale for the given y value
	"""
	return closescale-((row-y_value)*(closescale-farscale)/row)
