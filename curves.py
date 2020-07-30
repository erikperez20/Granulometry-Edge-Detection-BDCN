import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import optimize

def GaudinSchuhmannCurve(Pxt,size,fines):
	"""
	Function to get the Gaudin Schuhmann curve for the fiven points

	"""
	# adding the x and y values to the linear
	# form of the Gaudin-Schuhmann equation
	x = [[0 for i in range(1)] for j in range(len(size))]
	y = [[0]*1]*len(Pxt)

	for i in range(0,len(size)):
		x[i][0]=math.log(size[i],10)

	for j in range(0,len(Pxt)):
		y[j]=math.log(Pxt[j]/100,10)

	# calculating the parameters n and Xc using linear regression
	# since with this method we can calculate the parameters
	# linealizating the equation


	# Model initialization
	regression_model = LinearRegression()
	# Fit the data(train the model)
	regression_model.fit(x, y)

	# printing the parameters calculated
	n=regression_model.coef_
	Xc=10**(-(regression_model.intercept_)/n)
	P80=GaudinSchuhmannFormula(80,n,Xc)
	print("####GAUDIN SCHUHMANN PARAMETERS####")     #print("80% passing: ",P80) #P80 or F(80)
	print('n:' ,n) #uniformity index
	#print('Xc:',Xc ) #mean particle size


	y_pred1=[]

	for i in range(0,len(size)):
		#print (type(Xc), type(n), type(sizexd), type(y_pred1))
		#print(sizexd,"size/xc/n", Xc," ", n)
		y_pred1.append(100*((size[i]/float(Xc))**float(n)))
	error=r2_score(Pxt,y_pred1)
	#print("Pxt: ", Pxt)
	#print("y_pred: ", y_pred1)
	#print("coeficient of determination (r**2): ", error)

	fines=(100*((fines/float(Xc))**float(n)))

	# creaing a graph that will follow this curve

	xx = np.array(range(100))
	yy = GaudinSchuhmannFormula(xx,n,Xc)

	# plotting the graph created with blue color

	# Yo lo comente

	# plt.plot(yy,xx, color="b", label="Gaudin Schuhmann")


	return Xc,P80,error,fines, yy , xx

def GaudinSchuhmannFormula(P,n,Xc):
	"""
	Calculate x (mesh opening) for the given P with the given n and Xc parameters
	"""

	x=Xc*((P/100)**(1/n))
	return x


def RosinRammlerCurve(Pxt,size,fines):
	"""Function to get the Rossin Rammler curve
	"""

	# adding the x and y values to the linear
	# form of the Gaudin-Schuhmann equation
	x = [[0 for i in range(1)] for j in range(len(size)-1)]
	y = [[0]*1]*(len(Pxt)-1)

	for i in range(0,len(size)-1):
		x[i][0]=math.log(size[i],math.e)

	for j in range(0,len(Pxt)-1):
		y[j]=math.log(-math.log(1-(Pxt[j]/100),math.e),math.e)

	# calculating the parameters n and Xc by using
	# linear regression like with the rosin rammler curve

	# Model initialization
	regression_model = LinearRegression()
	# Fit the data(train the model)
	regression_model.fit(x, y)

	# printing the parameters
	n=regression_model.coef_
	Xc=math.e**(-(regression_model.intercept_)/n)
	P80=RosinRammlerFormula(80,n,Xc)
	print("####ROSIN RAMMLER PARAMETERS####")

	print('n:' ,n) #uniformity index
	#print('Xc:',Xc ) #mean particle size
	#print("80% passing: ",P80) #calculating the P80

	y_pred2=[]

	for i in range(0,len(size)):
		y_pred2.append(100*(1-(math.e**(-((size[i]/float(Xc))**float(n))))))

	#print("pxt: ", Pxt)
	#print("y_pred2: ", y_pred2)

	error=r2_score(Pxt,y_pred2)
	#print("coeficient of determination (r**2): ", error)

	fines=(100*(1-(math.e**(-((fines/float(Xc))**float(n))))))

	#creating a set of points to plot the graph
	xxx = np.array(range(100))
	yyy = RosinRammlerFormula(xxx,n,Xc)

	#Yo lo comente
	#plt.plot(yyy,xxx, color="r", label="Rosin Rammler")

	return n,Xc,P80,error,fines , yyy, xxx #getting the n value because we will use it for the swebrec function

def RosinRammlerFormula(P,n,Xc):
	"""
	Calculate the x (mesh opening) for the given P with the given n and Xc parameters
	"""
	x=Xc*((-(np.log(1-(P/100))))**(1/n))
	return x

def SwebrecCurve(Pxt,size,fines,n):
	""" Function to plot the swebrec function for the given set of points
	"""

	x = [[0 for i in range(1)] for j in range(len(size))]

	errors=0 #counting the number of errors(times the b parameter does not exist for a pair of points)

	Xm=size[len(size)-1] # the maximum particle size

	a=2*n*math.log(2) # creating a constant value for later

	blist=[] # creating a list for the b values to later get an average

	for i in range(0,len(size)): # looping in all the pair of P(x) and x data points
		x=size[i]
		P=Pxt[i]
		b=n
		if P>0: # if we are working with P=0 it will throw an error

			base=(100/P)-1

			def FunctionForb(b):
				#creating a function with the variables b, Px and x


				result=(b/a)*(base**(1/b))-math.log(Xm/x)

				return result

			def FunctionForb_der1(b):
				return ((base**(1/b))*(b-math.log(base)))/(a*b)

			def FunctionForb_der2(b):
				return ((((math.log(base))**2)*(base**(1/b)))/(a*(b**3)))


			try:
				b=optimize.newton(FunctionForb,fprime=FunctionForb_der1,fprime2=FunctionForb_der2,x0=b) # function to calculate the best fit for b
			except:
				errors=errors+1 # Number of times the b value isnt found
			else:
				blist.append(b) # adding the b value to the list
	#print(blist)
	b=cal_average(blist)# averaging the list and taking a representative b value

	Xc=Xm/(math.e**(b/a))
	P80=SwebrecFormula(80,b,Xc,Xm)   #P80
	y_pred3=[]

	for i in range(0,len(size)):
		y_pred3.append(100/(1+((math.log(float(Xm)/size[i])/math.log(float(Xm)/float(Xc)))**float(b))))
	error=r2_score(Pxt,y_pred3) # Calculating the error

	fines=(100/(1+((math.log(float(Xm)/fines)/math.log(float(Xm)/float(Xc)))**float(b))))



	# creating a set of points to plot the graph in color green

	Sx = np.array(range(99))
	Sx=Sx+0.0001
	Sy = SwebrecFormula(Sx,b,Xc,Xm)
	return Xc,P80,error,fines , Sy , Sx

def SwebrecFormula(P,b,Xc,Xm):
	"""Getting the mesh size as with the other curves
	"""

	logarith=math.log(float(Xm)/float(Xc))
	root=((100/P)-1)**(1/float(b))
	return float(Xm)/(math.e**(logarith*root))



def cal_average(num):
	"""
	Function to get the average of a list
	"""
	sum_num = 0
	for t in num:
		sum_num = sum_num + t
	avg = sum_num / len(num)
	return avg
