'''  Paste cropted image after edge detecting script '''

'''Import necessary modules and libraries'''
import numpy as np
import os 
import cv2
 
''' Define paste function '''
# Takes the image file and the crop dimension
def paste(images_files , dim):
    # List to save all images from IMG_number file
    # We divide the image in 64 or 4 pieces or 8x8 or 2x2:
    images2 = list(range(dim**2))
    
    # As images are loop in disorder we order them and save it in images2 list
    for filename in os.listdir(images_files):
        file_name = os.path.join(images_files,filename)
        if os.path.isfile(file_name):
            name = file_name
            num = file_name.split("_")[-1]
            if '.png' in num:
                num = num.split(".")[0]
            # Save each piece of the image in order of num, exp: IMG_9342_40 is stored in position 40 in list
            images2[int(num)-1] = file_name
            new_image=' ' # flag
    
    if new_image==' ':
        
        im = cv2.imread(images2[0],1)
        im = np.array(im)
        h,w = im.shape[:2]
        
        # Original size of image before cropping
        H = h*dim
        W = w*dim
        
        # Creates a list containing W columns and H rows, all set to 0
        Matrix = np.array([[0 for x in range(W)] for y in range(H)])
        k=0

        for i in range(dim):
            for j in range(dim):
                img = cv2.imread(images2[k],0)
                img = np.array(img)
                Matrix[h*i : h*(i+1) , w*j : w*(j+1)] = img
                k+=1

        #print(filename.split("_")[:2])        
        #new_name = '_'.join(filename.split("_")[:2])

        if '/' in images_files:
            new_name = images_files.split("/")[-1]
        elif '\\' in images_files:
            new_name = images_files.split("\\")[-1]

        imagen_reconstruida_name = '%s.jpg'%new_name
        #cv2.imwrite(imagen_reconstruida_name,Matrix)
        
        ''' Returns the reconstructed name image and the image array '''
        return imagen_reconstruida_name, Matrix

