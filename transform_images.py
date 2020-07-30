
import numpy as np
import os 
import cv2
import glob



# img_file = "C:/Users/erik_/Documents/erik documents/MS/Data Fragmentacion e Imagenes/7.- Fragmentacion_Julio_16"
# new
#img_file = "C:/Users/erik_/Documents/erik documents/MS/Data Fragmentacion e Imagenes/Julio_Imagenes_Selec/"


'''     Brightness and contrast level function  '''

'''  in -> (image, brigthness_level, contrast_level)
     out-> image with modified contrast and brightness '''
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    ''' Brightness '''
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    ''' Contrast '''
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    #out = cv2.bilateralFilter(buf,40,75,75)

    return buf

## File: Imagen_Prueba 
##     Files: IMG_0000.jpg , .JPG , .png, .PNG

''' Cut image in 64 pieces or 4 pieces depending on it's size '''
dim_dict = {}
def cut_img(img_file):
    
    # Windows
    if '\\' in img_file:
        if img_file[-1] != '\\':
            img_file = img_file + '\\'
        list_carpet = img_file.split('\\')
    # Linux
    elif '/' in img_file:
        if img_file[-1] != '/':
            img_file = img_file + '/'
        list_carpet = img_file.split('/')


    string = list_carpet[-2]
    list_carpet.pop(-1)
    list_carpet.pop(-1)

    # Linux
    if '/' in img_file:
        lst_directory = '/'.join(list_carpet)
    # Windows
    elif '\\' in img_file:
        lst_directory = '\\'.join(list_carpet)

    lst = string + '.lst'
    file_lst = os.path.join(lst_directory,lst)
    f = open(file_lst,"w+")
    
    for filename in os.listdir(img_file):
        # Concatenamos los archivos
        file_name = os.path.join(img_file,filename)

        # Si encontramos un archivo .JPG entonces lo analizamos 
        if os.path.isfile(file_name):
            if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.png') or \
            filename.endswith('.PNG'):
                print(file_name)
                img = cv2.imread(file_name)
                img = np.array(img)
                gray1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                H , W , D = img.shape

                # Valor de padding
                pad_val = 15 

                # Particicion de imagen 
                # Dim^2 = numero de imagenes 
                # If size of H or W is less than 1000 cut in 4 pieces
                if H < 1000 or W < 1000:
                    Dim = 2
                # Else in 64 pieces
                else:
                    Dim = 8
                carpet = file_name.replace('.JPG','')
                fs = filename.replace('.JPG','')
                dim_dict[fs] = Dim

                # Linux y Windows

                if '/' in img_file:
                    c = '/'
                    positions3 = [pos for pos, char in enumerate(carpet) if char == c]
                elif '\\' in img_file:
                    c = '\\'
                    positions3 = [pos for pos, char in enumerate(carpet) if char == c]

                if not os.path.exists(carpet):
                    os.mkdir(carpet)
                    print("Directory " , carpet ,  " Created ")
                else:    
                    print("Directory " , carpet ,  " already exists")

                k = 1
                for i in range(Dim):
                    for j in range(Dim):
                        # Extract the number of each image name file
                        num = f'{fs}_{k}.jpg'
                        num2 = f'{fs}_{k}'
                        f.write(os.path.join(carpet[positions3[0]+1:],num2) + '\n')
                        print(os.path.join(carpet[positions3[0]+1:],num2))
                        
                        #We extract each piece of the array in a for loop
                        array_img = img[int(H/Dim)*i:int(H/Dim)*(i+1),int(W/Dim)*j:int(W/Dim)*(j+1)]
                        
                        # Apply filters 
                        out = apply_brightness_contrast(array_img, -30 , 50)
                        median = cv2.medianBlur(out,3)

                        # We do padding, reflection so after passing through the model it does'nt generate edges
                        # at the borders of each piece of the image:
                        matrix4 = np.pad(median , pad_width = pad_val, mode = 'reflect' )
                        matrix5 = matrix4[:,:,pad_val:pad_val+D]

                        cv2.imwrite(os.path.join(carpet,num),matrix5)

                        k+=1
    
    '''Returns lst with all the images files in it and the lst directory where the .lst archive is in 
       Both are strings '''
    
    return lst, lst_directory, dim_dict
