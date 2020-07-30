"""
Script necessary to do the whole process
example: python rock_detect.py --images_file Data\Imagen_Prueba --cuda
If you are using a virtual machine, use:
python rock_detect.py --images_file Data\Imagen_Prueba

"""
import granulometry_stats
import os
import argparse
import paste_images
import cv2
import matplotlib.pyplot as plt
import preprocess
import time
import edge_detection_model
import re

def main():
    
    #Run all the Granulometry process
    #example: python rock_detect.py --images_file Data\Imagen_Prueba --cuda
    #If you are using a virtual machine, use:
    #python rock_detect.py --images_file Data\Imagen_Prueba
    
    
    start_time = time.time()

    # Call the preprocess script and import the result model images directory
    process_dir , dictionary =  preprocess.main()

    print('Dictionary: ', dictionary)

    # image_dir = args.res_dir

    ''' Create a directory of the filtered images after passing them though the model '''
    # Windows
    if "\\" in process_dir:
        divisions = image_dir.split("\\")
        filtradas_dir = divisions[-1]+'_filtradas'
    # Linux
    elif '/' in process_dir:
        divisions = image_dir.split("/")
        filtradas_dir = divisions[-1]+'_filtradas'
    else:
        filtradas_dir = process_dir + '_filtradas'


    ###   Contour images file  #####

    ifile = re.sub('\\_Result$', '', process_dir)
    ifile = ifile + '_Contours_Info_Graph'
    edge_detection_model.make_dir(ifile)

    ''' Loop through the images in the result file '''
    for filename in os.listdir(process_dir):
        file_name = os.path.join(process_dir,filename)
        if os.path.isdir(file_name):
            print("file_name: ",filename)
            for filename2 in os.listdir(file_name):

                file = os.path.join(file_name,filename2)
                print(file)

                # We apply the filters function to highlight the edges and close contours
                out = granulometry_stats.filters(file)

                # Create the directory to save images
                file_out = os.path.join(filtradas_dir,filename)
                edge_detection_model.make_dir(file_out)
                file_out1 = os.path.join(file_out,filename2)

                # Save each image in the filtered directory
                cv2.imwrite(file_out1, out)


            #print("file_name: ", file_name, "dictionary", dictionary[filename])
            filename_out = os.path.join(filtradas_dir,filename)

            # We paste the images using the paste function in paste_images module
            img_reconstruida_name , binary = paste_images.paste(filename_out , dictionary[filename])

            # Save the reconstructed image in the filtered directory
            img_filtrada = os.path.join(filtradas_dir,img_reconstruida_name)
            cv2.imwrite(img_filtrada , binary)

            # We call the granulometry_stats module and use the count_rocks function to generate the image with contours, granulometry graph
            # and the data .txt
            granulometry_stats.count_rocks(img_filtrada,ifile,filename)

    print('Overall taken time:  ', time.time() - start_time)



if __name__ == '__main__':
    main()
