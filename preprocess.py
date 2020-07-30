''' This script executes the edge detection and saves results in a directory '''

''' Import necessary modules and libraries '''
import transform_images
import edge_detection_model
import argparse
import os

''' Necessary inputs for processing images
    1. Images Files: File where cropped images are stored by name
    2. Cuda use to activate gpu
    3. GPU to use, default 0
    4. Pretrained BDCN model to use ''' 
def parse_args():

    parser = argparse.ArgumentParser('Read Fragmentation Data Images to obtain edge detected image')
    parser.add_argument( '--images_file' , type = str , help = 'File where images are stored' )
    parser.add_argument('-c', '--cuda', action='store_false',help='use --cuda if using in cpu, else nothing')
    parser.add_argument('-g', '--gpu', type=str, default='0',help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='bdcn_pretrained_on_bsds500.pth',help='the model to test')

    return parser.parse_args()

''' Main Function to excecute edge_detection_model2 and transform_images scripts '''
def main():

    args = parse_args()
    
    ''' Takes images_file and generates the cropted images with the directions saved in a .lst file and the 
        dimension(dim) of croped image.'''
    images_file = args.images_file
    lst, lst_dir , dim = transform_images.cut_img(images_file)
    
    ''' Result directory for rock edge images '''
    # Linux
    if '/' in images_file:
        result_dir = images_file.split('/')[-1]
    # Windows
    elif '\\' in images_file:
        result_dir = images_file.split('\\')[-1]    
    
    result_dir = result_dir + '_Result'
    
    ''' Calls edge_detection_model2 to generate rock edge images '''
    edge_detection_model.execute(args.gpu , args.cuda , args.model , result_dir , lst_dir , lst)
    
    return result_dir , dim

if __name__ == '__main__':
    main()
