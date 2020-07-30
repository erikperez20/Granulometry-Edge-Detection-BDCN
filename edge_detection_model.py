''' In this script we use the BDCN model to detect edges in rock blast fragmentation images  '''

'''  Import necessary modules and libraries  '''
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import cv2
import bdcn
# from datasets.dataset import Data
import argparse
import os
import os.path as osp

''' Make directory function '''
def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
# Default Value for Padding Image    
pad_val = 15

''' Test images function takes the model, cuda, results directiory, the data directory and a .lst file with all 
    locations of every image to analize '''
def test(Model, Cuda , ResDir, DataRoot, TestLst):
    test_root = DataRoot
    ''' Find each image direction '''
    if TestLst is not None:
        with open(osp.join(test_root, TestLst), 'r') as f:
            test_lst = f.readlines()
        test_lst = [x.strip() for x in test_lst]
        if ' ' in test_lst[0]:
            test_lst = [x.split(' ')[0] for x in test_lst]
    else:
        test_lst = os.listdir(test_root)
    save_dir = ResDir #agregue yo
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    make_dir(save_dir)
    
    if Cuda:
        Model.cuda()
    Model.eval()
    ''' Each element of the test_lst is the direction of each image, it loops through each one and opens image '''
    for nm in test_lst:

        direction = osp.join(test_root,nm)
        data = cv2.imread(direction + '.jpg')
        # data = cv2.resize(data, (data.shape[1]/2, data.shape[0]/2), interpolation=cv2.INTER_LINEAR)
        
        # Prepare data for model 
        data = np.array(data, np.float32)
        data -= mean_bgr
        data = data.transpose((2, 0, 1))       
        data = torch.from_numpy(data).float().unsqueeze(0)
        if Cuda:
            data = data.cuda()
        data = Variable(data)
        
        # Model use
        out = Model(data)
        out = [torch.sigmoid(x).cpu().data.numpy()[0, 0, :, :] for x in out]
        #out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
        img = out[-1]
        
        h,w = img.shape
        # Real height and width values since we paddle the image before processing through the model: 
        h1 = h-pad_val*2
        w1 = w-pad_val*2
        Matrix = img[pad_val:pad_val+h1 , pad_val:pad_val+w1]
        
        # Linux y Windows
        
        if '/' in nm:
            nm = nm.split('/')[-1]
            direction_list = direction.split('/')
        elif '\\' in nm:
            nm = nm.split('\\')[-1]
            direction_list = direction.split('\\')

        #direction_list = direction.split('/')
        dire = direction_list[-2]
        new_path = os.path.join(save_dir,dire)
        make_dir(new_path)
        new_file = os.path.join(new_path , '%s.png'%direction_list[-1])
        print(new_file)
        
        # Save file and out image
        cv2.imwrite(new_file , 255*Matrix)


''' Main function to excecute the whole script '''        
def execute(gpu,cuda,trained_model,res_dir,data_root,test_lst):
    
    # Import model and use gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    model = bdcn.BDCN()
    
    #####  Using GPU  ####
    if cuda:
        model.load_state_dict(torch.load('%s' % (trained_model)))
    else:
    #### Using CPU   ####
        model.load_state_dict(torch.load('%s' % (trained_model),map_location=torch.device('cpu')))
    
    # Call test function and detect edges
    test(model, cuda , res_dir, data_root,test_lst )
