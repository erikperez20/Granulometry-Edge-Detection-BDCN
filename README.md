## Rock detection in blasting images using a Bi-Directional Cascade Network (BDCN)

This project uses a Bi-Directional Cascade Network(BDCN) based on the repository https://github.com/pkuCactus/BDCN 
to detect edges in rock blasting images to obtain the granulometry study of the fragmented material.

### Prerequisites

- pytorch >= 1.3.1
- numpy >= 1.11.0
- pillow >= 3.3.0
- opencv >= 4.1.2
- sklearn >= 0.21.3
- scipy >= 1.4.1
- imutils >= 0.5.3
- skimage >= 0.15.0
## Clone this repository to local
```shell
git clone https://github.com/erikperez20/Granulometry-Edge-Detection-BDCN.git
```

### Pretrained models

BDCN model for BSDS500 dataset and NYUDv2 datset of RGB and depth are availavble on Baidu Disk.

    The link https://pan.baidu.com/s/18PcPQTASHKD1-fb1JTzIaQ
    code: j3de

## Usage 
```
python rock_detect.py
```
Options
```
  -h, --help            show this help message and exit
  --images_file IMAGES_FILE
                        File where images are stored (ex. Data\Imagen_Prueba)
  -c, --cuda            use --cuda if using in cpu, else nothing
  -g GPU, --gpu GPU     the gpu id to run net
  -m MODEL, --model MODEL
                        the model to test (defaults to bdcn_pretrained_on_bsds500.pth)
```
## Results
Rock blasting input image taken from mine dataset

![Image](</Data/Imagen_Prueba/IMG_2078.JPG>)

Rock detection results after applying BDCN model, morphological transformations and image thresholding.

![Image2](</Imagen_Prueba_Contours_Info_Graph/IMG_2078/contornos_IMG_2078.jpg>)

Rock sizes distribution considering a scale/size = 1, Gaudin Schuhmann rock distribution, Rosin Rammler distribution and Swebrec distribution.

![Image3](</Imagen_Prueba_Contours_Info_Graph/IMG_2078/grafica_IMG_2078.png>)
