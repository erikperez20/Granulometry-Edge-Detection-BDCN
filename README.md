## Rock detection in blasting images using a Bi-Directional Cascade Network (BDCN)

This project uses a Bi-Directional Cascade Network(BDCN) based on the repository https://github.com/pkuCactus/BDCN 
to detect edges in rock blasting images to obtain the granulometry study of the fragmented material

### Prerequisites

- pytorch >= 1.3.1
- numpy >= 1.11.0
- pillow >= 3.3.0

## Clone this repository to local
```shell
git clone https://github.com/pkuCactus/BDCN.git
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

![Image](</Data/Imagen_Prueba/IMG_9371.JPG>)

