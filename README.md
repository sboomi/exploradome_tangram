# Exploradome_tangram
Tangram form detection from live video stream

## Table Of Contents
-  [Installation and Usage](#Installation-and-Usage)
-  [Usage](#Usage)
-  [Configuration](#Configuration)
-  [In Details](#in-details)
-  [Team](#Team)

## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)

```bash
pip install opencv-python tensorflow
```

## Approach taken

Find the best accuracy with transfert learning model (CNN with Tensorflow) - see the Google Sheet

## In progress

Tested so far:
* MobileNet
* InceptionV3 + L2

## Results or improvement strategy

See the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

## Configuration

## In Details
```
├──  data  - here's the image classification datasets
│    └── train_full  - for the train and validation with all images (unbalanced).
│    └── train_balanced - for the train and validation with 140 images for each categories (balanced).
│    └── test_full  		- for the test with all images (unbalanced).
│    └── test_balanced  - for the test with 28 images for each categories (balanced) - 20% of train_balanced dataset.
│   
│
│
├──  modules  - this file contains the modules.
│    └── get_img_from_webcam.py  - here's the file to extract images of video cam, split in two, predict 
│                                  => output with pred of each categorie.
│
├── saved_model  - this folder contains any customed layers of your project.
│   └── 
│   └──
│
│ 
├── collab Notebooks  - this folder contains any model and preprocessing of your project.
│   └── trigram_model_v1.ipynb
│   └── trigram_model_v2.ipynb
│   
└── main.py  - this foler contains unit test of your project.
```

### Dataset
![image](https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS)

The dataset has the following directory structure:

```
├──  train  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├──  validation  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│   
└── 
```
To download the file:
- train_full: [train_full](https://drive.google.com/file/d/18RoZgzSzTE6nzHCzzMuDl9h4RktS3rNo/view?usp=sharing)
- train_balanced: [train_balanced](https://drive.google.com/file/d/1V_rKMpjhHeJHRY0YcShYBZeun1uTz_G0/view?usp=sharing)
- test_full: [test_full](https://drive.google.com/file/d/15EB3UGwrMkUzZvJIlf6uxeXYeDUtFhXf/view?usp=sharing)
- test_balanced: [test_balanced](https://drive.google.com/file/d/13tTo7ue3HUGeQXfq4aj215EZIEvHXs0M/view?usp=sharing)

### Trigram Preprocessing

### Trigram Model

To use the model, open a new terminal and copy this link:

```
wget -O model.h5 'https://drive.google.com/uc?export=download&id=13dDtd4jsCyA6Z4MEPK3RsWDLiCZJvEPc'
```

## Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD](https://github.com/jenguehard)
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
