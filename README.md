# Exploradome_tangram
Tangram form detection from live video stream

The tangram is a dissection puzzle consisting of seven flat polygons, called tans, which are put together to form shapes. 
The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap. 

The 12 shapes are:

![image](https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS)

boat(bateau), bowl(bol), cat(chat), heart(coeur), swan(cygne), rabbit(lapin), house(maison), hammer(marteau), mountain(montagne), bridge(pont), fox(renard), turtle(tortue)

## Objective

The objective of this project is to train a model to recognize in real time the realization of tangram (record in live and make by children) and to make predictions on the realized shapes.

Here we will use a model for the multiclass image classification by using a pre-trained TensorFlow 2 (using transfert learning framework).

## Table Of Contents
-  [Installation and Usage](#Installation-and-Usage)
-  [Dataset Creation](#Dataset-Creation)
-  [Model Creation](#Model-Creation)
  -  [Transfer learning](#Transfer-learning)
-  [Getting Started](#Getting-Started)
-  [Command Line Args Reference](#Command-Line-Args-Reference)
-  [References](#References)
-  [Team](#Team)

# Dataset Creation

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
│   └── tangram_mobilenetv2.h5
│   └── tangram_inceptionv3.h5
│
│ 
├── collab Notebooks  - this folder contains any model and preprocessing of your project.
│   └── trigram_model_v1.ipynb
│   └── trigram_model_v2.ipynb
│   
└──
```

## Dataset
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
├──  test  
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
for the train and validation with all images (unbalanced dataset)

- train_balanced: [train_balanced](https://drive.google.com/file/d/1V_rKMpjhHeJHRY0YcShYBZeun1uTz_G0/view?usp=sharing)
for the train and validation with 140 images for each categories (balanced dataset)

- test_full: [test_full](https://drive.google.com/file/d/15EB3UGwrMkUzZvJIlf6uxeXYeDUtFhXf/view?usp=sharing)
for the test with all images (unbalanced dataset)

- test_balanced: [test_balanced](https://drive.google.com/file/d/13tTo7ue3HUGeQXfq4aj215EZIEvHXs0M/view?usp=sharing)
for the test with 28 images for each categories (balanced dataset) = 20% of train_balanced dataset

## Trigram Preprocessing
Data Augmentation (applied on dataset):
- Contrast changes (factor = 0.5 #darkens the image or 1.5 #brightens the image)
- Blurring (applied after contrast change)
=> already in folder train_full and test_full

ImageDataGenerator with TensorFlow (applied on model):
- Rescale: 1./255 is to transform every pixel value from range [0,255] -> [0,1]
- Split train_full or train_balanced dataset to train and validation dataset (= 30% of train dataset)

# Model Creation
## Transfer learning
**What is Transfer Learning?**
Transfer learning is a machine learning technique in which a network that has already been trained to perform a specific task is repurposed as a starting point for another similar task. 

**Transfer Learning Strategies & Advantages**
There two transfer learning strategies, here we use:
   - Initialize the CNN network with the pre-trained weights
   - We then retrain the entire CNN network while setting the learning rate to be very small, which ensures that we don't drastically change the trained weights
   
The advantage of transfer learning is that it provides fast training progress since we're not starting from scratch. Transfer learning is also very useful when you have a small training dataset available, but there's a large dataset in a similar domain (i.e. ImageNet).

**Using Pretrained Model**
There are 2 ways to create models in Keras. Here we used the sequential model.
The sequential model is a linear stack of layers. You can simply keep adding layers in a sequential model just by calling add method. 

The two pretrained models used are: 
* [MobileNet](https://keras.io/api/applications/mobilenet/)
* [InceptionV3 + L2](https://keras.io/api/applications/inceptionv3/)

**Transfer Learning with Image Data**
It is common to perform transfer learning with predictive modeling problems that use image data as input.

This may be a prediction task that takes photographs or video data as input.

For these types of problems, it is common to use a deep learning model pre-trained for a large and challenging image classification task such as the [ImageNet](http://www.image-net.org/) 1000-class photograph classification competition.

These models can be downloaded and incorporated directly into new models that expect image data as input.

## Apply Transfer Learning
```python

```

## Results or improvement strategy

See the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

# Getting Started

## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)

```bash
pip install opencv-python tensorflow
```

## Configuration
**Trigram Model**

To use the model, open a new terminal and copy this link:

```
wget -O model.h5 'https://drive.google.com/uc?export=download&id=13dDtd4jsCyA6Z4MEPK3RsWDLiCZJvEPc'
```


# Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD](https://github.com/jenguehard)
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
