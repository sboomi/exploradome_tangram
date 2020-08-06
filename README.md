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

## 1. Video recording
To create the dataset our image classification, we need to have images with label of each tangram categorie.
To do this, we filmed continuously members of our team performing in turn the 12 shapes possibles
For this, we using the camera provided by exploradome to respect the conditions under which the algorithm will be used.

## 2. Image dataset preparation
For the image dataset preparation, we need to cut video into photos.
For cutting the video into images, we proceeded in this ways:
- with python script, the video is splitting in photo (every 1 second)

## 3. Images labeling
TensorFlow requires the dataset to be provided in the following directory structure:
Like this, that why each photo is order in folder with the name of categorie :
```
├──  multilabel_data  
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
├── 
```
We have already created the dataset in this format and provided a download link (and some instructions) in the GitHub repository. 

## 4. Initial Dataset

The initial dataset is unbalanced between categorie. 
We didn't split already the dataset between training data and testing before applying data augmentation.

| Label           |  Total images | 
|-----------------|------|
|boat(bateau)     | 716  | 
| bowl(bol)       | 248  |  
| cat(chat)       | 266  | 
| heart(coeur)    | 273  |  
| swan(cygne)     | 321  |  
| rabbit(lapin)   | 257  |  
| house(maison)   | 456  |  
| hammer(marteau) | 403  |  
| mountain(montagne)  |  573 |  
| bridge(pont)    | 709  |  
| fox(renard)     | 768  |  
| turtle(tortue)  | 314  |  
| TOTAL           | 5304 | 

## 5. Data augmentation
Having a large dataset is crucial for the performance of the deep learning model.
Data augmentation is a strategy to increase the diversity of data available for training models, without actually collecting new data.

For our dataset we applied different types images augmentations to obtain more images.

Data Augmentation with python script:
- Contrast changes (1.5 #brightens the image) with PIL and ImageEnhance.Brightness()
- Blurring (applied after contrast change) with OpenCV and cv2.gaussianblur() 

ImageDataGenerator with TensorFlow:
- Rescale: 1./255 is to transform every pixel value from range [0,255] -> [0,1]
- Split train_full or train_balanced dataset to train and validation dataset (= 30% of train dataset)

```python
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True,
                    validation_split=0.3)
```

| Label           |  Before Data Augmentation  |   After Data Augmentation* | 
|-----------------|---------------|----------------|
| boat(bateau)    | 716           |   2148         | 
| bowl(bol)       | 248           |   744          | 
| cat(chat)       | 266           |   800          | 
| heart(coeur)    | 273           |   820          | 
| swan(cygne)     | 321           |   964          | 
| rabbit(lapin)   | 257           |   772          | 
| house(maison)   | 456           |   1368         | 
| hammer(marteau) | 403           |   1209         | 
| mountain(montagne)  |  573      |   1720         | 
| bridge(pont)    | 709           |   2128         | 
| fox(renard)     | 768           |   2304         |  
| turtle(tortue)  | 314           |   942          |  
| TOTAL           | 5304          |   15919        | 

* with script python

Next step, we created a balanced datasets. 
For each categorie we keep randomly:
- 400 images for the training dataset 
- 80 images (20% of 400) for the test dataset 

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
- dataset: [train_balanced](https://drive.google.com/file/d/1upNDIpsRwdO8O08SnUJEfez4dJYMnQkS/view?usp=sharing)
for the train and validation with 400 images for each categories and for the test dataset with 80 images(balanced dataset)

# Model Creation
## Transfer learning
**What is Transfer Learning?**
Transfer learning is a machine learning technique in which a network that has already been trained to perform a specific task is repurposed as a starting point for another similar task. 

**Transfer Learning Strategies & Advantages:**
There two transfer learning strategies, here we use:
   - Initialize the CNN network with the pre-trained weights
   - We then retrain the entire CNN network while setting the learning rate to be very small, which ensures that we don't drastically change the trained weights
   
The advantage of transfer learning is that it provides fast training progress since we're not starting from scratch. Transfer learning is also very useful when you have a small training dataset available, but there's a large dataset in a similar domain (i.e. ImageNet).

**Using Pretrained Model:**
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
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

```python
inception = InceptionV3(weights='imagenet', include_top=False)
```

## Results or improvement strategy

See the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

# Getting Started

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
