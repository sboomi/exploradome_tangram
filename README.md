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
- [OpenCV] (https://opencv.org/) (Open Computer Vision Library)

```bash
pip install 
```

## Usage

```python
```

## Configuration

## In Details
```
├──  data  
│    └── test  - here's the image classification datasets (from video_to_img).
│    └── train - here's the file to train dataset.
│    └── validation  		 - here's the file to validation dataset.
│    └── video_to_img    - here's the file of raw image extraction of video file.
│    └── WIN_20200727_16_30_12_Pro.mp4    - here's the tangram video for the creation of the datasets.
│
├──  modules        - this file contains the modules.
│    └── get_img_from_webcam.py  - here's the file to extract images of video cam, split in two.
│ 
│
├── saved_model      - this folder contains any customed layers of your project.
│   └── 
│   └──
│
├── collab Notebooks - this folder contains any model of your project.
│   └── example_model.py
│   └── build.py
│   └── build.py
│   └── build.py
│   └── lr_scheduler.py
│   
└── main.py					- this foler contains unit test of your project.
```
### Dataset
```
├──  test
│    └── defaults.py  - here's the default config file.
│
│
├──  train  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  validation  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  video_to_img
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
└── 
```

### Trigram Preprocessing

### Trigram Model

## Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD]()
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
