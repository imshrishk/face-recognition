very lite face recog

# Low-Compute in-Browser Face Recognition 

This is a face recognition system utilising the [MobileNet v4 small](https://arxiv.org/abs/2404.10518) architecture from [timm](https://github.com/huggingface/pytorch-image-models), trained over the open [Celebrity Face Dataset](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset/). We utilise [ArcFace loss](https://arxiv.org/abs/1801.07698) from the [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library for improved deep metric learning. 

After training the model as a classifier, we remove the final classification layer and use the model as a feature extractor.

For each input face, we generate 960-dimensional vector embeddings which we compare using cosine similarity against a known database of embeddings. If the maximum similarity is greater than a set threshold, we state that the person is identified, else unknown.

We deploy our model using TensorFlow.js. This can be accessed [**here**](https://3.38.107.244).

## Tree

```
├───dataset
│       cleaning.py|ipynb
│       download.py|ipynb
│       train-val-test.ipynb
│
├───face-extraction
│       face-extraction.py|ipynb
│
├───model
│       train.py
│       checkpoints/
│       runs/
│
└───webpage
    │   dump-embeddings.ipynb
    │   index.html
    │   model-conversion.ipynb
    │   script.js
    │   style.css
    │
    ├───employee_embeddings
    │       embeddings.json
    │
    ├───haar-cascade
    │       haarcascade_frontalface_default.xml
    │
    └───model
            group1-shard1of2.bin
            group1-shard2of2.bin
            model.json
```

We now descend into the functionality of our codebase in the order they're in our directory tree.

### **dataset/**
Contains files related to our dataset. We download the dataset and clean it here.
- **download.py**: Automated method of downloading the dataset files and unzipping them. 
- **cleaning.py**: Script for cleaning the dataset -- removing invalid images.
- **train-val-test.ipynb**: Splitting the dataset into training, validation, and test sets.

### **face-extraction/**
Contains scripts for extracting faces from images.
- **face-extraction.py**: For detecting and cropping out faces using ```dlib/face-recognition``` library. Note that we ignore all images having multiple faces.

### **model/**
Contains files related to model training, checkpoints, logs.
- **train.py**: Script for training the face recognition model.
- **checkpoints/**: Checkpoint files.
- **runs/**: TensorBoard training logs.

### **webpage/**
Contains files for deploying the model as a web-app.
- **dump-embeddings.ipynb**: Notebook for generating and dumping face embeddings into embeddings.json from image dataset format.
- **model-conversion.ipynb**: Jupyter notebook for converting the trained torch model to TensorFlow.js format.
- **index.html**: HTML file for the web application's front-end.
- **style.css**: CSS file for styling the web application.
- **script.js**: Main backbone for handling the web application's logic, including loading the video stream, haar-cascade for extracting face, model for inference and performing face recognition using cosine similarity.

#### **employee_embeddings/**
Contains precomputed face embeddings for known individuals.
- **embeddings.json**: JSON file storing the embeddings of known faces.

#### **haar-cascade/**
Contains Haar Cascade [config](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) from OpenCV for face detection.
- **haarcascade_frontalface_default.xml**: XML file for the Haar Cascade classifier used for face detection.

#### **model/**
- **model.json**: JSON file describing the TensorFlow.js model architecture.
Contains the converted TensorFlow.js model graph and paths to the weight binaries.
- **group1-shard1of2.bin**: Binary file part 1 of the TensorFlow.js model.
- **group1-shard2of2.bin**: Binary file part 2 of the TensorFlow.js model.
