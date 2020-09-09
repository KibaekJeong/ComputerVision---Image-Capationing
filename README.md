# CVND---Image-Captioning-Project

## Project Overview
In following project, objective is to build an network for image captioning. Using CNN encoder and decoder consisting of LSTM units, caption for each image is estimated. 
Originally, Udacity requires student to build with basic LSTM. However, I wanted to go deeper and develop a better model with Attention based LSTM. So, the project repo is divided into two different section. One with using LSTM and another with using Attention based LSTM.

## Result
Two different network with LSTM decoder and attetion LSTM decoder has been developed and compared. Attention based LSTM definately was able to achieve lower loss value, but required much longer training time.
Both networks were able to output reasonable captioning for images. 

Here are some of the output from the network.
- LSTM Decoder:

![image](https://github.com/KibaekJeong/ComputerVision---Image-Capationing/blob/master/output/1.png)
![image](https://github.com/KibaekJeong/ComputerVision---Image-Capationing/blob/master/output/2.png)



- Attention LSTM Decoder:

![image](https://github.com/KibaekJeong/ComputerVision---Image-Capationing/blob/master/output/3.png)
![image](https://github.com/KibaekJeong/ComputerVision---Image-Capationing/blob/master/output/4.png)


## Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).
