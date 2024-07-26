# SegNet Implementation with TensorFlow 2

This repository contains an implementation of the SegNet deep convolutional neural network architecture for semantic pixel-wise image segmentation using TensorFlow 2. The implementation is provided in a Jupyter Notebook (`SegNet_Implementation.ipynb`). The dataset used for training and evaluation is stored in a Google Drive link, and a video explaining the article and the implementation is included.

## SegNet

SegNet is a deep convolutional neural network architecture specifically designed for semantic pixel-wise image segmentation. It aims to be efficient in terms of memory and computational time while still providing high-quality segmentation results.

### Architecture

SegNet consists of three main components:
1. **Encoder Network**: 
   - The encoder network is composed of 13 convolutional layers, identical to the VGG16 network.
   - Each convolutional layer is followed by a batch normalization layer and a ReLU activation.
   - Max-pooling is performed after some convolutional layers to reduce the spatial dimensions of the feature maps. The pooling indices (locations of maximum values) are stored for use in the decoder network.

2. **Decoder Network**:
   - The decoder network consists of corresponding upsampling layers for each max-pooling layer in the encoder.
   - Each upsampling layer uses the stored pooling indices to perform non-linear upsampling, ensuring accurate reconstruction of the input image's spatial dimensions.
   - After upsampling, convolutional layers refine the feature maps to produce dense, high-resolution outputs.

3. **Pixel-wise Classification Layer**:
   - The final layer is a softmax classifier that assigns a class label to each pixel in the input image.
   - This produces a segmented output where each pixel is classified into one of the predefined categories.

### Key Features

- **Efficient Memory Usage**: By reusing pooling indices from the encoder during the decoding process, SegNet reduces the memory overhead typically associated with large decoder networks.
- **High-Quality Segmentation**: The architecture preserves spatial resolution and reduces the loss of boundary information, resulting in accurate segmentation outputs.
- **Simplicity and Speed**: SegNet is designed to be both simple and fast, making it suitable for real-time applications.

### Applications

SegNet is widely used in various applications, including:
- Autonomous driving (road, vehicle, and pedestrian segmentation)
- Medical imaging (organ and tissue segmentation)
- Environmental monitoring (land cover classification)
- Industrial inspection (defect detection)

### Further Reading

For more detailed information, you can refer to the original paper: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561).

## Implementation

The implementation of SegNet using TensorFlow 2 is provided in the Jupyter Notebook: `SegNet_Implementation.ipynb`.

### Dataset

The dataset used for training and evaluation is stored in Google Drive. You need to upload the dataset folder to your Google Drive.you can find dataset from here: (https://drive.google.com/drive/folders/15xYx78EgD0TYFSDrsYpOBoWIg9jLc0xf?usp=sharing)

### Video Explanation

A detailed video explaining the article and the implementation can be found here:
Video Explanation Link: (https://drive.google.com/file/d/12Yyssij54xAImIDTCzVE-IYRhbN4B8Pl/view?usp=sharing) 

## Running the Notebook

To run the Jupyter Notebook, follow these steps:
1. upload dataset folder to your google drive. 
2. clone the repository in google colab
3. select GPU device
4. run the code

