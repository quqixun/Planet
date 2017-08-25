# Planet - Kaggle Competition

## 1. Introduction

## 2. Dataset

### 2.1 Labels

Labels: clear, cloudy, partly_cloudy, haze, primary, water, habitation,
agriculture, road, cultivation, bare_ground, slash_burn, selective_logging,
blooming, conventional_mine, artisinal_mine, blow_down.

### 2.2 Images

Image | Labels | Image | Labels
------|--------|-------|-------
<img src="https://github.com/quqixun/Planet/blob/master/Images/1.jpg" width="60"> | clear primary | <img src="https://github.com/quqixun/Planet/blob/master/Images/5.jpg" width="60"> | agriculture clear primary road
<img src="https://github.com/quqixun/Planet/blob/master/Images/2.jpg" width="60"> | haze primary water | <img src="https://github.com/quqixun/Planet/blob/master/Images/6.jpg" width="60"> | agriculture partly_cloudy primary
<img src="https://github.com/quqixun/Planet/blob/master/Images/3.jpg" width="60"> | clear habitation road | <img src="https://github.com/quqixun/Planet/blob/master/Images/7.jpg" width="60"> | agriculture clear habitation primary road water
<img src="https://github.com/quqixun/Planet/blob/master/Images/4.jpg" width="60"> | agriculture clear primary road | <img src="https://github.com/quqixun/Planet/blob/master/Images/8.jpg" width="60"> | agriculture cultivation habitation partly_cloudy primary road

### 2.3 Augmentation

Original | 90° | 180° | 270° | Flip H | Flip V
---------|-----|------|------|--------|-------
<img src="https://github.com/quqixun/Planet/blob/master/Images/7.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_90.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_180.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_270.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_flipH.png" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_flipV.png" width="60">

## 3. Models

### 3.1 Simple Convolutional Neural Network

<img src="https://github.com/quqixun/Planet/blob/master/Images/cnn.png" width="350">

### 3.2 Combination of Residual Network and Pyramid Network

<img src="https://github.com/quqixun/Planet/blob/master/Images/res.png" width="900">
<img src="https://github.com/quqixun/Planet/blob/master/Images/res_part.png" width="900">

## 4. Implementation

### 4.1 Dependencies

* tqdm 4.14.0
* numpy 1.12.1
* pandas 0.20.1
* opencv3 3.1.0
* imutils 0.4.3
* tensorlayer 1.4.5
* scikit-learn 0.18.1
* tensorflow 1.2.1 (compiled from source)

### 4.2 Code Orgnization

### 4.3 Usage

## 5. Evaluation

## 6. Discussion

## 7. References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
