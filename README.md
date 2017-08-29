# Planet - Understanding the Amazon from Space

## 1. Introduction

The description of this Kaggle competition can be found by clicking the [link](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).

The target of the competition is to detect 17 types objects in satellite data to track the human footprint in the Amazon rainforest. Code in this repository is a solution for the object detection in satellite images.

## 2. Dataset

The training set consists of **40479** images, while there are **61191** images in testing set.

The dimension of image id (256, 256, 3).

Each image that used for training has several labels. Totally, **17** labels are involved as shown in next section.

### 2.1 Labels

<table>
  <tr>
    <td>clear</td>
    <td>water</td>
    <td>agriculture</td>
    <td>habitation</td>
    <td>artisinal_mine</td>
    <td>conventional_mine</td>
  </tr>
  <tr>
    <td>haze</td>
    <td>cloudy</td>
    <td>blooming</td>
    <td>blow_down</td>
    <td>bare_ground</td>
    <td>selective_logging</td>
  </tr>
  <tr>
    <td>road</td>
    <td>primary</td>
    <td>cultivation</td>
    <td>slash_burn</td>
    <td>partly_cloudy</td>
    <td></td>
  </tr>
</table>

### 2.2 Images

Here are several samples with their labels.

Image | Labels | Image | Labels
------|--------|-------|-------
<img src="https://github.com/quqixun/Planet/blob/master/Images/1.jpg" width="60"> | clear primary | <img src="https://github.com/quqixun/Planet/blob/master/Images/5.jpg" width="60"> | agriculture clear primary road
<img src="https://github.com/quqixun/Planet/blob/master/Images/2.jpg" width="60"> | haze primary water | <img src="https://github.com/quqixun/Planet/blob/master/Images/6.jpg" width="60"> | agriculture partly_cloudy primary
<img src="https://github.com/quqixun/Planet/blob/master/Images/3.jpg" width="60"> | clear habitation road | <img src="https://github.com/quqixun/Planet/blob/master/Images/7.jpg" width="60"> | agriculture clear habitation primary road water
<img src="https://github.com/quqixun/Planet/blob/master/Images/4.jpg" width="60"> | agriculture clear primary road | <img src="https://github.com/quqixun/Planet/blob/master/Images/8.jpg" width="60"> | agriculture cultivation habitation partly_cloudy primary road

### 2.3 Augmentation

An example of image augmentation for one image.

Original | 90° | 180° | 270° | Flip H | Flip V
---------|-----|------|------|--------|-------
<img src="https://github.com/quqixun/Planet/blob/master/Images/7.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_90.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_180.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_270.jpg" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_flipH.png" width="60"> | <img src="https://github.com/quqixun/Planet/blob/master/Images/7_flipV.png" width="60">

## 3. Models

### 3.1 Simple Convolutional Neural Network

<img src="https://github.com/quqixun/Planet/blob/master/Images/cnn.png" width="350">

### 3.2 Combination of Residual Network and Pyramid Network

Read papers in reference 1 and 2 to get more details.

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

* **planet_labels.py** : convert string labels to binary list and save it into file;
* **planet_tfrecords.py** : read all images in training set into tfrecords file for training and validating;
* **planet_model.py** : construct models and helper functions for training and testing;
* **planet_train.py** : run the training process;
* **planet_test.py** : run the testing process.

### 4.3 Usage

<pre>
python planet_labels.py       (generate binary labels)
python planet_tfrecords.py    (generate tfrecords files)
python planet_train.py        (train the model)
python planet_test.py         (detect object in testing images)
</pre>

## 5. Evaluation

F2 score of detection result is 0.90473, which ranked at 504 over 938. The best result of this competition is 0.93318.

This is my first Kaggle competition, I am satisfied with the result under existing conditions.

## 6. Discussion

After this competition, I deeply realized the importance of a good GPU and larger RAM. I have no GPU avaible to accelerate the compution. Thus, three troubles occured:

* **Low computation speed** : It took 3 days to finish 6 epoches of the second model.
* **Low resolution input** : The dimension of original image is (256, 256, 3). However, as to the limitation of my computer's RAM, only images with the resolution (32, 32, 3) can be used to train the model.
* **Simple network** : It is quite difficult to apply a deeper network since the low computation speed and the low resolution input.

Some other tricks:

* **Augmentation** : GAN can be used to do image augmentation to enlarge training set.
* **TTA** : Test time augmentation could improve the accuracy of detection (see reference 3).

Though I did not obtain a good score, the experience and the skill I learnt from this competition is quit useful.

## 7. References

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
3. [Test data augmentation : generating new test data from existing test data](http://web4.cs.ucl.ac.uk/staff/S.Yoo/papers/Yoo2008it.pdf)
