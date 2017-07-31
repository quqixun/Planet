# Kaggle Competition: Planet
# Created by Qixun Qu
# quqixun@gmail.com


import os
import numpy as np
import pandas as pd


class PlanetLabels():

    def __init__(self):
        '''__INIT__

            Initialization of the instance.

        '''

        self.data_num = 0            # the number of data samples
        self.labels_num = 0          # the number of labels
        self.full_labels = []        # a list contains all labels
        self.train_str_labels = []   # a dataframe contains string labels
        self.train_num_labels = []   # a dataframe contains number labels

        return

    def read_full_labels(self, path):
        '''READ_FULL_LABELS

            Read labels from file "full_labels.csv".
            In this project, there are 17 labels that can be
            divided into 3 groups, which are:

            1. Weather: clear, cloudy, partly_cloudy, haze
            2. Common Scenes: primary, water, habitation,
            agriculture, road, cultivation, bare_ground
            3. Less Comman Scenes: slash_burn, selective_logging,
            blooming, conventional_mine, artisinal_mine, blow_down

            Input argument:

            - path : the file path where "full_labels.csv" is

        '''

        self.full_labels = pd.read_csv(path).values.flatten().tolist()
        self.labels_num = len(self.full_labels)

        return

    def read_train_str_labels(self, path):
        '''READ_TRAIN_TXT_LABELS

            There are several string labels for each patch, for example:
            train_0: haze primary
            train_1: agriculture clear primary water

            Input argument:

            - path : the path of the csv file which contains string labels

        '''

        self.train_str_labels = pd.read_csv(path)
        self.data_num = self.train_str_labels.shape[0]

        return

    def str_to_num(self, path):
        '''STR_TO_NUM

            Convert string labels to numbers, for example:
            train_0:
                str: haze primary
                num: 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            train_1:
                str: agriculture clear primary water
                num: 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0

            Store the number labels into file 'train_v2n.csv'.

            Input argument:

            - path : the path of the new label files

        '''

        # Initialize an array to keep number labels
        self.train_num_labels = np.zeros((self.data_num, self.labels_num))

        # If a label is included in a sample, the corresponding index of
        # the sample will be set to 1, otherwise 0
        for i in range(self.data_num):  # self.data_num
            label_temp = self.train_str_labels.iloc[i]['tags'].split()

            for j in range(len(label_temp)):
                idx = self.full_labels.index(label_temp[j])
                self.train_num_labels[i, idx] = 1

        # Write number labels into csv file
        df = pd.DataFrame(self.train_num_labels, columns=self.full_labels)
        image_name = self.train_str_labels['image_name'].tolist()
        df.insert(0, column='image_name', value=image_name)
        df.to_csv(path, index=False)

        return


if __name__ == '__main__':
    # Create the instance
    pl = PlanetLabels()

    # Load all labels
    fl_path = os.getcwd() + '\\Dataset\\full_labels.csv'
    pl.read_full_labels(fl_path)

    # Load labels of train patches
    tl_path = os.getcwd() + '\\Dataset\\train_v2.csv'
    pl.read_train_str_labels(tl_path)

    # Convert string labels to number labels
    nl_path = os.getcwd() + '\\Dataset\\train_v2n.csv'
    pl.str_to_num(nl_path)
