# Kaggle Competition: Planet
# Created by Qixun Qu
# quqixun@gmail.com


import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from planet_model import PlanetModel


def test_result(model, model_path, batch_size, img_size, data_dir, resu_dir):
    pm = PlanetModel(model=model, mode='test')

    labels_path = data_dir + 'train_v2n.csv'
    pm.read_labels(labels_path)

    full_labels_path = data_dir + 'full_labels.csv'
    pm.read_full_labels(full_labels_path)

    train_path = data_dir + 'train-jpg\\'
    test1_path = data_dir + 'test-jpg\\'
    test2_path = data_dir + 'test-jpg-additional\\'

    print("\nTest processing for {} model.".format(model))
    with tf.variable_scope('model') as scope:
        tl.layers.set_name_reuse(True)

        print("\nTest processing on testing data 1.")
        test1_labels, test1_probs = \
            pm.test_model(test1_path, model_path, batch_size, img_size, 'testset1', resu_dir)

        print("\nTest processing on testing data 2.")
        scope.reuse_variables()
        test2_labels, test2_probs = \
            pm.test_model(test2_path, model_path, batch_size, img_size, 'testset2', resu_dir)

    test_labels = np.vstack((test1_labels, test2_labels))
    test_probs = np.vstack((test1_probs, test2_probs))

    return pm, test_labels, test_probs


def check_inputs(input1, input2):
    if input1.shape[0] != input2.shape[0]:
        print("Two inputs should be same in number of samples.")
        return 1

    if input1.shape[1] != input2.shape[1]:
        print("Two inputs should be same in number of classes.")
        return 1

    return 0


def vote_result(test1_labels, test2_labels, vote=1):
    if check_inputs(test1_labels, test2_labels):
        print("Please check inputs.")
        return

    test_labels = (test1_labels + test2_labels)
    test_labels = (test_labels >= vote) * 1.

    return test_labels.astype(int)


def weighted_result(test1_probs, test2_probs, thresh1=.2, thresh2=.2, weights=[.5, .5]):
    if check_inputs(test1_probs, test1_probs):
        print("Please check inputs.")
        return

    test_probs = test1_probs * weights[0] + test2_probs * weights[1]
    threshold = thresh1 * weights[0] + thresh2 * weights[1]

    test_labels = (test_probs > threshold) * 1.

    return test_labels.astype(int)


if __name__ == '__main__':
    work_dir = os.getcwd()
    data_dir = work_dir + '\\Dataset\\'
    resu_dir = work_dir + '\\Result\\'
    batch_size = 128

    img_size = 64
    cnn_model_path = resu_dir + 'cnn_model.npz'
    cnn_pm, cnn_test_labels, cnn_test_probs = \
        test_result('cnn', cnn_model_path, batch_size, img_size, data_dir, resu_dir)
    cnn_pm.num_to_str(resu_dir, cnn_test_labels, 'cnn_result')

    img_size = 32
    res_model_path = resu_dir + 'res_model.npz'
    res_pm, res_test_labels, res_test_probs = \
        test_result('res', res_model_path, batch_size, img_size, data_dir, resu_dir)
    res_pm.num_to_str(resu_dir, res_test_labels, 'res_result')

    vote1_test_labels = vote_result(cnn_test_labels, res_test_labels, 1)
    res_pm.num_to_str(resu_dir, vote1_test_labels, 'cnn_and_res_vote1_result')

    vote2_test_labels = vote_result(cnn_test_labels, res_test_labels, 2)
    res_pm.num_to_str(resu_dir, vote2_test_labels, 'cnn_and_res_vote2_result')

    weighted_test_labels = weighted_result(cnn_test_labels, res_test_labels,
                                           cnn_pm.threshold, res_pm.threshold, [.3, .7])
    res_pm.num_to_str(resu_dir, weighted_test_labels, 'cnn_and_res_weighted_result')
