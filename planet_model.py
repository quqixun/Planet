# Kaggle Competition: Planet
# Created by Qixun Qu
# quqixun@gmail.com


import os
import cv2
import time
from tqdm import *
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from sklearn.metrics import fbeta_score
from tensorflow.contrib import keras as tfk
from planet_tfrecords import PlanetTFRecords as PTFR


class PlanetModel():

    PRINT_FREQ = 20

    def __init__(self, ptfr=None, model='cnn', mode='train'):
        self.model = model

        if mode == 'train':
            self.ptfr = ptfr
            self.labels_num = ptfr.labels_num
        elif mode == 'test':
            self.ptfr = PTFR()
            self.train_imgs = []
            self.test_imgs = []
            self.labels = []
            self.full_labels = []
            self.data_num = 0
            self.labels_num = 0
            self.threshold = 0
        else:
            print("Wrong mode!")

        return

    def net_input(self, input_var, name='input'):
        return InputLayer(inputs=input_var, name=name)

    def conv2d(self, net, shape, padding='SAME', name='conv'):
        if self.model == 'cnn':
            w = tfk.initializers.glorot_uniform()
            b = tf.constant_initializer(value=0.0)
            act = tf.nn.relu
        elif self.model == 'res':
            sd = 1 / np.sqrt(np.prod(shape[0:3]) * self.labels_num)
            w = tf.truncated_normal_initializer(stddev=sd)
            b = tf.constant_initializer(value=1e-12)
            act = tf.identity
        else:
            print("Wrong model! `cnn` or `res`.")
            return

        return Conv2dLayer(net, act=act, shape=shape, strides=[1, 1, 1, 1],
                           padding=padding, W_init=w, b_init=b, name=name)

    def dense(self, net, n_units, act=tf.nn.relu, name='ds'):
        if self.model == 'cnn':
            w = tfk.initializers.glorot_uniform()
        elif self.model == 'res':
            w = tf.truncated_normal_initializer(stddev=1 / n_units)
        else:
            print("Wrong model! `cnn` or `res`.")
            return

        return DenseLayer(net, n_units=n_units, act=act, W_init=w, name=name)

    def batch_normal(self, net, name='bn'):
        if self.model == 'cnn':
            act = tf.identity
        elif self.model == 'res':
            act = tf.nn.relu
        else:
            print("Wrong model! `cnn` or `res`.")
            return

        return BatchNormLayer(net, act=act, is_train=True, name=name)

    def max_pool(self, net, name='mp'):
        return PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         pool=tf.nn.max_pool, name=name)

    def add(self, t1, t2, name='sum'):
        return ElementwiseLayer([t1, t2], combine_fn=tf.add, name=name)

    def up_sample(self, net, name='up'):
        return UpSampling2dLayer(net, size=[2, 2], name=name)

    def global_max_pool(self, net, shape, name='gmp'):
        max_pool_shape = [1, shape[0], shape[1], 1]
        return PoolLayer(net, ksize=max_pool_shape, strides=max_pool_shape,
                         pool=tf.nn.max_pool, name=name)

    def feature_extraction(self, net, shape, n_units=256, name='fe'):
        net = self.global_max_pool(net, shape, name=name + '_gmp')
        net = FlattenLayer(net, name=name + '_flt')
        net = self.dense(net, n_units=n_units, name=name + '_ds')

        return net

    def res_block(self, net, shape, name='res'):
        net = self.conv2d(net, [1, 1, shape[0], shape[1]], name=name + '_conv1')
        net = self.batch_normal(net, name=name + '_bn1')
        net = self.conv2d(net, [3, 3, shape[1], shape[1]], name=name + '_conv2')
        net = self.batch_normal(net, name=name + '_bn2')
        net = self.conv2d(net, [1, 1, shape[1], shape[2]], name=name + '_conv3')
        net = self.batch_normal(net, name=name + '_bn3')

        return net

    def find_f_thresh(self, y, y_out, interval=0.001):
        ts = np.arange(0, 1, interval)
        ts_num = len(ts)

        fs = np.zeros(ts_num)
        for i in range(ts_num):
            fs[i] = fbeta_score(y, y_out > ts[i], beta=2, average='samples')

        best_f = np.max(fs)

        return best_f, ts[np.where(fs == best_f)[0]][0]

    def cnn_net(self, x):
        print("\n---------- CNN NET ----------\n")

        net = self.net_input(x, name='cnn_ip')
        net = self.batch_normal(net, name='cnn_ip_bn')
        net = self.conv2d(net, [3, 3, 3, 32], name='cnn_conv1')
        net = self.conv2d(net, [3, 3, 32, 32], padding='VALID', name='cnn_conv2')
        net = self.max_pool(net, name='cnn_mp1')
        net = DropoutLayer(net, keep=0.25, name='cnn_dp1')

        net = self.conv2d(net, [3, 3, 32, 64], name='cnn_conv3')
        net = self.conv2d(net, [3, 3, 64, 64], padding='VALID', name='cnn_conv4')
        net = self.max_pool(net, name='cnn_mp2')
        net = DropoutLayer(net, keep=0.25, name='cnn_dp2')

        net = self.conv2d(net, [3, 3, 64, 128], name='cnn_conv5')
        net = self.conv2d(net, [3, 3, 128, 128], padding='VALID', name='cnn_conv6')
        net = self.max_pool(net, name='cnn_mp3')
        net = DropoutLayer(net, keep=0.25, name='cnn_dp3')

        net = self.conv2d(net, [3, 3, 128, 256], name='cnn_conv7')
        net = self.conv2d(net, [3, 3, 256, 256], padding='VALID', name='cnn_conv8')
        net = self.max_pool(net, name='cnn_mp4')
        net = DropoutLayer(net, keep=0.25, name='cnn_dp4')

        net = FlattenLayer(net, name='cnn_flt')
        net = self.dense(net, n_units=512, name='cnn_ds')
        net = self.batch_normal(net, name='cnn_ds_bn')
        net = DropoutLayer(net, keep=0.5, name='cnn_dp5')
        net = self.dense(net, n_units=self.labels_num, act=tf.identity, name='cnn_out')

        return net

    def res_net(self, x):
        print("\n---------- RES NET ----------\n")

        net_in = self.net_input(x, name='ip')

        net_in = self.conv2d(net_in, [1, 1, 3, 16], name='conv1')
        net_in = self.batch_normal(net_in, name='bn1')
        net_in = self.conv2d(net_in, [1, 1, 16, 16], name='conv2')
        net_in = self.batch_normal(net_in, name='bn2')
        net_in = self.conv2d(net_in, [1, 1, 16, 16], name='conv3')
        net_in = self.batch_normal(net_in, name='bn3')

        # #1 32 by 32: 16 ==> 64
        net_bk1 = self.res_block(net_in, [16, 32, 64], name='block1')
        net_sc1 = self.conv2d(net_in, [1, 1, 16, 64], name='short1')
        net_rs1 = self.add(net_sc1, net_bk1, name='resid1')

        # #2 32 by 32: 64 ==> 128
        net_bk2 = self.res_block(net_rs1, [64, 64, 128], name='block2')
        net_sc2 = self.conv2d(net_rs1, [1, 1, 64, 128], name='short2')
        net_rs2 = self.add(net_sc2, net_bk2, name='resid2')
        net_mp1 = self.max_pool(net_rs2, name='res_mp1')

        # #3 16 by 16: 128 ==> 256
        net_bk3 = self.res_block(net_mp1, [128, 128, 256], name='block3')
        net_sc3 = self.conv2d(net_mp1, [1, 1, 128, 256], name='short3')
        net_rs3 = self.add(net_sc3, net_bk3, name='resid3')
        net_mp2 = self.max_pool(net_rs3, name='mp2')

        # #4 8 by 8: 256 ==> 256
        net_bk4 = self.res_block(net_mp2, [256, 256, 256], name='block4')
        net_rs4 = self.add(net_mp2, net_bk4, name='resid4')
        net_up1 = self.up_sample(net_rs4, name='up1')

        # #5 16 by 16: 256 ==> 128
        net_sm1 = self.add(net_up1, net_rs3, name='sum1')
        net_bk5 = self.res_block(net_sm1, [256, 128, 128], name='block5')
        net_up2 = self.up_sample(net_bk5, name='up2')

        # #6 32 by 32: 128 ==> 64
        net_sm2 = self.add(net_up2, net_rs2, name='sum2')
        net_bk6 = self.res_block(net_sm2, [128, 64, 64], name='block6')

        # #7 32 by 32: 64 ==> 16
        net_sm3 = self.add(net_bk6, net_rs1, name='sum3')
        net_bk7 = self.res_block(net_sm3, [64, 32, 16], name='block7')

        # Feature extraction
        net_fe1 = self.feature_extraction(net_bk1, [32, 32], n_units=256, name='fe1')
        net_fe2 = self.feature_extraction(net_bk2, [32, 32], n_units=256, name='fe2')
        net_fe3 = self.feature_extraction(net_bk3, [16, 16], n_units=256, name='fe3')
        net_fe4 = self.feature_extraction(net_bk4, [8, 8], n_units=256, name='fe4')
        net_fe5 = self.feature_extraction(net_bk5, [16, 16], n_units=256, name='fe5')
        net_fe6 = self.feature_extraction(net_bk6, [32, 32], n_units=256, name='fe6')
        net_fe7 = self.feature_extraction(net_bk7, [32, 32], n_units=256, name='fe7')

        net = [net_fe1, net_fe2, net_fe3, net_fe4, net_fe5, net_fe6, net_fe7]
        net = ConcatLayer(net, name='concate_layer')
        net = DropoutLayer(net, keep=0.5, name='drop')
        net = self.dense(net, n_units=self.labels_num, act=tf.identity, name='output')

        return net

    def optimize_step(self, x, y, bs, lr):
        if self.model == 'cnn':
            net = self.cnn_net(x)
        elif self.model == 'res':
            net = self.res_net(x)
        else:
            print("Wrong model! `cnn` or `res`.")
            return

        y_out = tf.reshape(net.outputs, shape=[bs, self.labels_num])
        prob = tf.nn.sigmoid(y_out)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                                      logits=y_out))
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        return net, y_out, prob, loss, train_step

    def train_model(self, train_path, valid_path, save_path, img_size=64,
                    batch_size=128, epochs=[10], lrs=[1e-3]):
        '''
        '''

        x = tf.placeholder(tf.float32, shape=[batch_size] + [img_size, img_size, 3])
        y = tf.placeholder(tf.float32, shape=[batch_size, self.labels_num])
        lr = tf.placeholder(tf.float32)

        net, y_out, prob, loss, train_opt = self.optimize_step(x, y, batch_size, lr)

        epoch_no_sum = np.sum(epochs)
        tra_img, tra_lab = self.ptfr.inputs(path=train_path, img_size=img_size, batch_size=batch_size,
                                            epoch_num=epoch_no_sum, mad=self.ptfr.train_num)

        val_img, val_lab = self.ptfr.inputs(path=valid_path, img_size=img_size, batch_size=batch_size,
                                            epoch_num=epoch_no_sum, mad=self.ptfr.valid_num)

        sess = tf.Session()
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        model_path = save_path + self.model + "_model.npz"
        if os.path.isfile(model_path):
            print("\nLoad the model from: {}.".format(model_path))
            load_params = tl.files.load_npz(path='', name=model_path)
            tl.files.assign_params(sess, load_params, net)

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_epoch_iter = int(np.floor(self.ptfr.train_num *
                                        self.ptfr.IMG_AUG / batch_size))
        valid_epoch_iter = int(np.floor(self.ptfr.valid_num *
                                        self.ptfr.IMG_AUG / batch_size))

        epoch_lr = np.array([])
        for i in range(len(epochs)):
            for j in range(epochs[i]):
                epoch_lr = np.append(epoch_lr, lrs[i])

        result = np.array([]).reshape((-1, 3))
        train_time = 0
        print()

        try:
            train_step = 0
            epoch_no = 1
            while not coord.should_stop():
                tra_lr = epoch_lr[epoch_no - 1]

                train_start_time = time.time()
                trai, tral = sess.run([tra_img, tra_lab])
                fd_train = {x: trai, y: tral, lr: tra_lr}
                fd_train.update(net.all_drop)
                sess.run(train_opt, feed_dict=fd_train)
                train_stop_time = time.time()

                train_step += 1
                train_time += train_stop_time - train_start_time

                if train_step == 1:
                    print(
                        "\n--- Training Process for Epoch {0} --- Learning Rate is {1:.6f} ---\n".format(epoch_no, tra_lr))

                if np.mod(train_step, self.PRINT_FREQ) == 0:
                    train_loss = sess.run(loss, feed_dict=fd_train)
                    print("--- Training Process --- Epoch {0} - Step {1} --- Loss: {2:.6f} --- Avgrage Time Per Step: {3:.3f}s"
                          .format(epoch_no, train_step, train_loss, train_time / self.PRINT_FREQ))
                    train_time = 0
                    train_loss = 0

                if np.mod(train_step, train_epoch_iter) == 0:
                    tl.files.save_npz(net.all_params, model_path, sess=sess)

                    val_loss = 0
                    val_label = np.array([]).reshape((-1, self.ptfr.labels_num))
                    val_prob = np.array([]).reshape((-1, self.ptfr.labels_num))

                    valid_time = 0
                    print("\n--- Validating Process for Epoch {} ---\n".format(epoch_no))
                    for valid_step in range(valid_epoch_iter):
                        valid_start_time = time.time()
                        vali, vall = sess.run([val_img, val_lab])
                        fd_valid = {x: vali, y: vall, lr: tra_lr}
                        dp_dict = tl.utils.dict_to_one(net.all_drop)
                        fd_valid.update(dp_dict)

                        val_loss_temp, val_prob_temp = sess.run(
                            [loss, prob], feed_dict=fd_valid)
                        val_loss += val_loss_temp / valid_epoch_iter
                        val_label = np.vstack((val_label, vall))
                        val_prob = np.vstack((val_prob, val_prob_temp))

                        valid_stop_time = time.time()
                        valid_time += valid_stop_time - valid_start_time
                        if np.mod(valid_step + 1, self.PRINT_FREQ) == 0:
                            print("--- Validating Process --- Epoch {0} - Step {1} --- Avgrage Time Per Step: {2:.3f}s"
                                  .format(epoch_no, valid_step + 1, valid_time / self.PRINT_FREQ))
                            valid_time = 0

                    val_f, val_t = self.find_f_thresh(val_label, val_prob, 0.001)
                    print("\n----------Validating Result for Epoch {} ----------".format(epoch_no))
                    print("Average Loss: {0:.6f}, F-Score: {1:.6f}, Threshold: {2:.3f}.\n"
                          .format(val_loss, val_f, val_t))

                    result = np.vstack((result, np.array([val_loss, val_f, val_t])))

                    train_step = 0
                    train_time = 0
                    epoch_no += 1

                if epoch_no > epoch_no_sum:
                    break

        except tf.errors.OutOfRangeError:
            print('---------------------\nTraining has stopped.')
        finally:
            coord.request_stop()
            print(result)
            np.save(save_path + self.model + "_result.npy", result)

        coord.join(thread)
        sess.close()

        return

    def read_labels(self, path):
        num_labels = pd.read_csv(path)
        self.train_imgs = num_labels['image_name'].tolist()
        self.labels = num_labels.drop('image_name', 1).as_matrix()
        self.data_num = self.labels.shape[0]
        self.labels_num = self.labels.shape[1]

        return

    def read_full_labels(self, path):
        full_labels = pd.read_csv(path)
        self.full_labels = full_labels['Labels'].tolist()
        return

    def load_images(self, dir_path, ith, input_shape, imgs_names):
        images = np.zeros(input_shape)
        non_use_num = 0

        imgs_num = len(imgs_names)
        str_idx = ith * input_shape[0]
        end_idx = (ith + 1) * input_shape[0]
        if end_idx >= imgs_num:
            non_use_num = end_idx - imgs_num
            end_idx = imgs_num - 1

        idx_range = np.arange(str_idx, end_idx)
        for idx in idx_range:
            img = cv2.imread(dir_path + imgs_names[idx])
            i = np.where(idx_range == idx)[0][0]

            img = cv2.resize(img, (input_shape[1], input_shape[2]),
                             interpolation=cv2.INTER_CUBIC)
            images[i, :, :, :] = img / 255.

        images = images.astype(np.float32)

        return images, non_use_num

    def test_model(self, dir_path, model_path, batch_size, img_size, scope, resu_dir):

        input_shape = [batch_size] + [img_size, img_size, 3]
        x = tf.placeholder(tf.float32, shape=input_shape)

        if self.model == 'cnn':
            net = self.cnn_net(x)
        elif self.model == 'res':
            net = self.res_net(x)
        else:
            print('Wrong mode! `cnn` or `res`.')
            return

        y_out = tf.reshape(net.outputs, shape=[batch_size, self.labels_num])
        prob = tf.nn.sigmoid(y_out)

        sess = tf.Session()

        load_params = tl.files.load_npz(path='', name=model_path)
        tl.files.assign_params(sess, load_params, net)

        res_probs = np.array([]).reshape((-1, self.labels_num))

        if scope == 'trainset':
            imgs_names = [name + '.jpg' for name in self.train_imgs]
        else:
            imgs_names = os.listdir(dir_path)
            self.test_imgs += [name.split('.')[0] for name in imgs_names]

        iter_num = int(np.ceil(len(imgs_names) * 1. / batch_size))
        for i in tqdm(range(iter_num)):
            imgs, non_use = self.load_images(dir_path, i, input_shape, imgs_names)
            fd_test = {x: imgs}
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            fd_test.update(dp_dict)
            prob_tmp = sess.run(prob, feed_dict=fd_test)  # prob.eval(feed_dict=fd_test)

            if non_use != 0:
                prob_tmp = prob_tmp[0:batch_size - non_use, :]

            res_probs = np.vstack((res_probs, prob_tmp))

        sess.close()

        if scope == 'trainset':
            fscore, self.threshold = self.find_f_thresh(self.labels, res_probs, 0.001)
            print("For all samples for training and validating," +
                  "f-score is {0:.6f}, threshold is {1:.3f}".format(fscore, self.threshold))
        self.threshold = .2

        res_labels = ((res_probs > self.threshold) * 1.).astype(int)
        np.save(resu_dir + self.model + '_' + scope + '.npy', res_labels)
        np.save(resu_dir + self.model + '_' + scope + '_probs.npy', res_probs)

        return res_labels, res_probs

    def num_to_str(self, resu_dir, test_labels, file_name):
        tags = []
        for i in range(test_labels.shape[0]):
            one_tag = []
            for j in range(self.labels_num):
                if test_labels[i, j] == 1:
                    one_tag.append(self.full_labels[j])

            tags.append(' '.join(one_tag))

        save_list = list(map(list, zip(self.test_imgs, tags)))
        header = ['image_name', 'tags']
        submit = pd.DataFrame.from_records(save_list, columns=header)
        submit_path = resu_dir + file_name + '.csv'
        submit.to_csv(submit_path, header=header, index=False)

        return


if __name__ == '__main__':
    from planet_tfrecords import PlanetTFRecords

    ptfr = PlanetTFRecords()
    nl_path = os.getcwd() + '\\Dataset\\train_v2n.csv'
    ptfr.read_labels(nl_path, 0.75, 'train')

    save_path = os.getcwd() + '\\Result\\'

    # cnn_train_path = os.getcwd() + '\\TFRecords\\train_64.tfrecords'
    # cnn_valid_path = os.getcwd() + '\\TFRecords\\valid_64.tfrecords'

    # print("\nStarting to train CNN net.")
    # pm_cnn = PlanetModel(ptfr=ptfr, model='cnn', mode='train')
    # pm_cnn.train_model(cnn_train_path, cnn_valid_path, save_path, img_size=64,
    #                    batch_size=128, epochs=[10, 10, 10], lrs=[1e-3, 1e-4, 1e-5])

    res_train_path = os.getcwd() + '\\TFRecords\\train_32.tfrecords'
    res_valid_path = os.getcwd() + '\\TFRecords\\valid_32.tfrecords'

    print("\nStarting to train RES net.")
    pm_res = PlanetModel(ptfr=ptfr, model='res', mode='train')
    pm_res.train_model(res_train_path, res_valid_path, save_path, img_size=32,
                       batch_size=128, epochs=[2, 2, 2], lrs=[1e-1, 1e-2, 1e-3])
