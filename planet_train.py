import os
from planet_model import PlanetModel
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
pm_res.train_model(res_train_path, res_valid_path, save_path, img_size=64,
                   batch_size=128, epochs=[2, 2, 2], lrs=[1e-1, 1e-2, 1e-3])
