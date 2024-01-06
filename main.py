from train import TrainManage
from sample import SampleManage
import tensorflow as tf
from tensorflow.python.client import device_lib
import os

if __name__ == '__main__':
    #os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 这里0表示第0块GPU，看个人情况分配

    #print(device_lib.list_local_devices())
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #TrainManage.train("",1)
    #TrainManage.continue_train('my_model.h5',50)
    #TrainManage.start_train(100)
    TrainManage.test_model('D:/YueShaoPu/Git/DLDeImageCode/bin/Debug/net6.0-windows7.0/Image/6/0968.jpg')


