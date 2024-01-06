from PIL import Image
import os
import numpy as np
class SampleManage:
    @staticmethod
    def hot_encode(numberArray):
        encoded_matrix = np.zeros((4, 10))
        # 遍历数组并转换为独热编码
        for i, num in enumerate(numberArray):
            encoded_matrix[i, num] = 1
        return encoded_matrix

    @staticmethod
    def image_to_sample_data(img_path):
        image = Image.open(img_path)
        # 确保图像为RGB模式,只有3个通道
        if image.mode != 'RGB':
            image = image.convert('RGB')
        X_train = np.empty((1,100, 200, 3))
        # 调整图片大小
        # image = image.resize((200, 100))
        # 将图片转换为numpy数组
        image_array = np.array(image)
        # 归一化像素值,方便训练使用，训练范围是0-1
        image_array = image_array / 255.0
        #axis=-1 表示去除掉尾部的行的数据
        X_train[0] = image_array
        return X_train
    @staticmethod
    def sample_data_conversion():
        image_main_folder = 'D:/YueShaoPu/Git/DLDeImageCode/bin/Debug/net6.0-windows7.0/Image/'
        combined_X_train_pre = None
        combined_X_train_value_pre = None
        for i in range(10):
            image_folder = image_main_folder + f"{i}/"
            # 获取文件夹中所有后缀名为.jpg的图片文件名
            image_files = [f for f in os.listdir(image_folder) if
            os.path.isfile(os.path.join(image_folder, f)) and f.endswith('.jpg')]
            # 初始化一个空的数组来存储图像数据 宽200 高100，先竖再横
            X_train_pre = np.empty((len(image_files), 100, 200, 3))
            X_train_value_pre = np.empty((len(image_files), 4, 10))
            #遍历所有的图片文件
            for i, image_file in enumerate(image_files):
                # 加载图片
                image = Image.open(os.path.join(image_folder, image_file))
                # 确保图像为RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # 转换为灰度图
                #image = image.convert('L')
                # 调整图片大小
                #image = image.resize((200, 100))
                # 将图片转换为numpy数组
                image_array = np.array(image)
                # 归一化像素值,方便训练使用，训练范围是0-1
                image_array = image_array / 255.0
                # 将处理后的图像数据添加到X_train中
                X_train_pre[i] = image_array
                #X_train_pre[i] = np.expand_dims(image_array, axis=-1)
                # 将每个字符转换为整数并存储在数组中
                number_str = os.path.splitext(image_file)[0]
                numberArray = [int(char) for char in number_str]
                #存储热编码
                X_train_value_pre[i] = SampleManage.hot_encode(numberArray)
            if combined_X_train_pre is None or not combined_X_train_pre.any():
                combined_X_train_pre = X_train_pre
            else:
                combined_X_train_pre = np.concatenate([combined_X_train_pre,X_train_pre],axis=0)
            if combined_X_train_value_pre is None or not combined_X_train_value_pre.any():
                combined_X_train_value_pre = X_train_value_pre
            else:
                combined_X_train_value_pre = np.concatenate([combined_X_train_value_pre, X_train_value_pre], axis=0)
        return combined_X_train_pre,combined_X_train_value_pre
