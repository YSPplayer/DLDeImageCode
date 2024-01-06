import tensorflow as tf
from sample import SampleManage
import numpy as np
import matplotlib.pyplot as plt
class TrainManage:
#pip install --upgrade tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install --upgrade keras -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install keras==2.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple 7706数据
#pip install tensorflow-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install "tensorflow<2.10“ -i https://pypi.tuna.tsinghua.edu.cn/simple"
#pip install "protobuf<3.20.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install --upgrade tensorflow-intel -i https://pypi.tuna.tsinghua.edu.cn/simple

    @staticmethod
    def test_model(img_path):
        # 加载已经训练好的模型
        model = tf.keras.models.load_model('my_model.h5')
        # 进行推断
        predictions = model.predict(SampleManage.image_to_sample_data(img_path))
        # 打印推断结果
        print(predictions[0])
        # 创建一个空数组来保存每一行最大值的索引
        max_indexes = []
        # 使用嵌套循环遍历数组，找到每一行最大值的索引并保存到max_indexes数组中
        for subarray in predictions[0]:  # 这里使用 predictions[0] 来获取第一个维度的内容
            max_index = np.argmax(subarray)  # 使用NumPy的argmax函数找到最大值的索引
            max_indexes.append(max_index)  # 将索引添加到max_indexes数组中
        # 打印最大值的索引数组
        print(max_indexes)
    @staticmethod
    def train(model,epochs):
        X_train_pre, X_train_value_pre = SampleManage.sample_data_conversion()
        # r1 = X_train_pre[0]
        # r2 = X_train_pre[1]
        # r3 = X_train_pre[2]
        # print(X_train_pre[0])
        # print(X_train_value_pre[0])
        # return
        # 训练模型,20%的数据作为验证集
        history = model.fit(X_train_pre, X_train_value_pre, epochs=epochs,validation_split=0.2)
        #训练完成后，您可以绘制训练损失和验证损失随着时间的变化来判断是否发生了过拟合：
        #如果训练损失高于验证损失，则为过拟合
        plt.plot(history.history['loss'], label='Training Loss') #训练损失
        plt.plot(history.history['val_loss'], label='Validation Loss') #验证损失
        plt.title('Training vs Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        # 保存模型
        model.save('my_model.h5')
    @staticmethod
    def continue_train(model_path,epochs):
        # 加载已有模型
        model = tf.keras.models.load_model(model_path)
        #训练模型
        TrainManage.train(model,epochs)
    @staticmethod
    def start_train(epochs):
        # 建立模型
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Flatten(input_shape=(100, 200, 3)),  # 输入层，200*100像素图像
        #     tf.keras.layers.Dense(120, activation='relu'),#第一个隐藏层120个神经元
        #     tf.keras.layers.Dense(120, activation='relu'),#第二个隐藏层120个神经元
        #     tf.keras.layers.Dense(40, activation='relu'),  # 输出层，使用40个神经元
        #     tf.keras.layers.Reshape((4, 10)),  # 将输出形状调整为 (4, 10)
        #     tf.keras.layers.Softmax(axis=2) #在第二个隐藏层之后使用Softmax，使输出概率分布
        # ])
        model = tf.keras.models.Sequential([
            # 卷积层1
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 200, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # 卷积层2
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # 卷积层3
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # 展平层
            tf.keras.layers.Flatten(),
            # 全连接层
            tf.keras.layers.Dense(128, activation='relu'),
            # 输出层
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Reshape((4, 10)),
            # 应用Softmax
            tf.keras.layers.Softmax(axis=2)
        ])
        # 编译模型
        #adam 梯度下降算法
        #categorical_crossentropy 交叉熵损失函数
        #accuracy 评估目标
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # 训练模型
        TrainManage.train(model,epochs)
