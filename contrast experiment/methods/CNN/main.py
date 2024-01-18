import io
import os
import tensorflow as tf
import cv2
from PIL.Image import Image
from keras import backend
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.utils import to_categorical
import numpy as np

train_images = []
train_labels = []
test_images = []
test_labels = []

names_list_1 = []
size_1 = 0
file_1 = open('imagedata/mfcc/after filtering(200hz, low)/dataset/train/train.txt')
for f in file_1:
    names_list_1.append(f)
    size_1 += 1
for index in range(size_1):
    image_path = 'imagedata/mfcc/after filtering(200hz, low)/dataset/train/' + names_list_1[index].split(' ')[0]  # 获取图片数据路径
    img = cv2.imread(image_path)  # 读取图片
    train_images.append(img)
    label = int(names_list_1[index].split(' ')[1])  # 读取标签
    train_labels.append(label)

names_list_2 = []
size_2 = 0
file_2 = open('imagedata/mfcc/after filtering(200hz, low)/dataset/test/test.txt')
for f in file_2:
    names_list_2.append(f)
    size_2 += 1
for index in range(size_2):
    image_path = 'imagedata/mfcc/after filtering(200hz, low)/dataset/train/' + names_list_2[index].split(' ')[0]  # 获取图片数据路径
    img = cv2.imread(image_path)  # 读取图片
    test_images.append(img)
    label = int(names_list_2[index].split(' ')[1])  # 读取标签
    test_labels.append(label)


# 转换数据集类型，使其符合神经网络训练
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 图片归一化
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# 重塑数据集，使其多一个维度
# train_images = np.expand_dims(train_images, axis=3)
# test_images = np.expand_dims(test_images, axis=3)

# train_labels = tf.squeeze(train_labels)
# train_labels = tf.one_hot(train_labels, depth=10)
# test_labels = tf.squeeze(test_labels)
# test_labels = tf.one_hot(test_labels, depth=10)

# 创建模型
model = Sequential()

# # 添加卷积层和池化层
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # 添加更多的卷积层和池化层
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # 扁平化特征图
# model.add(Flatten())
#
# # 添加全连接层
# model.add(Dense(128, activation='relu'))
#
# # 添加输出层
# model.add(Dense(3, activation='softmax'))

num_filters = 32
filter_size = 5
num_filters_2 = 16
filter_size_2 = 3
pool_size = 2

# Build the model.
model = Sequential([
    Conv2D(num_filters, filter_size),
    MaxPooling2D(pool_size=pool_size),
    # Conv2D(num_filters_2, filter_size_2),
    # MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(3, activation='softmax'),
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lr = backend.get_value(model.optimizer.lr)
backend.set_value(model.optimizer.lr, 0.00005)

# 打印模型概况
# model.summary()

# 训练网络
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=30,
    validation_data=(test_images, to_categorical(test_labels)),
)

predictions = model.predict(test_images[:])
# print(predictions)
print(np.argmax(predictions, axis=1))
print(test_labels[:])
