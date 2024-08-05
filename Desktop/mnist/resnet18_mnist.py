import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, Add, Input, GlobalAveragePooling2D, MaxPooling2D

import os
import time
import random
import numpy as np

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = np.expand_dims(train_images, axis=-1) / 255.0
test_images = np.expand_dims(test_images, axis=-1) / 255.0

train_images = tf.image.resize(train_images, [32, 32])
train_images = tf.repeat(train_images, 3, axis=-1)

test_images = tf.image.resize(test_images, [32, 32])
test_images = tf.repeat(test_images, 3, axis=-1)

def conv_bn_relu(x, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def identity_block(tensor, filters):
    x = conv_bn_relu(tensor, filters)
    x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, tensor])
    x = ReLU()(x)
    return x

def projection_block(tensor, filters, strides):
    x = conv_bn_relu(tensor, filters, strides=strides)
    x = Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def resnet_block(x, filters, num_blocks, strides):
    x = projection_block(x, filters, strides)
    for _ in range(1, num_blocks):
        x = identity_block(x, filters)
    return x

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64, kernel_size=7, strides=2)
    x = MaxPooling2D(3, strides=2, padding="same")(x)

    x = resnet_block(x, 64, 2, 1)
    x = resnet_block(x, 128, 2, 2)
    x = resnet_block(x, 256, 2, 2)
    x = resnet_block(x, 512, 2, 2)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

resnet18 = build_resnet18(input_shape=(32, 32, 3), num_classes=10)
resnet18.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

resnet18.summary()

checkpoint = ModelCheckpoint(
    '../backup/baseline/resnet18_mnist.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

resnet18.fit(
    train_images, 
    train_labels, 
    epochs=100, 
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint])

test_loss, test_acc = resnet18.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# 모델 크기 측정
model_size = os.path.getsize('../backup/baseline/resnet18_mnist.h5')
print(f'Model size: {model_size / 1024:.2f} KB')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# 예열 단계
with tf.device('/CPU:0'):
    for _ in range(10):
        resnet18.predict(test_images[:1])

# 추론 시간 측정
times = []
with tf.device('/CPU:0'):
    for _ in range(100):
        start_time = time.time()
        predictions = resnet18.predict(test_images)
        end_time = time.time()
        times.append(end_time - start_time)

average_inference_time = np.mean(times)
print(f'Average inference time per sample: {average_inference_time / len(test_images) * 1000:.2f} ms')
