import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

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

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = tf.image.resize(train_images, (32, 32)).numpy()
test_images = tf.image.resize(test_images, (32, 32)).numpy()

input = Input(shape=(32, 32, 1))

# AlexNet에서 사용하는 일반적인 구조를 간소화
x = Conv2D(96, (11, 11), strides=1, padding='same', activation='relu')(input)
x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(
    '../backup/baseline/alexnet_mnist.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

model.fit(
    train_images, 
    train_labels, 
    epochs=100, 
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# 모델 크기 측정
model_size = os.path.getsize('../backup/baseline/alexnet_mnist.h5')
print(f'Model size: {model_size / 1024:.2f} KB')  # KB 단위로 출력

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# 예열 단계
with tf.device('/CPU:0'):
    for _ in range(10):
        model.predict(test_images[:1])

# 실제 인퍼런스 타임 측정
times = []
with tf.device('/CPU:0'):
    for _ in range(100):
        start_time = time.time()
        predictions = model.predict(test_images)
        end_time = time.time()
        times.append(end_time - start_time)

average_inference_time = np.mean(times)
print(f'Average inference time per sample: {average_inference_time / len(test_images) * 1000:.2f} ms')
