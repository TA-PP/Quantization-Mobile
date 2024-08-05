import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Resizing
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model

import os
import time
import random
import numpy as np

# 시드 고정
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

# 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리 및 크기 조정
train_images = np.stack((train_images,) * 3, axis=-1) / 255.0
test_images = np.stack((test_images,) * 3, axis=-1) / 255.0

# 28x28 이미지를 32x32로 크기 조정
train_images = tf.image.resize(train_images, (32, 32)).numpy()
test_images = tf.image.resize(test_images, (32, 32)).numpy()

# MobileNetV1 모델 정의
base_model = MobileNet(weights=None, include_top=False, input_shape=(32, 32, 3))

# 커스터마이징 모델 정의
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 콜백 정의: validation accuracy가 가장 높을 때 모델을 저장
checkpoint = ModelCheckpoint(
    '../backup/baseline/mobilenetv1_mnist.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 모델 학습
model.fit(
    train_images, 
    train_labels, 
    epochs=100,  # 충분히 학습
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,  # Ensure the data is shuffled
    callbacks=[checkpoint])

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# 모델 크기 확인
model = keras.models.load_model('../backup/baseline/mobilenetv1_mnist.h5')
model_size = os.path.getsize('../backup/baseline/mobilenetv1_mnist.h5')
print(f'Model size: {model_size / 1024:.2f} KB')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# 추론 시간 측정
times = []
with tf.device('/CPU:0'):
    for _ in range(100):
        start_time = time.time()
        predictions = model.predict(test_images)
        end_time = time.time()
        times.append(end_time - start_time)

average_inference_time = np.mean(times)
print(f'Average inference time per sample: {average_inference_time / len(test_images) * 1000:.2f} ms')
