import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import time
import random
import numpy as np

# 시드 고정
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

# 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

# 28x28 이미지를 32x32로 크기 조정
train_images = tf.image.resize(train_images[..., tf.newaxis], [32, 32], method='bilinear') # tf.newaxis를 사용하여 채널 차원 추가
test_images = tf.image.resize(test_images[..., tf.newaxis], [32, 32], method='bilinear')

# LeNet-300-100 모델 정의
model = Sequential([
    Flatten(input_shape=(32, 32, 1)),
    Dense(300, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 콜백 정의: validation accuracy가 가장 높을 때 모델을 저장
checkpoint = ModelCheckpoint(
    '../backup/baseline/lenet_mnist.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 모델 학습
model.fit(
    train_images, 
    train_labels, 
    epochs=100, 
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint])

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# 모델 크기 확인
model = keras.models.load_model('../backup/baseline/lenet_mnist.h5')
model_size = os.path.getsize('../backup/baseline/lenet_mnist.h5')
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