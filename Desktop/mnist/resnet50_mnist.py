import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50  # ResNet50 사용
from tensorflow.keras.callbacks import ModelCheckpoint

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

# 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리 및 크기 조정
train_images = np.stack((train_images,) * 3, axis=-1) / 255.0
test_images = np.stack((test_images,) * 3, axis=-1) / 255.0

# 28x28 이미지를 32x32로 크기 조정
train_images = tf.image.resize(train_images, (32, 32)).numpy()
test_images = tf.image.resize(test_images, (32, 32)).numpy()

# ResNet50 모델 로드, ImageNet 가중치 사용
base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))

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

model.summary()

checkpoint_filepath = '../backup/baseline/resnet50_mnist.h5'
checkpoint = ModelCheckpoint(
    checkpoint_filepath,
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
model_size = os.path.getsize(checkpoint_filepath)
print(f'Model size: {model_size / 1024:.2f} KB')

# GPU 사용 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 예열 단계
with tf.device('/CPU:0'):
    for _ in range(10):
        model.predict(test_images[:1])

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

