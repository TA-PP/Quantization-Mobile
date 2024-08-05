import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
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
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# 데이터 불러오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리 및 크기 조정
train_images = np.stack((train_images,) * 3, axis=-1) / 255.0
test_images = np.stack((test_images,) * 3, axis=-1) / 255.0

# 28x28 이미지를 32x32로 크기 조정
train_images = tf.image.resize(train_images, (32, 32)).numpy()
test_images = tf.image.resize(test_images, (32, 32)).numpy()

model = load_model('../backup/baseline/mobilenetv1_mnist.h5')

# Quantization-aware training 적용
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# 모델 컴파일
q_aware_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# 모델 요약
q_aware_model.summary()

# 콜백 정의: validation accuracy가 가장 높을 때 모델을 저장
checkpoint = ModelCheckpoint(
    '../backup/quantization/mobilenetv1_mnist_qat.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 모델 학습
q_aware_model.fit(
    train_images, 
    train_labels, 
    epochs=100,  # 충분히 학습
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint])

# 모델 평가
test_loss, test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# TFLite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# TFLite 모델 저장
with open('../backup/tflite/qat_mobilenetv1_mnist.tflite', 'wb') as f:
    f.write(tflite_model)

# TFLite 모델 크기 확인
tflite_model_size = os.path.getsize('../backup/tflite/qat_mobilenetv1_mnist.tflite')
print(f'TFLite model size: {tflite_model_size / 1024:.2f} KB')

# TFLite 추론 시간 측정
interpreter = tf.lite.Interpreter(model_path='../backup/tflite/qat_mobilenetv1_mnist.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 추론 시간 측정
times = []
for _ in range(100):
    start_time = time.time()
    for i in range(len(test_images)):
        interpreter.set_tensor(input_details[0]['index'], [test_images[i]])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    times.append(end_time - start_time)

average_inference_time = np.mean(times) / len(test_images)
print(f'Average inference time per sample: {average_inference_time * 1000:.2f} ms')