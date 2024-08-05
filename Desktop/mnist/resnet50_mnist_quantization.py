import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.lite.python.interpreter import Interpreter

import os
import time
import random
import numpy as np
from tqdm import tqdm

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

# ResNet50 모델을 불러오기
model = load_model('../backup/baseline/resnet50_mnist.h5')

# Quantization-aware training 적용
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

q_aware_model.summary()

checkpoint_filepath = '../backup/quantization/resnet50_mnist_qat.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

q_aware_model.fit(
    train_images, 
    train_labels, 
    epochs=100,  # 에포크 수를 1로 설정하여 데모 목적
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint_callback]
)

# 모델 평가
test_loss, test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc*100:.2f}%')

# 저장된 최고의 양자화된 모델 로드
with tfmot.quantization.keras.quantize_scope():
    best_quantized_model = load_model(checkpoint_filepath)

# 양자화된 모델 평가
test_loss, test_acc = best_quantized_model.evaluate(test_images, test_labels)
print(f'Quantized Test accuracy: {test_acc*100:.2f}%')

# TensorFlow Lite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(best_quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# TFLite 모델 저장
tflite_model_path = '../backup/tflite/qat_resnet50_mnist.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f'Model saved to: {tflite_model_path}')

# 모델 크기 출력
model_size = os.path.getsize(tflite_model_path)
print(f'Quantized model size: {model_size / 1024:.2f} KB')

# TFLite 모델 로드 및 인터프리터 생성
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 예열 단계
for _ in range(10):
    input_data = np.expand_dims(test_images[0], axis=0).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

# 인퍼런스 시간 측정
times = []
for _ in tqdm(range(100), desc="Measuring Inference Time"):
    for i in range(len(test_images)):
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
    times.append(end_time - start_time)

average_inference_time = np.mean(times)
print(f'Average inference time per sample: {average_inference_time:.8f} ms')
