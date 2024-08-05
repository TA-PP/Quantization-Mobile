import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

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

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# 28x28 이미지를 32x32로 크기 조정
train_images = tf.image.resize(train_images[..., tf.newaxis], [32, 32], method='bilinear') # tf.newaxis를 사용하여 채널 차원 추가
test_images = tf.image.resize(test_images[..., tf.newaxis], [32, 32], method='bilinear')

model = load_model('../backup/baseline/lenet_mnist.h5')

# 양자화 인식 훈련 적용
quantize_model = tfmot.quantization.keras.quantize_model(model)
quantize_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

quantize_model.summary()

checkpoint_filepath = '../backup/quantization/lenet_mnist_qat.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

quantize_model.fit(
    train_images, 
    train_labels, 
    epochs=100, 
    batch_size=512, 
    validation_split=0.2,
    shuffle=True,
    callbacks=[checkpoint_callback]
)

# 저장된 최고의 양자화된 모델 로드
with tfmot.quantization.keras.quantize_scope():
    best_quantized_model = load_model(checkpoint_filepath)

# 서빙 서명 추가
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 32, 32, 1], dtype=tf.float32, name='x')
])
def serving_fn(x):
    return {'output': best_quantized_model(x)}

# 학습 서명 추가
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 32, 32, 1], dtype=tf.float32, name='x'),
    tf.TensorSpec(shape=[None], dtype=tf.int32, name='y')
])
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = best_quantized_model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, best_quantized_model.trainable_variables)
    best_quantized_model.optimizer.apply_gradients(zip(gradients, best_quantized_model.trainable_variables))
    return {'loss': tf.reduce_mean(loss)}

tf.saved_model.save(best_quantized_model, '../backup/tflite/saved_model', signatures={'serving_default': serving_fn, 'train': train_step})

# Tensorflow Lite 변환
converter = tf.lite.TFLiteConverter.from_saved_model('../backup/tflite/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # 기본 연산자 세트
    tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow Select 연산자 세트
]

tflite_quant_model = converter.convert()

# TFLite 모델 저장
tflite_model_path = '../backup/tflite/qat_lenet_mnist.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)
print(f'Model saved to: {tflite_model_path}')
