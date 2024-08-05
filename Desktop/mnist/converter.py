import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import Interpreter

import numpy as np
import os

def convert(model_name):
    # 데이터 불러오기
    (_, _), (test_images, test_labels) = mnist.load_data()
    
    # 데이터 전처리 및 크기 조정
    test_images = np.expand_dims(test_images, axis=-1) / 255.0
    test_images = tf.image.resize(test_images, [32, 32])
    test_images = tf.repeat(test_images, 3, axis=-1)

    test_images = test_images / 255.0
    
    # 모델 로드
    model_path = f'../backup/baseline/{model_name}.h5'
    model = load_model(model_path)
    
    # TensorFlow Lite 모델로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # 변환된 모델 저장
    tflite_model_path = f'../backup/tflite/{model_name}.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Converted TensorFlow Lite model is saved to: {tflite_model_path}")

    # 모델 크기 출력
    model_size = os.path.getsize(tflite_model_path)
    print(f'Quantized model size: {model_size / 1024:.2f} KB')
    
    # TFLite 인터프리터 설정
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # 입력 및 출력 텐서 정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # TFLite 모델 정확도 측정
    correct_predictions = 0
    for i in range(len(test_images)):
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        
        if prediction == test_labels[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_images)
    print(f'TFLite model accuracy: {accuracy * 100:.2f}%')

convert('mobilenetv2_mnist')