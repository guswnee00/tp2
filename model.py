"""
이미지와 라벨 데이터를 텐서 데이터로 변환한 뒤 모델에 학습
변환 -> 학습 -> 평가
"""

# 라이브러리 불러오기
import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from preprocess_functions import preprocess_image

# 데이터 루트 설정
d_root = '/Volumes/Hyun/data'

# 이미지와 라벨 데이터 디렉토리
image_data_dir = os.path.join(d_root, 'image_data')
new_label_data_dir = os.path.join(d_root, 'new_label_data')

# 이미지/라벨 파일 목록 가져오기
image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.jpg')]
new_label_files = [f for f in os.listdir(new_label_data_dir) if f.endswith('.json')]

# tensor 데이터셋 생성
def create_dataset(image_files, label_files):
    # JSON 파일을 TensorFlow 데이터셋으로 변환
    def json_to_tensors(file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        
        annotations = data.get('annotations', {})
        pm_annotations = annotations.get('PM', [])
        
        pm_codes = []
        shape_types = []
        points_lists = []
        
        for pm_annotation in pm_annotations:
            pm_code = pm_annotation.get('pm_code', 99)
            shape_type = pm_annotation.get('shape_type', '')
            points = pm_annotation.get('points', [])
            
            pm_codes.append(pm_code)
            shape_types.append(shape_type)
            points_lists.append(points)
        
        return pm_codes, shape_types, points_lists

    # JSON 파일을 텐서로 변환하여 리스트에 저장
    pm_codes_list = []
    shape_types_list = []
    points_lists = []

    for file_path in label_files:
        pm_codes, shape_types, points = json_to_tensors(file_path)
        pm_codes_list.append(pm_codes)
        shape_types_list.append(shape_types)
        points_lists.append(points)

    # 이미지 파일 경로와 라벨 데이터를 텐서로 변환
    image_paths = [os.path.join(image_data_dir, f) for f in image_files]
    pm_codes_tensor = tf.ragged.constant(pm_codes_list, dtype=tf.int32).to_tensor(default_value=99)
    shape_types_tensor = tf.ragged.constant(shape_types_list)
    
    # points_list가 비어있을 경우 기본값 설정
    padded_points_lists = [
        [points if points else [[0.0]] for points in pm_points_list]
        for pm_points_list in points_lists
    ]
    points_tensor = tf.ragged.constant(padded_points_lists, dtype=tf.float32).to_tensor(default_value=0.0)
    
    image_paths_tensor = tf.convert_to_tensor(image_paths)

    # 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, pm_codes_tensor, shape_types_tensor, points_tensor))

    # 데이터셋 맵 함수
    def preprocess_image_wrapper(image_path, pm_codes, shape_types, points):
        return preprocess_image(image_path, pm_codes, shape_types, points)  # 가져온 함수 사용

    dataset = dataset.map(preprocess_image_wrapper)
    
    return dataset

# 이미지와 라벨 파일 목록 가져오기
image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.jpg')]
label_files = [os.path.join(new_label_data_dir, f) for f in os.listdir(new_label_data_dir) if f.endswith('.json')]

# 데이터셋 생성
dataset = create_dataset(image_files, label_files)

# # 확인
# for image, pm_codes, shape_types, points in dataset:
#     for pm_code, shape_type, pm_points in zip(pm_codes, shape_types, points):
#         print("PM Code:", pm_code)
#         print("Shape Type:", shape_type)
#         print("Points:", pm_points)
#     print("Image Shape:", image.shape)

# EfficientNetB0 모델 불러오기
base_model = EfficientNetB0(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))

# 새로운 출력 레이어 추가
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(23, activation='softmax')

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# 데이터셋에서 이미지와 라벨을 추출하여 학습에 사용할 형식으로 가공
def prepare_dataset_for_training(dataset, num_classes, max_points):
    image_data = []
    label_data = []
    points_data = []

    for image, pm_codes, shape_types, points in dataset:
        for pm_code, shape_type, pm_point in zip(pm_codes, shape_types, points):
            image_data.append(image)
            label_data.append(pm_code.numpy())  # TensorFlow 텐서를 넘파이 배열로 변환
            # 포인트 데이터의 길이를 max_points로 맞추기 위해 패딩 추가
            padded_points = tf.pad(pm_point, [[0, max_points - pm_point.shape[0]]])
            points_data.append(padded_points)

    label_data = tf.keras.utils.to_categorical(label_data, num_classes=num_classes)

    image_data = tf.stack(image_data)
    label_data = tf.convert_to_tensor(label_data)
    points_data = tf.stack(points_data)

    return image_data, label_data, points_data

# 데이터셋 분할
train_size = int(0.75 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

train_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# 데이터 포인트의 최대 길이 계산
max_points = 0
for _, _, _, points in train_dataset:
    for pm_point in points:
        max_points = max(max_points, pm_point.shape[0])

# 데이터 포인트 길이를 맞추어서 데이터셋 가공
train_images, train_labels, train_points = prepare_dataset_for_training(train_dataset, num_classes=23, max_points=max_points)
val_images, val_labels, val_points = prepare_dataset_for_training(val_dataset, num_classes=23, max_points=max_points)
test_images, test_labels, test_points = prepare_dataset_for_training(test_dataset, num_classes=23, max_points=max_points)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# GPU 사용 가능한 경우
if tf.test.is_gpu_available():
    device = '/gpu:0'
else:
    device = '/cpu:0'

# TensorFlow 컨텍스트 매니저를 사용하여 연산 수행 장치 설정
with tf.device(device):
    batch_size = 32
    epochs = 10

    # 모델 훈련
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)  # 학습 진행 상황을 출력

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

