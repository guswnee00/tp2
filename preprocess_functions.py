# 데이터셋 맵 함수 정의

import tensorflow as tf
 
def preprocess_image(image_path, pm_codes, shape_types, points):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, pm_codes, shape_types, points
