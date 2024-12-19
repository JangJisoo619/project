# InceptionV3 모델을 로드하여 이미지에서 특징을 추출
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
inception_model.trainable = False

# 이미지를 사용하여 특징을 추출하는 함수
def extract_image_features(image_paths):
    features = {}
    for image_path in image_paths:
        img = preprocess_image(image_path)
        feature = inception_model.predict(img)
        features[image_path] = feature
    return features
