# 이미지를 로드하고 전처리하는 함수
def preprocess_image(image_path):
    # InceptionV3의 입력 크기인 299x299 크기로 이미지를 로드
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 정규화
    return img_array

# 캡션을 로드하는 함수 (캡션은 json 형식이나 텍스트 파일에 있다고 가정)
def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    return captions
