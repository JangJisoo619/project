# Project(Image Captioning)

Objective: Build a model that generates descriptive captions for images.

Approach:
Use a dataset like MS COCO, which provides images with captions.
Implement a CNN to extract image features and an RNN (LSTM or GRU) to generate captions.
Train the model on the dataset, fine-tuning it for accuracy.
Evaluate the captions using BLEU scores and qualitative analysis.
Optionally, add attention mechanisms to improve caption quality.

이 모델은 CNN을 사용하여 이미지를 처리하고, RNN(LSTM 또는 GRU)을 사용하여 캡션을 생성한다. 아래는 MS COCO와 같은 이미지-캡션 데이터셋을 사용하는 전체적인 코드 구현이다.

-필수 라이브러리 설치 및 데이터 로드
`import_libraries.py`

### 1. MS COCO 데이터셋 로드 및 전처리
MS COCO와 같은 이미지-캡션 데이터셋을 다운로드하여 사용한다고 가정한다. 이미지를 로드하고 캡션을 처리하는 방법은 아래와 같다.

```python
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
```




    
