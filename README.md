# Project(Image Captioning)

Objective: Build a model that generates descriptive captions for images.

Approach:\
Use a dataset like MS COCO, which provides images with captions.\
Implement a CNN to extract image features and an RNN (LSTM or GRU) to generate captions.\
Train the model on the dataset, fine-tuning it for accuracy.\
Evaluate the captions using BLEU scores and qualitative analysis.\
Optionally, add attention mechanisms to improve caption quality.

이 모델은 CNN을 사용하여 이미지를 처리하고, RNN(LSTM 또는 GRU)을 사용하여 캡션을 생성한다. 아래는 MS COCO와 같은 이미지-캡션 데이터셋을 사용하는 전체적인 코드 구현이다.

-필수 라이브러리 설치 및 데이터 로드
`import_libraries.py`

### 1. MS COCO 데이터셋 로드 및 전처리
MS COCO와 같은 이미지-캡션 데이터셋을 다운로드하여 사용한다고 가정한다. 이미지를 로드하고 캡션을 처리하는 방법은 아래와 같다.
`ms_coco.py`
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
### 2. CNN을 사용한 이미지 특징 추출 (InceptionV3)
이미지에서 특징을 추출하기 위해 InceptionV3 모델을 사용한다. 이 모델은 ImageNet에서 미리 학습된 모델이다.
`inception_v3.py`
```python
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
```
### 3. 캡션 전처리
캡션을 처리하기 위해 Tokenizer를 사용하여 텍스트를 정수 시퀀스로 변환다.
`preprocess_captions.py`
```python
# 캡션을 토크나이즈하는 함수
def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max([len(seq) for seq in sequences])  # 캡션의 최대 길이
    return tokenizer, vocab_size, max_length
```
### 4. 이미지 캡셔닝 모델 구축 (CNN + RNN)
CNN(InceptionV3)을 사용한 이미지 특징 추출과 RNN(LSTM 또는 GRU)을 사용한 캡션 생성을 위한 모델을 정의한다.
`cnn_rnn.py`
```python
# 이미지 캡셔닝 모델 정의
def build_model(vocab_size, max_length, image_features_shape):
    # 이미지 특징 입력
    image_input = Input(shape=image_features_shape)
    image_embedding = Dense(256, activation='relu')(image_input)

    # 캡션 입력
    caption_input = Input(shape=(max_length,))
    caption_embedding = Embedding(input_dim=vocab_size, output_dim=256)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)

    # 이미지 특징과 캡션 특징 합치기
    merged = Add()([image_embedding, caption_lstm])
    merged = Dropout(0.5)(merged)

    # 출력층
    output = Dense(vocab_size, activation='softmax')(merged)

    # 모델 정의
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
### 5. 모델 훈련
훈련 데이터(이미지 특징 및 캡션)를 준비하고 모델을 훈련시킨다.
`train_model.py`
```python
# 데이터 준비 (이미지 특징 및 캡션)
# 이미지 특징과 캡션은 전처리 후 사용할 준비가 되어 있다고 가정
image_features = extract_image_features(image_paths)
captions = load_captions(captions_file)

# 캡션을 토크나이즈
tokenizer, vocab_size, max_length = tokenize_captions(captions)

# 훈련 데이터 준비 (X1: 이미지 특징, X2: 캡션, Y: 캡션의 다음 단어)
X1_train, X2_train, Y_train = prepare_data(image_features, captions, tokenizer, max_length)

# 모델 구축 및 훈련
model = build_model(vocab_size, max_length, (image_features[image_paths[0]].shape))
model.fit([X1_train, X2_train], Y_train, epochs=10, batch_size=32)
```
### 6. 캡션 생성
훈련된 모델을 사용하여 새로운 이미지에 대한 캡션을 생성하는 방법은 아래와 같다.
`generate_caption.py`
```python
def generate_caption(image_path, model, tokenizer, max_length):
    image_features = extract_image_features([image_path])
    caption_input = np.zeros((1, max_length))

    # <start> 토큰으로 캡션을 시작
    caption_input[0, 0] = tokenizer.word_index['startseq']

    for i in range(1, max_length):
        # 다음 단어 예측
        predictions = model.predict([image_features, caption_input])
        predicted_word_index = np.argmax(predictions)
        predicted_word = tokenizer.index_word[predicted_word_index]

        if predicted_word == 'endseq':
            break

        # 예측된 단어를 캡션에 추가
        caption_input[0, i] = predicted_word_index

    # 정수 인덱스를 단어로 변환
    caption = ' '.join([tokenizer.index_word[i] for i in caption_input[0] if i != 0])
    return caption
```
### 7. BLEU 점수로 모델 평가
BLEU(Bilingual Evaluation Understudy) : 생성된 캡션의 품질을 평가하는 데 사용되는 일반적인 지표\
BLEU 점수로 모델을 평가한다.
`bleu_score.py`
```python
from nltk.translate.bleu_score import sentence_bleu

# 캡션 생성을 평가하는 함수
def evaluate_caption(image_path, model, tokenizer, max_length, true_caption):
    generated_caption = generate_caption(image_path, model, tokenizer, max_length)
    true_caption_tokens = true_caption.split()
    generated_caption_tokens = generated_caption.split()

    bleu_score = sentence_bleu([true_caption_tokens], generated_caption_tokens)
    return bleu_score
```

### 결론
위 코드는 CNN(InceptionV3)을 사용하여 이미지에서 특징을 추출하고, RNN(LSTM)을 사용하여 캡션을 생성하는 기본적인 이미지 캡셔닝 모델을 구현하는 방법을 보여준다.
이 모델은 추가적인 기법(예: Attention 메커니즘)을 사용하여 더욱 개선할 수 있다. 또한 BLEU 점수를 사용하여 생성된 캡션의 품질을 평가할 수 있다.

### Remark (Attention 메커니즘 추가)
캡션 품질을 개선하기 위해 Attention 메커니즘을 추가할 수 있다. Attention은 모델이 각 단어를 생성할 때 이미지의 특정 부분에 집중하도록 학습할 수 있게 해준다. 이는 더 복잡한 아키텍처를 요구하지만, 캡션 품질을 크게 향상시킬 수 있다.



