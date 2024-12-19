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
