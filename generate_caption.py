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
