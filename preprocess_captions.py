# 캡션을 토크나이즈하는 함수
def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max([len(seq) for seq in sequences])  # 캡션의 최대 길이
    return tokenizer, vocab_size, max_length
