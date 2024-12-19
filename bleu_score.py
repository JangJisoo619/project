from nltk.translate.bleu_score import sentence_bleu

# 캡션 생성을 평가하는 함수
def evaluate_caption(image_path, model, tokenizer, max_length, true_caption):
    generated_caption = generate_caption(image_path, model, tokenizer, max_length)
    true_caption_tokens = true_caption.split()
    generated_caption_tokens = generated_caption.split()

    bleu_score = sentence_bleu([true_caption_tokens], generated_caption_tokens)
    return bleu_score
