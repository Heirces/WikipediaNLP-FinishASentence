import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np



class Natural_Language_Processor:
    def __init__(self):
        self.MIN_ALLOWED_TOKENS_IN_A_SENTENCE = 6
        self.MAX_ALLOWED_TOKENS_IN_A_SENTENCE = 30
        self.OUTPUT_DIMENSION_SIZE_64 = 64
        self.vocab_size = None

        self.model = None
    
    def create_list_of_tokenized_sentences(self, doc):
        arrays_of_tokenized_sentences = []
        text_sentences = list(doc.sents)
        for text_sentence in text_sentences:
            num_of_tokens_in_sentence = len(text_sentence)
            if (num_of_tokens_in_sentence > self.MAX_ALLOWED_TOKENS_IN_A_SENTENCE) or (num_of_tokens_in_sentence < self.MIN_ALLOWED_TOKENS_IN_A_SENTENCE):
                continue

            list_of_tokens_building_into_a_sentence = []
            for token in text_sentence:
                if token.text.isalnum() or token.text in {".", ",", "'", "(", ")", "$", "!", "?", ":"}:
                    list_of_tokens_building_into_a_sentence.append(token.text)
            arrays_of_tokenized_sentences.append(list_of_tokens_building_into_a_sentence)
        return arrays_of_tokenized_sentences

    def create_list_of_sentence_training_arrays(self, list_of_arrays_of_tokenized_sentences):
        list_of_training_arrays_strings = []
        for tokenized_sentence in list_of_arrays_of_tokenized_sentences:
            sentence_length = len(tokenized_sentence)
            half_sentence_length = sentence_length // 2
            training_sentence = []

            if sentence_length > 24:
                step = 3
            elif sentence_length >= 15 and sentence_length <= 24:
                step = 2
            else:
                step = 1

            for token in range(half_sentence_length, sentence_length + 1, step):
                training_sentence = tokenized_sentence[:token]
                list_of_training_arrays_strings.append(training_sentence)
        return list_of_training_arrays_strings
            
            
            
