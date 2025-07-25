import tensorflow as tf
print("TF Version:", tf.__version__)
print("Has Keras:", hasattr(tf, 'keras'))



class Capstone_Tokenization:
    def __init__(self):
        self.MIN_ALLOWED_TOKENS_IN_A_SENTENCE = 6
        self.MAX_ALLOWED_TOKENS_IN_A_SENTENCE = 30
        self.avg_sentence_length_in_tokens = None
        self.total_sentences_being_trained_on = 0

        self.total_used_sentences_in_text_sentences = 0
        self.total_tokens_in_used_sentences = 0

        self.list_of_arrays_of_tokenized_sentences = [] 
        self.list_of_training_arrays_by_sentence = []  
    
    def create_list_of_tokenized_sentences(self, doc):
        text_sentences = list(doc.sents)
        for text_sentence in text_sentences:
            num_of_tokens_in_sentence = len(text_sentence)
            if (num_of_tokens_in_sentence > self.MAX_ALLOWED_TOKENS_IN_A_SENTENCE) or (num_of_tokens_in_sentence < self.MIN_ALLOWED_TOKENS_IN_A_SENTENCE):
                continue
            self.total_used_sentences_in_text_sentences += 1
            self.total_tokens_in_used_sentences += num_of_tokens_in_sentence

            list_of_tokens_building_into_a_sentence = []
            for token in text_sentence:
                if token.text.isalnum() or token.text in {".", ",", "'", "(", ")", "$"}:
                    list_of_tokens_building_into_a_sentence.append(token.text)
            self.list_of_arrays_of_tokenized_sentences.append(list_of_tokens_building_into_a_sentence)
        
        avg_tokens_in_sentence = self.total_tokens_in_used_sentences // self.total_used_sentences_in_text_sentences
        
        #print(f"Average: {avg_tokens_in_sentence}\nNum Sentences: {self.total_used_sentences_in_text_sentences}\nTokenized Sents: {self.list_of_arrays_of_tokenized_sentences}")

        return {
        "avg": avg_tokens_in_sentence,
        "num_sentences": self.total_used_sentences_in_text_sentences,
        "tokenized_sentences": self.list_of_arrays_of_tokenized_sentences
    }

    def create_list_of_sentence_training_arrays(self):
        for tokenized_sentence in self.list_of_arrays_of_tokenized_sentences:
            sentence_length = len(tokenized_sentence)
            half_sentence_length = sentence_length // 2
            training_sentences = []

            if sentence_length > 24:
                step = 3
            elif sentence_length >= 15 or sentence_length <= 24:
                step = 2
            else:
                step = 1

            for token in range(half_sentence_length, sentence_length + 1, step):
                training_sentences.append(tokenized_sentence[:token])
            self.list_of_training_arrays_by_sentence.append(training_sentences)

            
            
