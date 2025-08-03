from natural_language_processor import Natural_Language_Processor
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import traceback



    # nlp_instance.list_of_training_arrays_ids = tokenizer.texts_to_sequences(nlp_instance.list_of_training_arrays_strings)


    # list_of_inputs = []
    # list_of_labels = []
    # for training_array in batch:
    #     if len(training_array) < nlp_instance.MIN_ALLOWED_TOKENS_IN_A_SENTENCE:
    #         continue  # Skip sequences that are too short to split
    #     list_of_inputs.append(training_array[:-1])
    #     list_of_labels.append(training_array[-1])
    # padded_inputs = pad_sequences(list_of_inputs, padding='pre', maxlen=nlp_instance.MAX_ALLOWED_TOKENS_IN_A_SENTENCE - 1, dtype='int32')
    # labels = np.array(list_of_labels, dtype=np.int32)
    # nlp_instance.list_of_training_inputs = padded_inputs
    # nlp_instance.list_of_training_labels = labels
    # nlp_instance.model.fit(padded_inputs, nlp_instance.list_of_training_labels, epochs=1, verbose=1)




def tokenize_a_batch(batch, nlp_instance, spacy_instance):
    tokenization_doc = spacy_instance(batch)
    batch_of_tokenized_sentences = nlp_instance.create_list_of_tokenized_sentences(tokenization_doc)
    nlp_instance.tokenizer.fit_on_texts(batch_of_tokenized_sentences)

def tokenize_a_file(file_path, batch_size = 999_000):
    print(f"Beginning file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as working_file:
        while True:
            batch = working_file.read(batch_size)
            if batch == "":
                break
            yield batch

def tokenize_training_data(nlp_instance, spacy_instance):
    directory = r"C:\Users\Garre\OneDrive\Desktop\WGU\Capstone\cleaned_text_files"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            for batch_num, batch in enumerate(tokenize_a_file(file_path)):
                try:
                    if batch.strip():  # skip empty/blank batches
                        tokenize_a_batch(batch, nlp_instance, spacy_instance)
                        print(f"{filename}\t--Batch #{batch_num} processed.")
                    else:
                        print(f"Empty batch in {filename}, batch #{batch_num}, skipping.")
                except Exception as batch_err:
                    print(f"[ERROR] Failed processing batch #{batch_num} in {filename}")
                    print(traceback.format_exc())
                    continue  # Skip to next batch
        except Exception as file_err:
            print(f"[ERROR] Failed reading file: {filename}")
            print(traceback.format_exc())
            continue  # Skip to next file

def save_tokenizer(tokenizer, file_path="tokenizer.json"):
    with open(file_path, "w", encoding="utf-8") as saved_tokenizer_file:
        json.dump(tokenizer.to_json(), saved_tokenizer_file)
    


def predict_next_word(incomplete_sentence, tokenizer, model, max_len):
    sequence = tokenizer.texts_to_sequences([incomplete_sentence])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_len, padding='pre')
    pred_id = np.argmax(model.predict(padded_sequence), axis=-1)[0]
    
    return tokenizer.index_word.get(pred_id, "<UNK>")

def main(build_tokenizer:bool):
    spacy_lib = spacy.load('en_core_web_sm', disable=["ner", "parser", "tok2vec", "tagger", "lemmatizer"])
    spacy_lib.add_pipe("sentencizer")
    nlp = Natural_Language_Processor()
    
    if build_tokenizer:
        tokenize_training_data(nlp, spacy_lib)
        save_tokenizer(nlp.tokenizer)


    # vocab_size = len(tf_tokenizer.word_index) + 1
    # nlp.model = Sequential()   # Sequential model is a good choice because my model is simple with a single input.
    #                                     # Each layer can transform the data and output it as the input to the next layer.

    # nlp.model.add(Embedding(input_dim=vocab_size, output_dim=nlp.OUTPUT_DIMENSION_SIZE_64))
    # nlp.model.add(LSTM(nlp.OUTPUT_DIMENSION_SIZE_64))
    # nlp.model.add(Dense(vocab_size, activation='softmax'))
    # nlp.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # user_sentence = input("Enter half a sentence: ")
    # in_generating_mode = True
    # while in_generating_mode:
    #     next_word = predict_next_word(user_sentence, tf_tokenizer, nlp.model, nlp.MAX_ALLOWED_TOKENS_IN_A_SENTENCE)
    #     user_sentence = user_sentence + " " + next_word
    #     print(user_sentence)
    #     keep_generating = input("Keep generating type '.'")
    #     if keep_generating != ".":
    #         in_generating_mode = False

main(build_tokenizer=True)