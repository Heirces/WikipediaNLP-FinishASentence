from natural_language_processor import Natural_Language_Processor
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import pickle
import time

def get_batch_of_text_from_file(file_path, batch_size = 25 * 1024):
    with open(file_path, 'r', encoding="utf-8") as working_file:
        while True:
            batch = working_file.read(batch_size)
            if not batch:
                break
            yield batch

def tokenize_training_data(nlp_instance, spacy_instance, tokenizer_instance, file_path):
    for batch_num, batch in enumerate(get_batch_of_text_from_file(file_path)):
        batch_doc = spacy_instance(batch)
        sentences = nlp_instance.create_list_of_tokenized_sentences(batch_doc)
        tokenizer_instance.fit_on_texts(sentences)
        print(f"Batch #{batch_num} processed with {len(sentences)} sentences.")
    with open("tokenizer.pkl", "wb") as tokenizer_save:
        pickle.dump(tokenizer_instance, tokenizer_save)
    print("====== Tokenizer saved ======")

def trim_the_tokenizer_to_30k_words():    
    vocab_limit = 30000
    with open("tokenizer.pkl", "rb") as file:
        original_tokenizer = pickle.load(file)

    sorted_vocab = sorted(original_tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_vocab[:vocab_limit]]

    new_tokenizer = Tokenizer(num_words=vocab_limit, oov_token=original_tokenizer.oov_token)
    new_tokenizer.fit_on_texts([' '.join(top_words)])

    with open("tokenizer_trimmed.pkl", "wb") as trimmed_file:
        pickle.dump(new_tokenizer, trimmed_file)

def get_training_arrays_from_batch_of_sentences(nlp_instance, spacy_instance, tokenizer_instance, file_path):
     for batch_num, batch in enumerate(get_batch_of_text_from_file(file_path)):
        batch_doc = spacy_instance(batch)
        sentences = nlp_instance.create_list_of_tokenized_sentences(batch_doc)
        list_of_sentence_training_arrays = nlp_instance.create_list_of_sentence_training_arrays(sentences)
        print(f"Processing Text Batch #{batch_num}")

        yield list_of_sentence_training_arrays

def train_model(nlp_instance, spacy_instance, tokenizer_instance, file_path, sentence_batch_size=128, num_epochs=100):
    oov_token_index = tokenizer_instance.word_index.get(tokenizer_instance.oov_token)
    max_len = nlp_instance.MAX_ALLOWED_TOKENS_IN_A_SENTENCE - 1

    best_loss = float('inf')  # Initialize best loss to a very high value

    for epoch in range(1, num_epochs + 1):
        print(f"\n🚀 Starting Epoch {epoch}/{num_epochs}")
        epoch_loss = 0
        batch_count = 0

        for training_arrays in get_training_arrays_from_batch_of_sentences(nlp_instance, spacy_instance, tokenizer_instance, file_path):
            x_batch = []
            y_batch = []

            for training_sentence in training_arrays:
                sequence = tokenizer_instance.texts_to_sequences([" ".join(training_sentence)])[0]
                if len(sequence) < 2:
                    continue
                input_sequence = sequence[:-1]
                label = sequence[-1]
                if label == oov_token_index:
                    continue
                x_batch.append(input_sequence)
                y_batch.append(label)

                # When batch is full, train on it
                if len(x_batch) == sentence_batch_size:
                    padded_inputs = pad_sequences(x_batch, padding='pre', maxlen=max_len, dtype='int32')
                    labels_array = np.array(y_batch, dtype='int32')
                    loss = nlp_instance.model.train_on_batch(padded_inputs, labels_array)
                    if isinstance(loss, list):
                        loss = loss[0]
                    epoch_loss += loss
                    batch_count += 1
                    if batch_count % 100 == 0:
                        print(f"Epoch {epoch}, Batch {batch_count} — Loss: {loss:.4f}")
                    x_batch, y_batch = [], []

            # Train on leftover examples
            if x_batch:
                padded_inputs = pad_sequences(x_batch, padding='pre', maxlen=max_len, dtype='int32')
                labels_array = np.array(y_batch, dtype='int32')
                loss = nlp_instance.model.train_on_batch(padded_inputs, labels_array)
                if isinstance(loss, list):
                    loss = loss[0]
                epoch_loss += loss

        print(f"✅ Epoch {epoch} complete. Total loss: {epoch_loss:.4f}")

        # Save weights every epoch (your existing save)
        nlp_instance.model.save_weights("model.weights.h5")
        print("💾 Model weights saved.")

        # Manual checkpoint: save full model if loss improved
        if epoch_loss < best_loss:
            print(f"💾 New best loss {epoch_loss:.4f} < {best_loss:.4f}, saving best model...")
            best_loss = epoch_loss
            nlp_instance.model.save("best_model.keras")  # Saves full model (architecture + weights + optimizer)
        else:
            print(f"📉 Loss did not improve this epoch (best: {best_loss:.4f})")

def create_practice_data_txt(large_file_path, practice_file_path, file_size_bytes= 50_000_000):
    with open(large_file_path, 'rb') as source_file, open(practice_file_path, 'wb') as destination_file:
        fifty_MB_chunk = source_file.read(file_size_bytes)
        destination_file.write(fifty_MB_chunk)

def predict_next_word(incomplete_sentence, tokenizer, model, max_len, temperature=1.0):
    sequence = tokenizer.texts_to_sequences([incomplete_sentence])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_len, padding='pre')
    preds = model.predict(padded_sequence)[0]  # predicted probabilities for next token
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    pred_id = np.random.choice(len(preds), p=preds)
    return tokenizer.index_word.get(pred_id, "<UNK>")

def main(build_tokenizer:bool, file_path):
    start_time = time.time()
    spacy_lib = spacy.load('en_core_web_sm', disable=["ner", "parser", "tok2vec", "tagger", "lemmatizer"])
    spacy_lib.add_pipe("sentencizer")
    nlp = Natural_Language_Processor()
    tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
    
    if build_tokenizer:
        tokenize_training_data(nlp, spacy_lib, tokenizer, file_path)
    with open(r"tokenizer_trimmed.pkl", 'rb') as tokenizer_data:
        tokenizer = pickle.load(tokenizer_data)

    nlp.vocab_size = len(tokenizer.word_index) + 1
    nlp.model = Sequential()
    nlp.model.add(Embedding(input_dim=nlp.vocab_size, output_dim=nlp.OUTPUT_DIMENSION_SIZE_64))
    nlp.model.add(LSTM(nlp.OUTPUT_DIMENSION_SIZE_64))
    nlp.model.add(Dense(nlp.vocab_size, activation='softmax'))
    nlp.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_model(nlp, spacy_lib, tokenizer, file_path)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    testing_sentences = True
    while testing_sentences:
        user_sentence = input("Enter half a sentence: ")
        in_generating_mode = True
        while in_generating_mode:
            next_word = predict_next_word(user_sentence, tokenizer, nlp.model, nlp.MAX_ALLOWED_TOKENS_IN_A_SENTENCE)
            user_sentence = user_sentence + " " + next_word
            print(user_sentence)
            keep_generating = input("Keep generating type '.'")
            if keep_generating != ".":
                in_generating_mode = False
        keep_testing_sentences = input("Press Y to keep testing new sentences")
        if keep_testing_sentences != "y":
            testing_sentences = False

#create_practice_data_txt(large_file_path=r"clean_text1.txt", practice_file_path=r"practice_data.txt")
main(file_path=r"practice_data.txt", build_tokenizer=False )
