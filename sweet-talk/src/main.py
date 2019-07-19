import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np
from model import create_model_cnn
from data import vectorize_word_list, vector_to_word, vector_to_word_list
from datetime import datetime

logdir = "../logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def main(epochs, batch_size=32, max_word_count=5000, maxlen=400, seed=124):
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # Load imdb data with allow_pickle implicitly set to True
    (words_train, labels_train), (words_test, labels_test) = imdb.load_data(num_words=max_word_count)
    print(type(words_train), words_train, words_train.shape)
    # Pad inputs to have shape (_, maxlen)
    words_train = sequence.pad_sequences(words_train, maxlen=maxlen)
    words_test = sequence.pad_sequences(words_test, maxlen=maxlen)
    # restore np.load for future normal usage
    np.load = np_load_old

    # Create model
    model = create_model_cnn("first-cnn", max_word_count, maxlen)
    # Create map from words to their equivalent vectors
    embeddings = model.layers[0].get_weights()[0]
    word_to_index = imdb.get_word_index()
    words_embeddings = {
        word: embeddings[index] for word, index in word_to_index.items() if index < embeddings.shape[0]
    }

    print(
          "x shape:", words_train.shape,
          "y shape:", labels_train.shape
    )

    # Train model on all data `epochs` times
    for i in range(epochs):
        training_history = model.fit(words_train, labels_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(words_test, labels_test),
            callbacks=[keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)],
        )

        # shuffle words_train and labels_train
        np.random.seed(seed)
        train_indices = np.arange(len(words_train))
        np.random.shuffle(train_indices)
        words_train = words_train[train_indices]
        labels_train = labels_train[train_indices]

        # shuffle words_test and labels_test
        test_indices = np.arange(len(words_test))
        np.random.shuffle(test_indices)
        words_test = words_test[test_indices]
        labels_test = labels_test[test_indices]

    # Test model
    positive_test = vectorize_word_list(
        ["basically", "getting", "action", "right", "from", "the", "start"], words_embeddings)
    negative_test = vectorize_word_list(["poor", "terrible", "awful"], words_embeddings)
    print(positive_test)
    positive_test = sequence.pad_sequences(positive_test, maxlen=maxlen, dtype=np.float32)
    negative_test = sequence.pad_sequences(negative_test, maxlen=maxlen, dtype=np.float32)
    print(positive_test)
    #print(quick_test.shape, quick_test)
    print("Positive: ", model.predict(positive_test))
    print("Negative: ", model.predict(negative_test))

    imdb_test = words_test[0:5, :]
    print(imdb_test.shape)
    print("IMDB[0]:", vector_to_word_list(imdb_test, words_embeddings), model.predict(imdb_test))
    return


main(1)
