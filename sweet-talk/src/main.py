import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np
from model import create_model_cnn
from data import vectorize_word_list, vector_to_word, vector_to_word_list
from datetime import datetime

logdir = "../logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def load_data(unique_word_count, max_word_count):
    """
    :param unique_word_count:
    :param maxlen:
    :return: four numpy arrays of shape (_, max_word_count)
    """
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # Load imdb data with allow_pickle implicitly set to True
    (words_train, labels_train), (words_test, labels_test) = imdb.load_data(num_words=unique_word_count)
    # Pad inputs to have shape (_, maxlen)
    words_train = sequence.pad_sequences(words_train, maxlen=max_word_count)
    words_test = sequence.pad_sequences(words_test, maxlen=max_word_count)
    # restore np.load for future normal usage
    np.load = np_load_old

    return (words_train, labels_train), (words_test, labels_test)


def train(model, epochs, batch_size, train_data, test_data):
    words_train = train_data[0]
    labels_train = train_data[1]
    words_test = test_data[0]
    labels_test = test_data[1]
    # Train model on all data `epochs` times
    for i in range(epochs):
        training_history = model.fit(words_train, labels_train,
                                     batch_size=batch_size, epochs=epochs,
                                     validation_data=(words_test, labels_test),
                                     callbacks=[keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True,
                                                                            write_images=True)],
                                     )

        # shuffle words_train and labels_train
        seed = 124
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


def test_words(model, words_list, word_to_embedding):
    positive_test = vectorize_word_list(words_list, word_to_embedding)
    negative_test = vectorize_word_list(["poor", "terrible", "awful"], word_to_embedding)
    print(positive_test)
    positive_test = sequence.pad_sequences(positive_test, maxlen=max_word_count, dtype=np.float32)
    negative_test = sequence.pad_sequences(negative_test, maxlen=max_word_count, dtype=np.float32)
    print(positive_test)
    print("Positive: ", model.predict(positive_test))
    print("Negative: ", model.predict(negative_test))

def test_imdb(model, words_test):
    imdb_test = words_test[0:5, :]
    print("IMDB[0]:", vector_to_word_list(imdb_test, word_to_embedding), model.predict(imdb_test))


def main(epochs, batch_size=32, unique_word_count=5000, max_word_count=400, seed=124):
    # save np.load
    # Load imdb data
    (words_train, labels_train), (words_test, labels_test) = load_data(unique_word_count, max_word_count)

    # Create model
    model = create_model_cnn("first-cnn", unique_word_count, max_word_count)

    # Create map from words to their equivalent vectors
    embeddings = model.layers[0].get_weights()[0]
    word_to_token = imdb.get_word_index()
    word_to_embedding = {
        word: embeddings[token] for word, token in word_to_token.items() if token < embeddings.shape[0]
    }

    # Train model on all data `epochs` times
    train(model, epochs, batch_size)

    # Test model
    positive_test = vectorize_word_list(
        ["basically", "getting", "action", "right", "from", "the", "start"], word_to_embedding)
    negative_test = vectorize_word_list(["poor", "terrible", "awful"], word_to_embedding)
    print(positive_test)
    positive_test = sequence.pad_sequences(positive_test, maxlen=max_word_count, dtype=np.float32)
    negative_test = sequence.pad_sequences(negative_test, maxlen=max_word_count, dtype=np.float32)
    print(positive_test)
    print("Positive: ", model.predict(positive_test))
    print("Negative: ", model.predict(negative_test))

    imdb_test = words_test[0:5, :]
    print(imdb_test.shape)
    print("IMDB[0]:", vector_to_word_list(imdb_test, word_to_embedding), model.predict(imdb_test))
    return


main(1)
