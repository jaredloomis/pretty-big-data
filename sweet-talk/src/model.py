from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, Conv1D, GlobalMaxPooling1D, Activation


def create_model_simple(vocabulary_size, input_word_count, embedding_dims=50):
    """Creates and compiles a sequential model.
    :param embedding_dims: int
        Number of dimensions of data provided as input to the convolutional layer (?)
        A configurable property of the network, which is not directly dependent on the input data. Try different values.
    :param vocabulary_size: int
        Number of unique words in text.
    :param input_word_count: int
        Number of words in each document.
    :return:
    """
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(vocabulary_size, embedding_dims, input_length=input_word_count))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_model_cnn(name, max_word_count, maxlen, filters=250, embedding_dims=50, kernel_size=3, hidden_dims=250):
    model = Sequential(name=name)

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_word_count, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
