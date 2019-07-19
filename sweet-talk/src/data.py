import numpy as np
from keras.preprocessing.sequence import _remove_long_seq


class Parser(object):
    def __init__(self, words_embeddings):
        """
        :param words_embeddings: Dictionary string->vector
        """
        self.words_embeddings = words_embeddings

def vector_to_word_list(vector_array, word_dict, default="INVALID"):
    """
    :param vector_array: 2-dimensional array
    :param word_dict:
    :param default:
    :return:
    """
    ret = []

    for i in range(vector_array.shape[0]):
        ret.append(vector_to_word(vector_array[i], word_dict, default=default))

    return ret

def vector_to_word(vector, word_dict, default=None):
    for k, v in word_dict.items():
        print("SHAPE OF DICT VALUES:", v.shape)
        if np.array_equal(v, vector[-1:-50]):
            return k

    if default:
        print("INVALID WORD VECTOR:", vector)
        return default
    else:
        raise Exception("Encountered unknown vector:", vector)


def vectorize_word_list(word_list, word_dict):
    word_list_vec = []

    for word in word_list:
        if word not in word_dict:
            raise Exception("Encountered unknown word:", word)

        word_list_vec.append(word_dict[word])

    return np.array(word_list_vec, dtype=np.float32)


def vectorize_words_file(path, num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)
