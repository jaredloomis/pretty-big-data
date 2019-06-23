"""
# An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be reversed, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
Two digits reversed:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
Three digits reversed:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
Four digits reversed:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
Five digits reversed:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
"""

from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

REVERSE_OPTIMIZATION = True


class Colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class BinaryOp(object):
    def __init__(self, char, operand_digits, result_digits, op):
        self.char = char
        self.op = op
        self.operand_digits = operand_digits
        self.result_digits = result_digits

        self.chars = char + '0123456789- '
        self.ctable = CharacterTable(self.chars)

    def max_expression_len(self):
        return self.operand_digits * 2 + 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BinaryOp(" + self.char + ", " + str(self.operand_digits) + ", " + str(self.result_digits) + ")"


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class ExampleGenerator(object):
    def __init__(self, binop):
        self.binop = binop

    def generate_vectorized_examples(self, training_size, reverse=False):
        # Generate raw examples
        questions, expected_answers = self.generate_examples(training_size, reverse)

        # Put them in a 3d matrix of booleans
        questions_matrix = np.zeros((len(questions), binop.max_expression_len(), len(self.binop.chars)), dtype=np.bool)
        answers_matrix = np.zeros((len(questions), binop.result_digits, len(self.binop.chars)), dtype=np.bool)
        for i, sentence in enumerate(questions):
            questions_matrix[i] = self.binop.ctable.encode(sentence, binop.max_expression_len())
        for i, sentence in enumerate(expected_answers):
            answers_matrix[i] = self.binop.ctable.encode(sentence, binop.result_digits)

        # Shuffle (x, y) in unison as the later parts of x will almost all be larger
        # digits.
        indices = np.arange(len(answers_matrix))
        np.random.shuffle(indices)
        questions_matrix = questions_matrix[indices]
        answers_matrix = answers_matrix[indices]

        return questions_matrix, answers_matrix

    def generate_examples(self, training_size, reverse=False):
        """Generate a set of question and answer pairs.
        # Arguments
            training_size: number of examples to generate
            reverse: whether to reverse
        # Returns
            (questions: array, answers: array)
            the solution to questions[i] is answers[i]
        """
        questions = []
        expected = []
        seen = set()

        while len(questions) < training_size:
            a, b = self.generate_operand(), self.generate_operand()

            # Skip any addition questions we've already seen
            # Also skip any such that x+Y == Y+x (hence the sorting).
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)

            # Create answer string.
            # Pad the data with spaces such that it is always binop.max_expression_len.
            q = '{}{}{}'.format(a, self.binop.char, b)
            query = q + ' ' * (self.binop.max_expression_len() - len(q))
            ans = str(binop.op(a, b))

            # Answers can be of maximum size binop.result_digits
            if len(ans) > self.binop.result_digits:
                raise Exception("Number of digits in result is greater than expected. Found " + str(len(ans)))
            ans += ' ' * (self.binop.result_digits - len(ans))
            if reverse:
                # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
                # space used for padding.)
                query = query[::-1]
            questions.append(query)
            expected.append(ans)

        return questions, expected

    def generate_operand(self):
        return int(''.join(
            np.random.choice(list('0123456789'))
              for i in range(np.random.randint(1, self.binop.operand_digits + 1))
        ))


def build_model():
    # Try replacing with GRU, or SimpleRNN.
    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    LAYERS = 1

    mdl = Sequential()

    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    mdl.add(RNN(HIDDEN_SIZE, input_shape=(binop.max_expression_len(), len(binop.chars))))

    # As the decoder RNN's input, repeatedly provide with the last output of
    # RNN for each time step. Repeat 'binop.digits + 1' times as that's the maximum
    # length of output, e.g., when binop.digits=3, max output is 999+999=1998.
    #mdl.add(layers.RepeatVector(binop.operand_digits + 1))
    mdl.add(layers.RepeatVector(binop.result_digits))

    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        mdl.add(RNN(HIDDEN_SIZE, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    mdl.add(layers.TimeDistributed(layers.Dense(len(binop.chars), activation='softmax')))
    mdl.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return mdl


class Trainer(object):
    def __init__(self, model, binop, batch_size):
        self.model = model
        self.batch_size = batch_size

        # Initialize variables for storing most recent validation set
        self.validation_set_x = {}
        self.validation_set_y = {}

        # For storing accuracy of past predictions
        self.previous_predictions = []

    def train(self, x_train, y_train, x_val, y_val):
        self.validation_set_x = x_val
        self.validation_set_y = y_val

        self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=1,
            validation_data=(x_val, y_val)
        )

    def predict(self, rowx):
        preds = model.predict_classes(rowx, verbose=0)
        return binop.ctable.decode(preds[0], calc_argmax=False)

    def display_accuracy(self):
        samples = 10
        correct_count = 0
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(samples):
            ind = np.random.randint(0, len(self.validation_set_x))
            rowx, rowy = self.validation_set_x[np.array([ind])], y_val[np.array([ind])]
            q = binop.ctable.decode(rowx[0])
            correct = binop.ctable.decode(rowy[0])
            guess = self.predict(rowx)
            self.log_prediction(correct == guess)

            print('Q:', q[::-1] if REVERSE_OPTIMIZATION else q, end=' ')
            print('=', correct, end=' ')

            if correct == guess:
                print(Colors.ok + '☑' + Colors.close, end=' ')
                correct_count = correct_count + 1
            else:
                print(Colors.fail + '☒' + Colors.close, end=' ')

        accuracy = self.scoped_accuracy(100)
        print("accuracy:", str(accuracy * 100) + "%")

        return correct_count / samples

    def log_prediction(self, prediction):
        self.previous_predictions.append(prediction)

    def scoped_accuracy(self, previous_n=10):
        history_len = len(self.previous_predictions)
        if history_len == 0:
            return -1

        predictions = self.previous_predictions[-previous_n:]
        correct_count = len([prediction for prediction in predictions if prediction])
        return correct_count / len(predictions)

    def historical_accuracy(self):
        return self.scoped_accuracy(len(self.previous_predictions))

    def weighted_accuracy(self, n=10):
        return (self.historical_accuracy() + self.scoped_accuracy(n)) / 2


# The operation to learn
binop = BinaryOp('*', 1, 2, lambda x, y: x * y)
# Example questions and answers, vectorized
x, y = ExampleGenerator(binop).generate_vectorized_examples(5000, REVERSE_OPTIMIZATION)

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Build model...')
model = build_model()
model.summary()

print("Build trainer")
trainer = Trainer(model, binop, 128)


MAX_ITERATIONS = 10000
iteration = 0

while trainer.scoped_accuracy(100) < 98 and iteration < MAX_ITERATIONS:
    print()
    print('-' * 50)
    print('Iteration', iteration)
    trainer.train(x_train, y_train, x_val, y_val)
    trainer.display_accuracy()
    iteration = iteration + 1
