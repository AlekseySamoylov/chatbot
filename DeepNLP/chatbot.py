# Building a ChatBot with Deep NLP

import numpy as np
import tensorflow as tf
import re
import time

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

print("Hello world")

# Import datasets
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Create Dictionary id to line
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all conversations
conversations_ids = []
for conversation in conversations[:-1]: #exclude the last row
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

#%%
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    idx = 0
    end = len(conversation)
    while idx < end:
        questions.append(id2line[conversation[idx]])
        idx += 1
        if len(conversation) > idx:
            answers.append(id2line[conversation[idx]])
        else:
            answers.append("")
        idx += 1


#%%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-!()\"#/@;:<>{}+=-\\.?,\\|]", "", text)
    return text


# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

#%%
# Creating a dictionary tthat maps each word to its number of occurrences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

#%%
# Creating two dictionaries that map the questions words and the answers words to a unique integer
# Filter out not frequent and useless words
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        answerswords2int[word] = word_number
        word_number += 1

#%%
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

#%%
# Creating the inverse dictionary of the anserswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

#%%
# Adding the End of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

#%%
# Translating all the questions and the answers into integers
# and Replacing all the words that where filtered out by <OUT>
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in questionswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)

#%%
# Sorting questions and answers by the length of questions
# and exclude long lines
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25):
    for entry in enumerate(questions_to_int):
        line = entry[1]
        if len(line) == length:
            line_idx = entry[0]
            sorted_clean_questions.append(questions_to_int[line_idx])
            sorted_clean_answers.append(answers_to_int[line_idx])

#%%
######################### BUILDING THE SEQ2SEQ MODEL ##################################

# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.int32, name='learning_rate')
    keep_prob = tf.placeholder(tf.int32, name='keep_prob')
    return inputs, targets, learning_rate, keep_prob

#%%
# Preprocessing the targets
# Create batches and add <SOS> at the beginning of all answers
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # Зачем удалили последний элемент???
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

#%%
# Creating the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, kep_prob, sequence_length):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=kep_prob)
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                 cell_bw=encoder_cell,
                                                 sequence_length=sequence_length,
                                                 inputs=rnn_inputs,
                                                 dtype=tf.float32)
    return encoder_state

#%%
# Decoding the training set