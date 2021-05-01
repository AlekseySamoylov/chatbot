# Building a ChatBot with Deep NLP

import numpy as np
import tensorflow as tf
import re
import time

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
