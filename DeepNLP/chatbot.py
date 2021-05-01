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



