# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("Data/quora_duplicate_questions.tsv", delimiter="\t")

# encode into unicode
df["question1"] = df["question1"].apply(lambda x: str(x))
df["question2"] = df["question2"].apply(lambda x: str(x))

import gensim

questions = list(df["question1"]) + list(df["question2"])

# tokenize
c = 0
for question in tqdm(questions):
    questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
    c += 1
    
# train model
model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)

# trim memory
model.init_sims(replace=True)

# create a dict
w2v = dict(zip(model.index2word, model.syn0))

# save model
model.save("Data/3_word2vec.md1")
model.wv.save_word2vec_format("Data/3_word2vec.bin", binary=True)

# Use pretrained model from spacy for better results

import spacy
nlp = spacy.load("en")

vecs1 = [doc.vector for doc in nlp.pipe(df["question1"], n_threads=50)]
