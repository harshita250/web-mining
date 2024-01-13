#!/usr/bin/env python
# coding: utf-8

# # <center>HW #1 - Harshita Mahesh Hiremath</center>

# In[61]:


import string
import pandas as pd
import numpy as np
import re

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[62]:


def tokenize(text):
    
    vocab = {}
    
    # add your code here
    list_of_punc = string.punctuation
    
    for i in text:
        if i in list_of_punc:
            text = text.replace(i," ") #
    
    text = text.split(" ")
    tokens = []
    
    for i in text:
        if i != "" and len(i)>1:
            tokens.append(i)        
            
    for token in tokens:
        if token not in vocab.keys():
            vocab[token] = 1
        else:
            vocab[token] += 1     
    return vocab


# In[64]:


def get_dtm(sents, min_count = 1):
    
    dtm, all_words = None, None
    
    # add your code here
    list_of_dict = []
    for sent in sents:
        list_of_dict.append(tokenize(sent))

    unique_words = np.array([])
    all_words = " ".join(sents)

    for word in all_words:
        if word in string.punctuation:
            all_words = all_words.replace(word," ")
            
    all_words = set(all_words.split(" "))
    unique_words = np.array([word for word in all_words if len(word)>1])
    
    dtm = np.zeros((len(sents),unique_words.shape[0]))
    
    for i in range(dtm.shape[0]):
        for j in range(dtm.shape[1]):
            if unique_words[j] in sents[i]:
                dtm[i, j] += 1

    idx = np.where(np.sum(dtm,axis=0)>=min_count)

    dtm = dtm[:,idx].reshape(dtm.shape[0],-1)
    all_words = unique_words[idx]

    return dtm, all_words


# In[68]:


def analyze_dtm(dtm, words, sents):
    
    # add your code here
    df = np.sum(dtm,axis=0,keepdims=True)
    tf = dtm/np.sum(dtm,axis=1,keepdims=True)       
    tf_idf = tf/(1 + np.log(df))   
    avg_words = dtm.sum()/dtm.shape[0]
    
    print('The total number of words:',words.shape[0])
    
    total_word_frequency = np.sum(dtm, axis=0)
    sorted_word_indices = np.argsort(total_word_frequency)[::-1]
    top_10_word_freq = []
    for index in range(0,10):
        tuple_of_word_freq = (words[sorted_word_indices[index]],total_word_frequency[sorted_word_indices[index]])
        top_10_word_freq.append(tuple_of_word_freq)

    print('\nThe top 10 frequent words:\n',top_10_word_freq)
    print(f'\nAverage words per sentence is {avg_words:.2f}')
    print('\nThe top 10 words with highest df values:\n',top_10_word_freq)
    
    idx_of_longest_sentence = np.argmax(dtm.sum(axis=1))
    print('\nThe longest sentence:\n',sents[idx_of_longest_sentence])

    tf_idf_val_longest_sent = tf_idf[idx_of_longest_sentence, :]
    sorted_idx_tf_idf = np.argsort(tf_idf_val_longest_sent)[::-1]
    top_10_largest_tf_idf_val = sorted_idx_tf_idf[:10]

    top_10_word_freq = []
    for index in range(0,10):
        tuple_of_word_freq = (words[top_10_largest_tf_idf_val[index]],tf_idf_val_longest_sent[top_10_largest_tf_idf_val[index]])
        top_10_word_freq.append(tuple_of_word_freq)
        
    print('\nThe top 10 words with highest tf-idf vaues in the longest sentence:\n',top_10_word_freq)

    return tf_idf
    


# In[72]:


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    text = """it's a hello world!!!
           it is hello world again."""
    print("Test Question 1")
    print(tokenize(text))
    
    
    # Test Question 2
    print("\nTest Question 2")
    sents = ["it's a hello world!!!",
         "it is hello world again.",
         "world-wide problem"]
    
    dtm, all_words = get_dtm(sents, min_count = 1)
    print(dtm.shape)
    
    
    #3 Test Question 3
    print("\nTest Question 3")
    sents = pd.read_csv("sents.csv")
    dtm, words = get_dtm(sents.text, min_count = 1)
    tfidf= analyze_dtm(dtm, words, sents.text)
    


# The min_count changes the top 10 words with highest tf_idf values in the longest sentence - since, tf_idf values are impacted by df and tf values. min_count plays a crucial role in creation of dtm, if min_count is a lesser value the matrix is denser, whereas if the value increases the matrix turns out to be a sparser matrix since typically the words in the vocab might fail to meet the threshold specified by min_count.
# 
# Since, calculation of df and tf are majorly impacted by the values in dtm matrix, they end up changing drastically too. On the other hand, tf_idf will be impacted by both the parameters, which indeed impacts the values of tf_idf in the longest sentence.
