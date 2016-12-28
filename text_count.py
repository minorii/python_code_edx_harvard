# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:57:53 2016

@author: minori
"""

#os.listdir("C:/Users/minori/Desktop/GNE-158")


        
text = 'For those who want to dive deeper into data science and machine learning, I would recommend the course Programming with Python for Data Science from Microsoft that will be available in edX starting January 1, 2017. You can also browse courses that are part of their Data Science Curriculum.'
def count_words(text):
    text = text.lower()
    skips = [',', '.', ';', ':', '"', "'"]
    for ch in skips:
        text = text.replace(ch, '')
    word_counts = {}
    for word in text.split(' '):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts
    
from collections import Counter

def count_words_fast(text):
    text = text.lower()
    skips = [',', '.', ';', ':', '"', "'"]
    for ch in skips:
        text = text.replace(ch, '')
    word_counts = Counter(text.split(' '))
    return word_counts
    
def read_book(title_path):
    with open(title_path, 'r', encoding = 'utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n', '').replace('\r', '')
    return text
    
#text = read_book(title_path)
#len(text)
#ind = text.find("what's the fuck?")
#sampletext = text[ind:ind + 1000]

def words_stats(word_counts):
    num_unque = len(word_counts)
    counts = word_counts.values()
    return (num_unque, counts)

import pandas as pd    
import os
book_dir = './Books'
stats = pd.DataFrame(columns = ('language', 'author', 'title', 'length', 'unique'))
title_num = 1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + '/' + language):
        for title in os.listdir(book_dir+ '/' + language + '/' + author):
            input_file = book_dir+ '/' + language + '/' + author + '/' + title
#            print(input_file)
            text = read_book(input_file)
            (num_unque, counts) = words_stats(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), title.replace('.txt', ''), sum(counts), num_unque
            title_num += 1
#print(stats.head())
#print(stats.tail())
#print(stats.length)
#print(stats.unique)

import pylab as plt
#plt.figure()
#plt.plot(stats.length, stats.unique, 'bo')
#plt.figure()
#plt.loglog(stats.length, stats.unique, 'ro')
        
def word_count_distribution(text):
    count_word = count_words_fast(text)
    distribution = {}
    for i in range(1, max(count_word.values())+1):
        x = 0
        if i in count_word.values():
            for j in count_word:
                if count_word[j] == i:
                    x += 1
            distribution[i] = x
    return distribution
#distribution = word_count_distribution(text)
#print(distribution)

def word_count_distribution_fast(text):
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution

#print(dict(word_count_distribution_fast(text)) == distribution)
#import pandas as pd
#table = pd.DataFrame(columns = ('name', 'age'))
#table.loc[1] = 'James', 22
#table.loc[2] = 'Jess', 32
#print(table)
#print(table.columns)
        
        
        
        
        
        
        
        
        
        
    