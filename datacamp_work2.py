# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:29:31 2016

@author: minori
"""
import pylab as plt
import pandas as pd    
import os
from collections import Counter

def word_count_distribution(text):
    count_words = count_words_fast(text)
    return Counter(count_words.values())
    
    
def more_frequent(distribution):
    sum_all = sum(distribution.values())
    distribution_copy = distribution.copy()
    for i in distribution:
        distribution_copy[i] /= sum_all
    return distribution_copy
    
hamlets = pd.DataFrame(columns = ('language', 'distribution'))## Enter code here! ###
book_dir = "./Books"
title_num = 1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + '/' + language):
        for title in os.listdir(book_dir+ '/' + language + '/' + author):
            if title == "Richard III.txt":
                inputfile = book_dir+ '/' + language + '/' + author + '/' + title 
                text = read_book(inputfile)
                distribution = word_count_distribution(text)## Enter code here! ###
                hamlets.loc[title_num] = language, distribution
                title_num += 1
                
colors = ["crimson", "forestgreen", "blueviolet"]
handles, hamlet_languages = [], []
for index in range(hamlets.shape[0]):
    language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]
    dist = more_frequent(distribution)
    plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
        reverse = True), color = colors[index], linewidth = 2)
    handles.append(plot)
    hamlet_languages.append(language)
plt.title("Word Frequencies in Hamlet Translations")
xlim    = [0, 2e3]
xlabel  = "Frequency of Word $W$"
ylabel  = "Fraction of Words\nWith Greater Frequency than $W$"
#plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
# show your plot using `plt.show`!
#plt.savefig('Richard III.pdf')
plt.show()

