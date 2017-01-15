import numpy as np
import re
import sys
import csv
import string
import operator
import unicodedata
from numpy import bincount
from string import digits
from bs4 import BeautifulSoup
np.set_printoptions(threshold=np.nan)

f = open('test.csv', 'r')
reader = csv.reader(f)
corpus = []
id_list = []
reader.next()

outfile = open('clean_test.csv', 'w')
writer = csv.writer(outfile)
first = ['id', 'title', 'content']
writer.writerow(first)
count = 0

for row in reader:
    id_list.append(row[0])
    title = unicodedata.normalize('NFKD', BeautifulSoup(row[1]).text).encode('ascii', 'ignore')
    content = unicodedata.normalize('NFKD', BeautifulSoup(row[2]).text).encode('ascii', 'ignore')
    title_sentences = title.split('\n')
    content_sentences = content.split('\n')

    out_title = ''
    flag = False
    for sentence in title_sentences:
        # Remove everything between \begin{equation} and \end{equation}
        if (sentence == '$$' or sentence == '\\begin{equation}' or sentence == '\\begin{align}') and not flag:
            flag = True
        elif (sentence == '$$' or sentence == '\\end{equation}' or sentence == '\\end{align}') and flag:
            flag = False
        elif flag:
            continue
        # Remove everything between two '$$', which is an equation
        sentence = re.sub(r'\$\$.*?\$\$', '', sentence)
        sentence = re.sub(r'\$.*?\$', '', sentence)
        sentence = re.sub(r'\\.*?\\', '', sentence)
        sentence = sentence.lower()
        # correct the typo "schrdinger"
        sentence = sentence.replace("schrdinger", "schroedinger")
        sentence = sentence.replace("schrodinger", "schroedinger")
        if not (sentence == '\\begin{equation}' or sentence == '\\end{equation}'
                or sentence == '\\begin{align}' or sentence == '\\end{align}'):
            out_title = out_title + sentence.lower() + ' '

    out_content = ''
    flag = False
    for sentence in content_sentences:
        # Remove everything between \begin{equation} and \end{equation}
        if (sentence == '$$' or sentence == '\\begin{equation}' or sentence == '\\begin{align}') and not flag:
            flag = True
        elif (sentence == '$$' or sentence == '\\end{equation}' or sentence == '\\end{align}') and flag:
            flag = False
        elif flag:
            continue
        # Remove everything between two '$$', which is an equation
        sentence = re.sub(r'\$\$.*?\$\$', '', sentence)
        sentence = re.sub(r'\$.*?\$', '', sentence)
        sentence = re.sub(r'\\.*?\\', '', sentence)
        # correct the typo "schrdinger"
        sentence = sentence.lower()
        sentence = sentence.replace("schrdinger", "schroedinger")
        sentence = sentence.replace("schrodinger", "schroedinger")
        if not (sentence == '\\begin{equation}' or sentence == '\\end{equation}'
                or sentence == '\\begin{align}' or sentence == '\\end{align}'):
            out_content = out_content + sentence.lower() + ' '

    count += 1
    if count % 1000 == 0:
        print count

    out_row = [str(row[0]), str(out_title), str(out_content)]
    writer.writerow(out_row)
