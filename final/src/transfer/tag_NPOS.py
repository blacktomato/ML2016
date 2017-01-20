from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import feature_extraction
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from string import digits
import numpy as np
import sklearn
import random
import string
import nltk
import time
import sys
import csv
import re
np.set_printoptions(threshold=np.nan)

stop_words = {'a', "a's", 'able', 'about', 'above', 'according', 'accordingly',
              'across', 'actually', 'after', 'afterwards', 'again', 'against',
              "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along',
              'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
              'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone',
              'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear',
              'appreciate', 'appropriate', 'are', "aren't", 'around', 'as',
              'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away',
              'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes',
              'becoming', 'been', 'before', 'beforehand', 'behind', 'being',
              'believe', 'below', 'beside', 'besides', 'best', 'better',
              'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon",
              "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause',
              'causes', 'certain', 'certainly', 'changes', 'clearly', 'co',
              'com', 'come', 'comes', 'concerning', 'consequently', 'consider',
              'considering', 'contain', 'containing', 'contains',
              'corresponding', 'could', "couldn't", 'course', 'currently', 'd',
              'definitely', 'described', 'despite', 'did', "didn't",
              'different', 'do', 'does', "doesn't", 'doing', "don't", 'done',
              'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight',
              'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially',
              'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone',
              'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',
              'f', 'far', 'few', 'fifth', 'first', 'five', 'followed',
              'following', 'follows', 'for', 'former', 'formerly', 'forth',
              'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets',
              'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got',
              'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly',
              'has', "hasn't", 'have', "haven't", 'having', 'he', "he's",
              'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter',
              'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him',
              'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit',
              'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if',
              'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed',
              'indicate', 'indicated', 'indicates', 'inner', 'insofar',
              'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll",
              "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps',
              'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later',
              'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's",
              'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks',
              'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean',
              'meanwhile', 'merely', 'might', 'more', 'moreover', 'most',
              'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely',
              'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither',
              'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody',
              'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing',
              'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often',
              'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only',
              'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our',
              'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own',
              'p', 'particular', 'particularly', 'per', 'perhaps', 'placed',
              'please', 'plus', 'possible', 'presumably', 'probably',
              'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're',
              'really', 'reasonably', 'regarding', 'regardless', 'regards',
              'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw',
              'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing',
              'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves',
              'sensible', 'sent', 'serious', 'seriously', 'seven', 'several',
              'shall', 'she', 'should', "shouldn't", 'since', 'sigma', 'six', 'so',
              'some', 'somebody', 'somehow', 'someone', 'something', 'sometime',
              'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry',
              'specified', 'specify', 'specifying', 'still', 'sub', 'successfully', 'such',
              'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th',
              'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats',
              'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence',
              'there', "there's", 'thereafter', 'thereby', 'therefore',
              'therein', 'theres', 'thereupon', 'these', 'they', "they'd",
              "they'll", "they're", "they've", 'think', 'third', 'this',
              'thorough', 'thoroughly', 'those', 'though', 'three', 'through',
              'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took',
              'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying',
              'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless',
              'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used',
              'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value',
              'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants',
              'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've",
              'welcome', 'well', 'went', 'were', "weren't", 'what', "what's",
              'whatever', 'when', 'whence', 'whenever', 'where', "where's",
              'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
              'wherever', 'whether', 'which', 'while', 'whither', 'who',
              "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
              'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder',
              'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you',
              "you'd", "you'll", "you're", "you've", 'your', 'yours',
              'yourself', 'yourselves', 'z', 'zero', '', 'iii', 'll', 'aa',
              'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
              'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron',
              'pi', 'pho', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
              'omega', 'aa', 'aaa', 'dt', 'dx', 'xdx', 'dxd', 'zz', 'dy', 'cos', 'sin',
              'tan', 'cot', 'sec', 'csc', 'ge', 'le', 'nu', 'ds', 'dv', 'ln',
              'rt', 'mm', 'mu', 'ir', 'ccc', 'pb', 'kg', 'kgs', 'mg', 'mgh',
              'hz', 'thz', 'ghz', 'mhz', 'centermetre', 'find', 'http',
              'cm', 'cmb', 'hat', 'sqrt', 'left', 'right' 'sf', 'bar',
              'abcd', 'cd', 'spinning', 'km', 'meaningful', 'meaningfully', 'uniform',
              'ww', 'gg', 'amp', 'frac', 'vec', 'rod', 'hot', 'cold', 'don', 'www',
              'http', 'couldn', 'shouldn', 'wouldn', 'isn', 'doesn' , 'dont',
              'wasn', 'car', 'cover', 'noodles', 'add', 'wait', 'salt', 'end',
              'iss', 'whats', 'whos', 'whens', 'wheres', 'whichs' 'hows', 'back',
              'start', 'starting', 'coil', 'zee'}

def get_words(text):
    """Split a sentence into a list of words
    Input: string,
    Output: list
    """
    word_split = re.compile('[^a-zA-Z\-]')
    return [word.strip().lower() for word in word_split.split(text)]

################################################################################
# Declaration of variables for future usage                                    #
# The file that has been preprocessed should be named as 'clean_test.csv'      #
################################################################################
f = open('clean_' + sys.argv[1], 'r')
reader = csv.reader(f)
corpus = []
id_list = []
reader.next()

################################################################################
# Read in file and clean up numbers and punctuations in title and content      #
# Store them into corpus                                                       #
################################################################################
print ('Reading in file...', end='              ')
start_time = time.time()
for row in reader:
   id_list.append(row[0])
   words = get_words(row[1]+row[2])
   clean_row = ''
   for word in words:
      if word.isalpha() and len(word) > 2:
         clean_row = clean_row + ' ' + word
   corpus.append(clean_row)
print (str(time.time() - start_time) + 's')


################################################################################
# Get phrases in the corpus using CountVectorizer by using only bigram         #
# Get only 600 possible phrase candidates (which can be modified and tested)   #
# Store these phrases in 'phrase_candidates'                                   #
################################################################################
print ('Getting phrases...', end = '              ')
start_time = time.time()
vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.3, min_df=50, max_features=10000, ngram_range=(2,2))
tfidf = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

tfidf = tfidf.toarray()
tfidf = np.asarray(tfidf)
a = np.sum(tfidf, 0)
phrase_indices = a.argsort()[-600:][::-1]
phrase_candidates = []
for i in phrase_indices:
    phrase_candidates.append(feature_names[i])
print (str(time.time() - start_time) + 's')

################################################################################
# Read in the preprocessed file again and add a '-' between two words that can #
# be combined to a phrase candidate. (ex: black hole -> black-hole)            #
# Write the result to a file named 'phrase_test.csv'                           #
################################################################################
print ('Regenerating contents...', end = '        ')
start_time = time.time()
f.close()
f = open('clean_' + sys.argv[1], 'r')
reader = csv.reader(f)
corpus = []
reader.next()
out_phrase = open('phrase_' + sys.argv[1], 'w')
writer = csv.writer(out_phrase)
first = ['id', 'title', 'content']
writer.writerow(first)

for row in reader:
    title_sentence = row[1]
    content_sentence = row[2]
    for phrase in phrase_candidates:
        a = phrase.split()
        title_sentence = title_sentence.replace(phrase, a[0] + '-' + a[1])
        content_sentence = content_sentence. replace(phrase, a[0] + '-' + a[1])
    corpus.append(title_sentence + ' ' + content_sentence)
    out_row = [str(row[0]), str(title_sentence), str(content_sentence)]
    writer.writerow(out_row)
out_phrase.close()

for i in range(len(phrase_candidates)):
    a = phrase_candidates[i].split()
    phrase_candidates[i] = a[0] + "-" + a[1]
print (str(time.time() - start_time) + 's')
f.close()

################################################################################
# Use POS tag to clean up some irrelevant words in title and content, leaving  #
# only "noun-type" words.                                                      #
################################################################################
print ('Cleaning texts...', end = '              ')
start_time = time.time()
f = open('phrase_' + sys.argv[1], 'r')
reader = csv.reader(f)
reader.next()
for row in reader:
   words = get_words(row[1]+row[2])
   clean_row = ''
   for word in words:
      if word.isalpha() and len(word) > 2:
         clean_row = clean_row + ' ' + word
   corpus.append(clean_row)

vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.3, min_df=50, max_features=10000, ngram_range=(1,1))
tfidf = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

tfidf = tfidf.toarray()
tfidf = np.asarray(tfidf)
a = np.sum(tfidf, 0)
phrase_indices = a.argsort()[-600:][::-1]
phrase_candidates = []
for i in phrase_indices:
    phrase_candidates.append(feature_names[i])

f.close()
f = open('phrase_' + sys.argv[1], 'r')
reader = csv.reader(f)
reader.next()

outfile = open('output_exp_' + sys.argv[1], 'w')
writer = csv.writer(outfile)
first = ['id', 'tags']
writer.writerow(first)
count = 0

for row in reader:
    # title
    sentence = str(row[1])
    s_list = get_words(sentence)
    sentence = ''
    for i in s_list:
        sentence = sentence + i + ' '
    sentence = nltk.word_tokenize(sentence)

    # content
    sentence2 = str(row[2])
    s_list2 = get_words(sentence2)
    sentence2 = ''
    for i in s_list2:
        sentence2 = sentence2 + i + ' '
    sentence2 = nltk.word_tokenize(sentence2)
    sent = pos_tag(sentence)
    sent2 = pos_tag(sentence2)
    # print (sentence, sent)
    # print (sentence2, sent2)

    new_title = ''
    new_content = ''

    # Leave only "noun-type" words in title
    for s in sent:
        if len(s[0]) > 2 and not s[0] in stop_words:
            if '-' in s[0] and not len(filter(None, s[0].split('-'))) == 0:
                if len(filter(None, s[0].split('-'))) == 1:
                    a = str(filter(None, s[0].split('-'))[0])
                    new_title = new_title + a + ' '
                else:
                    new_title = new_title+s[0] + ' '
            elif not '-' in s[0]:
                new_title = new_title + s[0] + ' '

    # Leave only "noun-type" words in content
    for s in sent2:
        if len(s[0]) > 2 and not s[0] in stop_words:
            if '-' in s[0] and not len(filter(None, s[0].split('-'))) == 0:
                if len(filter(None, s[0].split('-'))) == 1:
                    a = str(filter(None, s[0].split('-'))[0])
                    new_content = new_content + a + ' '
                else:
                    new_content = new_content + s[0] + ' '
            elif not '-' in s[0]:
                new_content = new_content+s[0] + ' '

    # Coution!!!!
    # Add only "phrases" in content!
    extra = ''
    for word in new_title.split():
        if word in phrase_candidates:
            extra = extra + word + ' '
    for word in new_content.split():
        if word in phrase_candidates:
            extra = extra + word + ' '

    tag = extra
    tag = list(set(tag.split()))

    # If there is no output tag (all words are removed from above process)
    rand_words = str(row[1]).split()
    rand_words = list(set(rand_words) - stop_words)
    if len(tag) == 0 and not len(rand_words) == 0:
        tag = get_words(rand_words[random.randint(0, len(rand_words)-1)])
        tag = filter(None, tag)
    count += 1
    if count % 1000 == 0:
        print (count)

    # output
    out_row = [row[0], " ".join(tag)]
    writer.writerow(out_row)

print (str(time.time() - start_time) + 's')
outfile.close()

################################################################################
# From the output file above (output_exp.csv), get the abbrevation of all      #
# predicted pharses (ex: quantum-field-theory -> qft). Subsitute abbreviated   #
# tag in output with its original name. (ex: qtf -> quantum-field-theory)      #
################################################################################
print ('Substitute abbreviations...', end = '     ')
start_time = time.time()
f = open('output_exp_' + sys.argv[1], 'r')
reader = csv.reader(f)
reader.next()

phrases = []
phrase_abbr = {}
for row in reader:
    tags = row[1].split()
    for tag in tags:
        abbr = ''
        if '-' in tag:
            phrases.append(tag)
for phrase in phrases:
    words = filter(None, phrase.split('-'))
    abbr = ''
    for word in words:
        abbr = abbr + word[0]
    phrase_abbr[abbr] = phrase


f.close()
f = open('output_exp_' + sys.argv[1], 'r')
reader = csv.reader(f)
reader.next()

outfile = open('prediction_' + sys.argv[1], 'w')
writer = csv.writer(outfile)

first = ['id', 'tags']
writer.writerow(first)

prediction_tags = {}
for row in reader:
    out_tag = []
    tags = list(set(row[1].split()))
    for tag in tags:
        if (len(tag) == 2 or len(tag) == 3) and tag in phrase_abbr:
            out_tag.append(phrase_abbr[tag])
        else:
            out_tag.append(tag)
    out_row = [row[0], " ".join(out_tag)]
    writer.writerow(out_row)
    prediction_tags[int(row[0])] = out_tag

print (str(time.time() - start_time) + 's')
outfile.close()
