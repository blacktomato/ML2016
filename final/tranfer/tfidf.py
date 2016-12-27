import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import feature_extraction
import numpy as np
import re
import sys
import csv
import string
from numpy import bincount
import random
from string import digits
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
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
              'hz', 'thz', 'ghz', 'mhz', 'acoustic', 'temperatures', 'centermetre',
              'cm', 'cmb', 'hat', 'sqrt', 'left', 'right', 'magnet', 'sf', 'bar',
              'abcd', 'cd', 'spinning', 'km', 'electric', 'meaningful', 'meaningfully',
              'ww', 'gg', 'amp', 'frac', 'vec', 'big', 'rod', 'hot', 'cold', 'don',
              'find', 'http', 'youtube', 'subject', 'subjects', }

def clean_html(raw_html):
   cleanr = re.compile('<.*?>')
   cleantext = re.sub(cleanr, '', raw_html)
   cleantext = re.sub(r'\$.*?\$', '', cleantext)
   return cleantext

def get_words(text):
   word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')
   return [word.strip().lower() for word in word_split.split(text)]

stemmer = PorterStemmer()

reload(sys)
sys.setdefaultencoding('utf8')
f = open('test.csv', 'r')
reader = csv.reader(f)
corpus = []
id_list = []
reader.next()
for row in reader:
   id_list.append(row[0])
   row = clean_html(row[1]) + clean_html(row[2])
   words = get_words(row)
   clean_row = ''
   for i in words:
      if i.isalpha() and len(i) > 2:
         clean_row = clean_row + ' ' + i

   corpus.append(clean_row)

vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.3, min_df=20, max_features=40000, ngram_range=(1,2))
tfidf = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

tfidf = tfidf.toarray()
tfidf = np.asarray(tfidf)

outfile = open('output_1.csv', 'w')
writer = csv.writer(outfile)
first = ['id', 'tags']
writer.writerow(first)
for i in range(tfidf.shape[0]):
    top_indices = tfidf[i].argsort()[-3:][::-1]
    candidate = []
    tag = []
    for j in top_indices:
        if not len(feature_names[j].split()) == 2:
            tag.append(feature_names[j])
    for j in top_indices:
        if len(feature_names[j].split()) == 2:
            a = feature_names[j].split()
            """
            if a[0] in tag and not a[0] == a[1]:
                tag.remove(a[0])
            if a[1] in tag and not a[0] == a[1]:
                tag.remove(a[1])
            """
            if not a[0] == a[1]:
                tag.append(a[0] + '-' + a[1])
    row = [id_list[i], " ".join(tag)]
    writer.writerow(row)

