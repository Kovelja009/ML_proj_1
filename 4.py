"""MLProject1Exercise4a.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c6kfJGmo8RGD79p5NWlpMw4yhfsQRdPi
"""

!pip install nltk

import nltk
nltk.download()
nltk.download('stopwords')

import numpy as np
import pandas as pd

df = pd.read_csv('disaster-tweets.csv')

df.dropna(axis=1)
df.shape

"""### Data cleaning"""

from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math

# corpus = ["Jack likes action movies. 'Action movies are great' - Jack",
#           "Jack also likes to watch football games.",
#           "Jessica and Jack like to watch a movie together."]

porter = PorterStemmer()

# Cistimo korpus
print('Cleaning the corpus...')
clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))
for item in df["text"]:
  words = wordpunct_tokenize(item)
  words_lower = [w.lower() for w in words]
  words_filtered = [w for w in words_lower if w not in stop_punc]
  words_stemmed = [porter.stem(w) for w in words_filtered]
  clean_corpus.append(words_stemmed)

# # Kreiramo vokabular
print('Creating the vocab...')
vocab_set = set()
most_frequent = {}

# uzimamo unikatne reci
for item in clean_corpus:
  for word in item:
    most_frequent[word] = 0
    vocab_set.add(word)
vocab = list(vocab_set)

# uzimamo 10000 najfrekventnijih unikatnih reci
for item in clean_corpus:
  for word in item:
    most_frequent[word] += 1

sorted_dict = dict(sorted(most_frequent.items(), key=lambda item: -item[1]))

cleaned_words = list(sorted_dict.keys())
cleaned_words = cleaned_words[4:10004]
print(cleaned_words)
print('Feature vector size: ', len(vocab))
print(len(cleaned_words))
# --------------------


# # 1: Bag of Words  
def numocc_score(word, doc):
  return doc.count(word)

print('Creating BOW features...')
X = np.zeros((len(clean_corpus), len(cleaned_words)), dtype=np.float32)
for item_idx in range(len(clean_corpus)):
  item = clean_corpus[item_idx]
  for word_idx in range(len(cleaned_words)):
    word = cleaned_words[word_idx]
    cnt = numocc_score(word, item)
    X[item_idx][word_idx] = cnt
print('X:')
print(X)

import numpy as np

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
    nb_examples = X.shape[0]

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    self.priors = np.bincount(Y) / nb_examples
    print('Priors:')
    print(self.priors)

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += cnt
    print('Occurences:')
    print(occs)
    
    # Racunamo P(Rec_i|Klasa) - likelihoods
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
    print('Likelihoods:')
    print(self.like)
          
  def predict(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    # print('\"Probabilites\" for a test BoW (with log):')
    # print(probs)
    prediction = np.argmax(probs)
    return prediction

from sklearn.model_selection import train_test_split
y = df["target"].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y)


model = MultinomialNaiveBayes(nb_classes=2, nb_words=10000, pseudocount=1)
model.fit(X, y)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
y = df["target"].values

for i in range(3):
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y)
  predictions = []
  for sample in x_test:
    prediction = model.predict(sample)
    predictions.append(prediction)
  accuracy = accuracy_score(y_test, predictions)
  print('accuracy: ', accuracy)



positive = []
for i, item in enumerate(df['target']):
  if item == 1:
    positive.append(clean_corpus[i])

negative = []
for i, item in enumerate(df['target']):
  if item == 0:
    negative.append(clean_corpus[i])


pos_dict = {}
for item in positive:
  for word in item:
    pos_dict[word] = 0

for item in positive:
  for word in item:
    pos_dict[word] += 1


neg_dict = {}
for item in negative:
  for word in item:
    neg_dict[word] = 0

for item in negative:
  for word in item:
    neg_dict[word] += 1

# print(pos_dict.keys())
# print(neg_dict.keys())

pos_dict = dict(sorted(pos_dict.items(), key=lambda item: -item[1]))
neg_dict = dict(sorted(neg_dict.items(), key=lambda item: -item[1]))

pos_words = list(pos_dict.keys())[5:]
neg_words = list(neg_dict.keys())[5:]
print(pos_words[:5])
print(neg_words[:5])


#u negativnim tvitovima mozemo videti reci kao sto su love ili new a to ne podseca ni na sta na srecu
#dok kod stvarne nesrece vidimo reci kao sto su kill,disaster, flood... koje podsecaju na stvarnu nesrecu



# uzimamo unikatne reci
pos_freq = {}
neg_freq = {}
vocab_set = set()
for item in clean_corpus:
  for word in item:
    vocab_set.add(word)
vocab = list(vocab_set)

res = {}

for i in range(len(vocab)):
  if vocab[i] in pos_dict and vocab[i] in neg_dict:
    p = pos_dict[vocab[i]]
    n = neg_dict[vocab[i]]
    if p and n and p>=10 and n>=10:
        res[vocab[i]] = p/n

res= dict(sorted(res.items(), key=lambda item: -item[1]))
result = list(res.keys())
print(result[:5])
print(result[-5:])

#hocemo da nam LR metrika za neku rec bude broj sto veci jer to znaci da je ta rec veoma cesto koristi u stvarnim nesrecama
#i mozemo videti da se kill, report i latest javlja na vrh kao sto se kill i malopre javaljalo sto oznacava da ta rec u bias-u povecava sansu da ce taj tvit biti pozitivan











