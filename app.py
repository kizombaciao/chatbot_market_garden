# https://www.youtube.com/watch?v=QpMsT0WuIuI

from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

# ignore any warning messages   
#warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

article = Article('https://en.wikipedia.org/wiki/Operation_Market_Garden')
article.download()
article.parse()
article.nlp()
corpus = article.text

print(corpus)

text = corpus
# convert the text into a list of sentences
sent_tokens = nltk.sent_tokenize(text)
# print the list of sentences
print(sent_tokens)

# print the punctuations
print(string.punctuation)

# create a dictionary (key:value) pair to remove punctuations
remove_punct_dict = dict((punct, None) for punct in string.punctuation)
print(remove_punct_dict)

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
print(remove_punct_dict)

# returns a list of lemmatized lower case words after removing punctuations
# https://www.youtube.com/watch?v=uoHVztKY6S4
def LemNormalize(text):
  return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

print(LemNormalize(text))

# keyword matching

# greeting inputs
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "hey"]

# greeting responses
GREETING_RESPONSES = ["howdy", "hi", "hey", "hello", "hey there"]

# returns a random greeting response
def greeting(sentence):
  for word in sentence.split():
    if word.lower() in GREETING_INPUTS:
      return random.choice(GREETING_RESPONSES)

# generate the response
def response(user_response):

  #user_response = 'What is chronic kidney disease'
  user_response = user_response.lower()
  #print(user_response)

  robo_response = ''

  # append the users response to the sentence list
  sent_tokens.append(user_response)
  #print(sent_tokens)

  # create a TfidfVectorizer object
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')

  # convert the text to a matrix of TF-IDF features
  tfidf = TfidfVec.fit_transform(sent_tokens)
  #print(tfidf)

  # get the measure of similarity (similarity scores)
  # note, tfidf[-1] reflects the user_response
  vals = cosine_similarity(tfidf[-1], tfidf)
  #print(vals)

  # get the index of the most similar text/sentence to the users response
  # the last one would be the user_response
  # so you want the second to the last one
  idx = vals.argsort()[0][-2]

  # reduce the dimensionality of vals
  flat = vals.flatten()

  # sort the list in ascending order
  flat.sort()

  # get the most similar score to the user response
  score = flat[-2]
  #print(score)

  # if the variable score is 0, then their is no text similar to the users response
  if (score == 0):
    robo_response = robo_response + "I apologize, I don't understand."
  else:
    robo_response = robo_response + sent_tokens[idx]
  #print(robo_response)

  # remove the users response from the sentence tokens list
  sent_tokens.remove(user_response)
  
  return robo_response

flag = True
print("Enter your queries on Market Garden Operation in WW2.\nType bye to end.")
while (flag == True):
  user_response = input()
  user_response = user_response.lower()
  if (user_response != 'bye'):
    if (greeting(user_response) != None):
      print("Answer: " + greeting(user_response))
      print("Next query?")
    else:
      print("Answer: " + response(user_response))
      print("Next query?")
  else:
    flag = False
    print("Bye.")

