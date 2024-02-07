import pickle
import os
import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text


# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('spanish')]
    return ' '.join(a)

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)
    
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

def infer_from_text(new_text):
    filename_model =  os.path.join("models",  "clasification_nseg.sav")
    filename_vectorizer = os.path.join("models", "vectorizer_nseg.sav")
    lr_tfidf = pickle.load(open(filename_model, 'rb'))
    tfidf_vectorizer = pickle.load(open(filename_vectorizer, 'rb'))    
    X_new = tfidf_vectorizer.transform([new_text])
    prediction = lr_tfidf.predict(X_new)
    prediction_prob = lr_tfidf.predict_proba(X_new)
    confidence = dict(zip(lr_tfidf.classes_, prediction_prob[0]))
    results = {'prediction' : prediction,
               'confidence' : confidence}
    print(f"{prediction=} and {confidence=}")
    return results
    