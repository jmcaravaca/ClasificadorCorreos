
import json
from clasificadorcorreos import infer
import os
import pickle

class Main(object):
   def __init__(self):
      filename_model =  os.path.join("models",  "clasification_nseg.sav")
      filename_vectorizer = os.path.join("models", "vectorizer_nseg.sav")
      self.lr_tfidf = pickle.load(open(filename_model, 'rb'))
      self.tfidf_vectorizer = pickle.load(open(filename_vectorizer, 'rb'))   
      
   def predict(self, skill_input):
      import io
      file_like = io.BytesIO(skill_input)
      with file_like as f:
         text_data = f.read()      
      try:
         results = infer.infer_from_text(new_text=text_data, lr_tfidf=self.lr_tfidf,
                                       tfidf_vectorizer=self.tfidf_vectorizer)
      except Exception as e:
         results = {'prediction' : 'exception', 'confidence' : '1.0'}
      return json.dumps(results)
      
      
      
if __name__ == '__main__':
   a = Main()
   import os
   for filename in os.listdir("testdata"):
      try:
         print(f"....{filename}.....")
         with open(os.path.join("testdata", filename), 'rb') as f:
            results = a.predict(skill_input=f.read())
      except Exception as e:
         print(e)
      
      