
import json
from ClasificadorCorreos import infer

class Main(object):
   def __init__(self):
      pass
   def predict(self, skill_input):
      import io
      file_like = io.BytesIO(skill_input)
      with file_like as f:
         text_data = f.read()
      results = infer.infer_from_text(text_data)
      return json.dumps(results)
      
      
      
if __name__ == '__main__':
   a = Main()
   with open("message_474.txt") as f:
         text_data = f.read()
   results = infer.infer_from_text(text_data)
   print(results)