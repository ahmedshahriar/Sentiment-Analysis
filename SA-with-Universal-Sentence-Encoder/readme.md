## How to use The Exported Model

```
# use the latest `tensorflow_text` version
# !pip install tensorflow_text

import numpy as np
from numpy import newaxis

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

# load Universal Sentence Encoder Multilingual Model, v3 is ~300 MB
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
use = hub.load(module_url)

imported_model = tf.keras.models.load_model('/content/lstm_final_model.h5')

def predict_sentiment(txt):
  # generate embedding
  emb_txt = use(txt)
  
  # reshape to pass into the model
  emb_test_reshaped = emb_txt[:, newaxis, :]
  
  # predict sentiment score
  sentiment_val = np.argmax(imported_model.predict(emb_test_reshaped))
  
  # return sentiment value based on score
  return "Positive" if sentiment_val == 1 else "Negative"

sample_text = "I like the room service"
pred_sentiment = predict_sentiment(sample_text)
print(f"The sentiment of this sentence is : {pred_sentiment}")

# output : 
# The sentiment of this sentence is : Positive

```
