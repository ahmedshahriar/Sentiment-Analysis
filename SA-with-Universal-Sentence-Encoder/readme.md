[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/ahmedshahriar/Sentiment-Analysis/blob/main/SA-with-Universal-Sentence-Encoder/SA-Hotel-Reviews.ipynb) [![Open in HTML](https://img.shields.io/badge/Html-Open%20Notebook-blue?logo=HTML5)](https://nbviewer.org/github/ahmedshahriar/Sentiment-Analysis/blob/main/SA-with-Universal-Sentence-Encoder/SA-Hotel-Reviews.html)

# Introduction

The objective of this project is to perform sentiment analysis (only **positive** and **negative**) on a large hotel review dataset.

The review texts are embedded using universal sentence encoder model from Tensorflow HUB

The final LSTM model achieved an accuracy of **~81%** in Test Dataset (**75:25** split)

## Sneak Peek Into Data
![image](https://user-images.githubusercontent.com/40615350/141520815-5dda08ec-62e5-4eed-b7cc-ba28e55fb42b.png)


## Dataset Source
* [Kaggle Dataset URL - 515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)

## Text Embedding Model
* [universal-sentence-encoder-multilingual-large-v3](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)

## How to Use The Exported Model

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


## Built With
```
tensorflow==2.7.0
tensorflow_text==0.12.0
tensorflow-hub==0.12.0
keras==2.7.0
keras-vis==0.4.1
scikit-learn==0.22.2.post1
```
