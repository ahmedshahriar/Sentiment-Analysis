import os
from typing import Optional, Dict

import uvicorn
import tensorflow as tf
import tensorflow_hub as hub

# https://github.com/tensorflow/tensorflow/issues/38597#issuecomment-720347886
# `tensorflow_text` is to be installed and required to import as well for the TfHub to work
import tensorflow_text

from numpy import newaxis
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

PATH_SENTIMENT_MODEL = './model/lstm_sentiment_model.h5'
PATH_HUB_MODEL = "tfhub\\universal-sentence-encoder-multilingual-large_3"


class Review(BaseModel):
    """
    Review model: input for the model
    """
    review: str


class Prediction(BaseModel):
    """
    Prediction model: output of the model
    """
    sentiment_obj: Optional[Dict]


class SentimentModel:
    """
    Sentiment model: LSTM model
    """
    model: Optional[tf.keras.Model] = None
    hub_model: Optional = None

    def load_model(self):
        """Loads the model"""
        self.model = tf.keras.models.load_model(PATH_SENTIMENT_MODEL)
        self.hub_model = hub.load(PATH_HUB_MODEL)

    async def predict(self, input_data: Review) -> Prediction:  # dependency
        """Runs a prediction"""
        if not self.model:
            # raise RuntimeError("Model is not loaded")
            raise HTTPException(status_code=400, detail="Model is not loaded")
        input_data = input_data.dict()
        emb_txt = self.hub_model(input_data['review'])
        emb_test_reshaped = emb_txt[:, newaxis, :]
        sentiment_val = (self.model.predict(emb_test_reshaped) > 0.5).astype('int32')
        score = self.model.predict(emb_test_reshaped).flatten()[0]
        #     # orjson doesn't support serializing individual numpy input_data types yet, converting to python float
        #     # https://github.com/tiangolo/fastapi/issues/1733
        sentiment_dic = {"prediction": "positive" if sentiment_val == 1 else "negative", "score": float(score)}
        return Prediction(sentiment_obj=sentiment_dic)


app = FastAPI()

sentiment_model = SentimentModel()


@app.get('/')
def index():
    return {'message': 'This is Hotel Reviews Classification API!'}


@app.post('/sentiment')
async def predict(output: Prediction = Depends(sentiment_model.predict, )) -> Prediction:
    return output


@app.on_event("startup")
async def startup():
    sentiment_model.load_model()


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
