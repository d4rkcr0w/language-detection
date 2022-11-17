from typing import Union
from fastapi import FastAPI, HTTPException
from starlette import status
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)


@app.get("/")
def sentiment_analysis(text: Union[str] = ""):
    if len(text) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The text cannot be empty",
        )

    return classifier(text)[0]
