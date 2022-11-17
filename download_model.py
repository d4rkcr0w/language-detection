from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "papluca/xlm-roberta-base-language-detection"
model_path = "./model"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
