from flask import Flask, render_template, request, jsonify
import yaml
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd

app = Flask(__name__)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

model_name = "models/distilbert-capital-classification/checkpoint-375"
tokenizer = DistilBertTokenizerFast.from_pretrained("models/tokenizer_distilbert")
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.eval()
dataset_path = config.get("dataset_name", "data/dataset_bert.csv")
df = pd.read_csv(dataset_path)

capitales_dict = dict(zip(df["question"], df["capitale"]))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/qa", methods=["POST"])
def qa():
    question = request.form.get("question", "")

    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    if prediction == 1 and question in capitales_dict:
        response = f"La capitale correcte est {capitales_dict[question]}"
    else:
        response = "Je ne connais pas cette capitale."

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)