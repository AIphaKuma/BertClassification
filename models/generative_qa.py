from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline
import torch

class CapitalBERTClassifier:
    def __init__(self, model_name="distilbert-capital-classification"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def answer(self, question):
        capitales = [
            "Paris", "Londres", "Tokyo", "Séoul", "Washington", "New York", "Rome", "Milan",
            "Bruxelles", "Berlin", "Madrid", "Pékin", "Ottawa", "Brasilia", "Moscou", "New Delhi",
            "Canberra", "Pyongyang", "Ulaanbaatar", "Vientiane", "Phnom Penh", "Bangkok",
            "Sydney", "Canberra"
        ]

        best_capital = None
        best_score = 0

        for capitale in capitales:
            inputs = self.tokenizer(f"{question} {capitale}", return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            score_capitale = probabilities[0][1].item()

            print(f"🧐 Score pour {capitale} : {score_capitale}")

            if score_capitale > best_score:
                best_score = score_capitale
                best_capital = capitale

        if best_score > 0.5:
            return best_capital
        else:
            return "Je ne connais pas cette capitale."