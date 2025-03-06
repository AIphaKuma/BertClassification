import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import load_wikitext_dataset
from data.tokenizer import tokenize_text, build_vocab
from models.transformer_model import TransformerLanguageModel
from utils.helpers import get_batch

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_text = load_wikitext_dataset(
        dataset_name=config.get("dataset_name", "wikitext-2-raw-v1"),
        split="train"
    )
    tokens = tokenize_text(full_text, config["tokenizer_type"])
    stoi, itos = build_vocab(tokens)
    vocab_size = len(stoi)
    data = [stoi[t] for t in tokens]
    
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        embed_size=config["embed_size"],
        num_heads=config["num_heads"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1)
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["num_epochs"]):
        model.train()
        inputs, targets = get_batch(data, config["batch_size"], config["seq_length"])
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)  # (batch_size*seq_length, vocab_size)
        loss = criterion(outputs, targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), config["model_save_path"])
    torch.save({"stoi": stoi, "itos": itos}, config["vocab_save_path"])
    print("Modèle et vocab sauvegardés.")

if __name__ == '__main__':
    main()