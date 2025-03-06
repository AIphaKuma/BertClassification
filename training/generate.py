import torch
import yaml
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.langchain_rag import RAGModel

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    model_name = config.get("rag_model_name", "facebook/rag-sequence-base")
    max_length = config.get("max_length", 50)
    use_dummy_dataset = config.get("use_dummy_dataset", False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Le prompt ou la question à poser")
    parser.add_argument("--max_length", type=int, default=max_length,
                        help="Longueur maximale de la réponse générée")
    args = parser.parse_args()
    
    # Instancier le modèle RAG
    rag_model = RAGModel(model_name=model_name, use_dummy_dataset=use_dummy_dataset)
    
    responses = rag_model.generate(args.prompt, max_length=args.max_length)
    answer = responses[0]
    print("Réponse :", answer)

if __name__ == '__main__':
    main()