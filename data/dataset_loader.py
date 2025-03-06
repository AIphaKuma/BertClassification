from datasets import load_dataset

def load_wikitext_dataset(dataset_name="wikitext-2-raw-v1", split="train"):
    try:
        dataset = load_dataset("wikitext", dataset_name, split=split)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset : {e}")
    
    full_text = " ".join(dataset["text"])
    full_text = full_text.replace("@-@", "-")
    return full_text