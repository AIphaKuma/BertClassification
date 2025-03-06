import random
import torch

def get_batch(data, batch_size, seq_length):
    """
    Génère un batch d'inputs et de cibles à partir d'une séquence d'indices.
    Retourne deux tensors de forme (batch_size, seq_length).
    """
    inputs = []
    targets = []
    data_len = len(data)
    for _ in range(batch_size):
        start_idx = random.randint(0, data_len - seq_length - 1)
        inp_seq = data[start_idx:start_idx + seq_length]
        tgt_seq = data[start_idx + 1:start_idx + seq_length + 1]
        inputs.append(inp_seq)
        targets.append(tgt_seq)
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_tensor, targets_tensor