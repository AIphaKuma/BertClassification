import torch
import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    """
    Modèle de langage basé sur un Transformer Decoder.
    """
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(positions)
        embedded = token_emb + pos_emb
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, x.device)
        out = self.transformer_decoder(embedded, embedded, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)
        logits = self.fc_out(out)
        return logits.reshape(-1, logits.size(-1))
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask