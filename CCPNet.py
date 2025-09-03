import torch
import torch.nn as nn
import torch.nn.functional as F

class CCPNet(nn.Module):
    def __init__(self, numeric_dim, categorical_vocab_sizes, embed_dim=8, hidden_dim=32, num_heads=2):
        """
        CCP-Net: Self-Attention + BiLSTM + CNN for churn prediction.

        Args:
            numeric_dim (int): Number of numeric features
            categorical_vocab_sizes (dict): {feature_name: vocab_size} for categorical features
            embed_dim (int): Embedding size for categorical features
            hidden_dim (int): Hidden size for BiLSTM and CNN
            num_heads (int): Number of attention heads
        """
        super(CCPNet, self).__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, embed_dim)
            for name, vocab_size in categorical_vocab_sizes.items()
        })

        # Projection for numeric features into same space
        self.num_proj = nn.Linear(numeric_dim, embed_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # BiLSTM
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # CNN
        self.conv1d = nn.Conv1d(in_channels=2*hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # Classifier
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, X_num, X_cat_dict):
        """
        Args:
            X_num: Tensor [batch_size, numeric_dim]
            X_cat_dict: Dict {feature_name: Tensor[batch_size]} with categorical features
        """

        # Embeddings for categorical
        cat_embeds = []
        for name, emb_layer in self.embeddings.items():
            emb = emb_layer(X_cat_dict[name])   # [batch, embed_dim]
            cat_embeds.append(emb)

        # Project numeric features
        num_emb = self.num_proj(X_num)  # [batch, embed_dim]

        # Combine into sequence: [batch, seq_len, embed_dim]
        feature_seq = torch.stack(cat_embeds + [num_emb], dim=1)

        # Self-attention
        attn_out, _ = self.attn(feature_seq, feature_seq, feature_seq)
        x = self.attn_norm(attn_out + feature_seq)  # residual + norm

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, 2*hidden_dim]

        # CNN expects [batch, channels, seq_len]
        cnn_in = lstm_out.transpose(1, 2)  # [batch, 2*hidden_dim, seq_len]
        cnn_out = self.conv1d(cnn_in)      # [batch, hidden_dim, seq_len]
        cnn_out = F.relu(cnn_out)
        cnn_out = torch.max(cnn_out, dim=2).values  # GlobalMaxPooling1D

        # Dense layers
        x = F.relu(self.fc1(cnn_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out = torch.sigmoid(self.fc_out(x))

        return out.squeeze()
