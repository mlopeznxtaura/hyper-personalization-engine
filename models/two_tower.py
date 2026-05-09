"""
HuggingFace two-tower retrieval model for recommendation.
User tower + item tower trained with contrastive loss.
SDKs: HuggingFace Transformers, PyTorch, W&B, MLflow
"""
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
import wandb
import mlflow
import mlflow.pytorch


@dataclass
class TwoTowerConfig:
    n_users: int = 100_000
    n_items: int = 500_000
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    tower_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 10
    temperature: float = 0.07    # InfoNCE temperature
    n_negative_samples: int = 64
    device: str = "cuda"
    wandb_project: str = "hyper-personalization-engine"


class UserTower(nn.Module):
    """User representation tower. Input: user features -> output: user embedding."""

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.user_emb = nn.Embedding(cfg.n_users, cfg.user_embedding_dim)
        layers = []
        in_dim = cfg.user_embedding_dim + 8  # + contextual features
        for h in cfg.tower_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.output_dim)

    def forward(self, user_ids: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        emb = self.user_emb(user_ids)
        x = torch.cat([emb, context_features], dim=-1)
        return self.norm(self.mlp(x))


class ItemTower(nn.Module):
    """Item representation tower. Input: item features -> output: item embedding."""

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.item_emb = nn.Embedding(cfg.n_items, cfg.item_embedding_dim)
        layers = []
        in_dim = cfg.item_embedding_dim + 12  # + item metadata features
        for h in cfg.tower_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.output_dim)

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(item_ids)
        x = torch.cat([emb, item_features], dim=-1)
        return self.norm(self.mlp(x))


class TwoTowerModel(nn.Module):
    """
    Full two-tower retrieval model.
    User tower + item tower trained with in-batch negatives (InfoNCE loss).
    """

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.user_tower = UserTower(cfg)
        self.item_tower = ItemTower(cfg)

    def forward(
        self,
        user_ids: torch.Tensor,
        user_ctx: torch.Tensor,
        item_ids: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(user_ids, user_ctx)
        item_emb = self.item_tower(item_ids, item_feats)
        return user_emb, item_emb

    def infonce_loss(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        """In-batch negatives InfoNCE loss."""
        # user_emb: (B, D), item_emb: (B, D)
        # Positive: user[i] with item[i], negatives: all other items in batch
        logits = torch.matmul(user_emb, item_emb.T) / self.cfg.temperature  # (B, B)
        labels = torch.arange(len(user_emb), device=user_emb.device)
        return F.cross_entropy(logits, labels)

    def get_user_embedding(self, user_id: int, context: Optional[np.ndarray] = None) -> np.ndarray:
        """Get embedding for a single user."""
        self.eval()
        with torch.no_grad():
            uid = torch.tensor([user_id], device=next(self.parameters()).device)
            ctx = torch.tensor(
                context if context is not None else np.zeros(8),
                dtype=torch.float32,
            ).unsqueeze(0).to(uid.device)
            emb = self.user_tower(uid, ctx)
        return emb.cpu().numpy()[0]

    def get_item_embeddings_batch(
        self, item_ids: List[int], item_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get embeddings for a batch of items."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            iids = torch.tensor(item_ids, device=device)
            feats = torch.tensor(
                item_features if item_features is not None else np.zeros((len(item_ids), 12)),
                dtype=torch.float32,
            ).to(device)
            emb = self.item_tower(iids, feats)
        return emb.cpu().numpy()


class InteractionDataset(Dataset):
    def __init__(self, df: pl.DataFrame, cfg: TwoTowerConfig):
        self.user_ids = df["user_id_encoded"].to_numpy() if "user_id_encoded" in df.columns else np.zeros(len(df), dtype=np.int64)
        self.item_ids = df["item_id_encoded"].to_numpy() if "item_id_encoded" in df.columns else np.zeros(len(df), dtype=np.int64)
        self.labels = df["label"].to_numpy() if "label" in df.columns else np.ones(len(df))
        self.n = len(df)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx] % 100_000, dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx] % 500_000, dtype=torch.long),
            "user_ctx": torch.zeros(8, dtype=torch.float32),
            "item_feats": torch.zeros(12, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class TwoTowerTrainer:
    def __init__(self, cfg: TwoTowerConfig = None):
        self.cfg = cfg or TwoTowerConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        print(f"[TwoTower] {sum(p.numel() for p in self.model.parameters()):,} params | device={self.device}")

    def train(self, data_path: str, run_name: Optional[str] = None) -> Dict[str, Any]:
        run_name = run_name or f"two-tower-{int(time.time())}"
        df = pl.read_parquet(data_path) if data_path.endswith(".parquet") else pl.read_csv(data_path)
        dataset = InteractionDataset(df, self.cfg)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=2)

        wb_run = wandb.init(project=self.cfg.wandb_project, name=run_name, config=self.cfg.__dict__)
        mlflow.set_experiment(self.cfg.wandb_project)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.cfg.__dict__)
            history = []
            for epoch in range(self.cfg.epochs):
                self.model.train()
                epoch_loss = []
                for batch in loader:
                    user_emb, item_emb = self.model(
                        batch["user_id"].to(self.device),
                        batch["user_ctx"].to(self.device),
                        batch["item_id"].to(self.device),
                        batch["item_feats"].to(self.device),
                    )
                    loss = self.model.infonce_loss(user_emb, item_emb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss.append(loss.item())

                mean_loss = np.mean(epoch_loss)
                history.append(mean_loss)
                wandb.log({"train_loss": mean_loss, "epoch": epoch})
                mlflow.log_metric("train_loss", mean_loss, step=epoch)
                if (epoch + 1) % 2 == 0:
                    print(f"  Epoch {epoch+1}/{self.cfg.epochs} | loss={mean_loss:.4f}")

        wb_run.finish()
        return {"final_loss": history[-1], "history": history}

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "config": self.cfg.__dict__}, path)
        print(f"[TwoTower] Saved to {path}")
