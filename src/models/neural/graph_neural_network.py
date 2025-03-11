"""
Graf Sinir Agi (GNN) modulu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional

class GraphNeuralNetwork(nn.Module):
    """
    Graf Sinir Agi modeli.
    GCN ve Attention mekanizmalarini icerir.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        GNN modelini baslatir.
        
        Args:
            config: Model konfigurasyonu
        """
        super().__init__()
        self.config = config["graph_neural_network"]
        self.layers = self.config["model"]["layers"]
        
        # GCN katmanlari
        self.convs = nn.ModuleList()
        for i in range(len(self.layers) - 1):
            self.convs.append(
                GCNConv(self.layers[i], self.layers[i + 1])
            )
        
        # Attention mekanizmasi
        self.attention = MultiHeadAttention(
            self.layers[-1],
            self.config["attention"]["num_heads"],
            self.config["attention"]["head_dim"],
            self.config["attention"]["dropout"]
        )
        
        self.dropout = nn.Dropout(self.config["model"]["dropout"])
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Ileri gecis islemi.
        
        Args:
            data: Graf verisi
            
        Returns:
            Islenmis tensor
        """
        x, edge_index = data.x, data.edge_index
        
        # GCN katmanlarindan gecir
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Son GCN katmani
        x = self.convs[-1](x, edge_index)
        
        # Attention uygula
        x = self.attention(x)
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Cok basli attention mekanizmasi.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float):
        """
        Attention modulunu baslatir.
        
        Args:
            input_dim: Giris boyutu
            num_heads: Attention basi sayisi
            head_dim: Her basin boyutu
            dropout: Dropout orani
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.q_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.k_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.v_linear = nn.Linear(input_dim, num_heads * head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ileri gecis islemi.
        
        Args:
            x: Giris tensoru
            
        Returns:
            Attention uygulanmis tensor
        """
        batch_size = x.size(0)
        
        # Linear projeksiyonlar
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention skorlari
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        
        # Attention dagilimi
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Attention ciktisi
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Son linear katman
        out = self.output_linear(out)
        
        return out

class GraphBuilder:
    """
    Metin verilerinden graf yapisi olusturur.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Graf olusturucu sinifini baslatir.
        
        Args:
            config: Konfigurasyon ayarlari
        """
        self.config = config
    
    def build_graph(self, 
                   nodes: List[str],
                   edges: List[tuple],
                   node_features: Optional[torch.Tensor] = None) -> Data:
        """
        Verilen dugum ve kenarlardan graf olusturur.
        
        Args:
            nodes: Dugum listesi
            edges: Kenar listesi (kaynak, hedef) seklinde
            node_features: Dugum ozellikleri (opsiyonel)
            
        Returns:
            PyTorch Geometric Data objesi
        """
        # Dugumleri indeksle
        node_indices = {node: idx for idx, node in enumerate(nodes)}
        
        # Kenarlari tensor formatina cevir
        edge_index = torch.tensor(
            [[node_indices[src], node_indices[dst]] for src, dst in edges],
            dtype=torch.long
        ).t()
        
        # Dugum ozelliklerini kontrol et
        if node_features is None:
            node_features = torch.eye(len(nodes))
        
        # Graf olustur
        return Data(x=node_features, edge_index=edge_index) 