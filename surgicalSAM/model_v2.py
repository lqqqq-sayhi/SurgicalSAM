import torch 
import torch.nn as nn 
from einops import rearrange


class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_tokens=8):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        # 冻结除最后一层外的权重
        for param in [self.dense_fc_1.parameters(), self.sparse_fc_1.parameters()]:
            for p in param:
                p.requires_grad = False
                
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)]
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
        print(f"self.dense_fc_1: {self.dense_fc_1}")
        print(f"self.dense_fc_2: {self.dense_fc_2}")
        print(f"self.sparse_fc_1: {self.sparse_fc_1}")
        print(f"self.sparse_fc_2: {self.sparse_fc_2}")
                
    def forward(self, feat, prototypes, cls_ids):
        # print("feat shape:", feat.shape, "prototypes shape:", prototypes.shape)
        # feat shape: torch.Size([2, 4096, 256]) prototypes shape: torch.Size([28, 256])

        B, hw, c = feat.shape  # B=2, hw=4096, c=256
        num_cls = prototypes.shape[0]
        cls_prompts = prototypes.unsqueeze(0).unsqueeze(-1).expand(B, num_cls, c, 1)  # [B, num_cls, 256, 1]         
        # cls_prompts = prototypes.unsqueeze(-1) 
        # cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)
        # print("feat shape:", feat.shape, "cls_prompts shape:", cls_prompts.shape)
        # feat shape: torch.Size([2, 28, 4096, 256]) cls_prompts shape: torch.Size([2, 28, 256, 1])
        sim = torch.matmul(feat, cls_prompts)
        feat = feat + feat*sim
        feat_sparse = feat.clone()
        
        one_hot = torch.nn.functional.one_hot(cls_ids-1, 28)
        feat = feat[one_hot ==1]
        feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=28)
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings


class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=28, feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
        # 冻结除最后一层外的权重
        for param in self.class_embeddings.parameters():
            param.requires_grad = False
        
        # 仅训练最后一层的权重
        self.final_layer = nn.Linear(feat_dim, num_classes)
        for param in self.final_layer.parameters():
            param.requires_grad = True
            
    def forward(self):
        return self.class_embeddings.weight
