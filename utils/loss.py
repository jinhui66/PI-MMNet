import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_structure_loss(adj_pred, adj_true):
    # print(adj_pred, adj_true)
    return F.binary_cross_entropy(adj_pred, adj_true)

# def node_representation_loss(node_embeddings, labels):
#     loss_fn = nn.CrossEntropyLoss()
#     print(labels)
#     logits = torch.matmul(node_embeddings, node_embeddings.t())
#     labels = labels.unsqueeze(1).repeat(1, labels.size(0))
#     print(node_embeddings, logits, labels)
#     return loss_fn(logits, labels)

def classification_loss(logits, targets):
    return F.cross_entropy(logits, targets)

class Loss(nn.Module):
    def __init__(self, alpha=0.2, beta=1.0):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.gamma = gamma

    def forward(self, adj_pred, adj_true, node_embeddings, labels, logits):
        structure_loss = graph_structure_loss(adj_pred, adj_true)
        # representation_loss = node_representation_loss(node_embeddings, labels)
        classification_loss_val = classification_loss(logits, labels)
        
        total_loss = self.alpha * structure_loss  + self.beta * classification_loss_val
        return structure_loss, classification_loss_val
    

def compute_ground_truth_adjacency(labels):
    batch_size = labels.size(0)
    adj_true = torch.zeros((batch_size, batch_size), device=labels.device)
    for i in range(batch_size):
        for j in range(batch_size):
            adj_true[i, j] = 1 if labels[i] == labels[j] else 0
    return adj_true