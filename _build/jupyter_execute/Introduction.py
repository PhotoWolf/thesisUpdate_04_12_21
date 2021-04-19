## Introduction

Graph Neural Networks (GCNs) are an extension of the familiar Convolutional Neural Network to arbitrary topologies. Given the graph $G=\{V,E\}$, Gilmer et al. (2017) define the message-passing framework of GCNs as

$$x_{i}^{l+1} = \Theta^{l}(v_{i}^{l},\gamma(x_{i}^{l},\{x_{j}^{l}:j\in{}N^{1}_{i}\},e_{ij}))$$

where $N_{i}^{1}$ is the 1-neightborhood of vertex $v_{i}$, and $l$ indexes the $l$-th layer of the model. $x_{i}\in{}R^{n}$ is the feature vector associated with $v_{i}$. We refer to $\gamma{}$ as our aggregation function; typical choices are the sum or max operators, but some more exotic options do exist(ie: LSTMs). $\Theta$ is a neural network of some description, most often a single linear layer [1].

While GCNs are fairly well-studied, we have a limited understanding of how well they capture the topological information of $G$. The literature finds clear benefit to incorporating structural features into GCNs; in particular:

* Pretraining a GCN on various centrality tasks improves the accuracy of downstream classifiers [2]. 
* Appending Laplacian eigenvectors to vertex features surpasses existing benchmark peroformance, in some cases quite significantly [3].
* Retaining centrality information produces more discriminative node embeddings [4]. 

This motivates our desire to assess the ability of GCNs to learn graph structure and develop means by which it may be improved. Thus far, [5] is the only work in a similar vein; however, their experimental setting is unclear to the point of being unreproducible. They also include a fixed node ID parameter, which makes it difficult to generalize results outside of a transductive setting. [6] demonstrates the applicability of ML to learning topological graph metrics, but it predates the advent of GCNs.

## Model Code

It is convient to classify GCNs into two groups: node-wise convolutions and edge-wise convolutions. The former process all edges equivalently i.e: $\gamma$ is independent of $x_{i}$ and $x_{j}$ for $e_{ij}$. This allows us to generalize to varied topologies without much in the way of computational cost. The prototypical example of a node-wise GCN is the GraphConv architecture [7]:

$$x_{i}^{l+1} = \Theta_{1}^{l}(x_{i}^{l+1}) + \Theta_{2}^{l}(\sum_{j\in{}N_{i}^{1}}w_{ij}x_{j}^{l})$$

for $x_{i}^{l}\in{}R^{n}$ and $x_{i}^{l}\in{}R^{k}$. $w_{ij}$ is the scalar weight assocaited with $e_{ij}$. The model employs two feedforward networks, $\Theta_{1}$ and $\Theta_{2}$, which can project the features of the target node and those aggregated from $N_{i}^{l}$ into different subspaces. Assuming both $\Theta$ are  $R^{kxn}$ matrices, each GraphConv layer is $O(|V|kn + |E|n)$ in time and $O(|V|n + |E|)$ in space.

In many cases, it is actually beneficial to operate on *pairs* of node features, and for that we require edge-wise convolutions, of which the most prominent are the Graph Attention Network (GAT) and it's numerous derivatives. We do not find GATs to be particularily performant (or efficient, for that matter), so we instead choose to focus on EdgeConv [8,9]:

$$x_{i}^{l} = \sum_{j\in{}N_{i}^{1}}w_{ij}\Theta{}^{l}(x_{i}^{l}||x_{j}^{l}-x_{i}^{l})$$

If $\Theta$ is restricted to a $R^{k\times{}2n}$ matrix, EdgeConv possesses a layerwise time complexity of $\textit{O}(|E|kn)$ and a $\textit{O}(|V|n + |E|n)$ space complexity.

Our implementations follow each layer with a 1D BatchNorm and LeakyReLU activation [10,11].

import torch
import torch_geometric
import torch_sparse
import torch_scatter
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# GraphConv Model
class GraphConv(torch.nn.Module):
    # in_channels and out_channels are self-explanatory. int_channels is the number of 
    # features in the intermediate layers. Depth controls the number of aggregations.
    def __init__(self,in_channels,int_channels,out_channels,depth):
        super(GraphConv,self).__init__()
        self.start = torch.nn.Linear(in_channels,int_channels)
        self.intermediate = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Linear(int_channels,int_channels),\
                                                                      torch.nn.Linear(int_channels,int_channels)])\
                                                 for _ in range(depth)])
        self.bn = torch.nn.ModuleList([torch_geometric.nn.BatchNorm(int_channels) for _ in range(depth)])
        self.finish = torch.nn.Linear(int_channels,out_channels)
        
    def forward(self,X,edge_index,edge_weight,batch):
        # Project to int_channels
        X = self.start(X)
        
        # Run through GraphConv layers
        for idx,m in enumerate(self.intermediate):
            X = m[0](X) + torch_scatter.scatter_sum(edge_weight[:,None] * m[1](X)[edge_index[1]], edge_index[0],dim=0)
            X = torch.nn.LeakyReLU()(self.bn[idx](X))
            
        # Project to out_channels
        return self.finish(X)

# EdgeConv Model
class EdgeConv(torch.nn.Module):
    
    # in_channels and out_channels are self-explanatory. int_channels is the number of 
    # features in the intermediate layers. Depth controls the number of aggregations.
    def __init__(self,in_channels,int_channels,out_channels,depth):
        super(EdgeConv,self).__init__()
        self.start = torch.nn.Linear(in_channels,int_channels)
        self.intermediate = torch.nn.ModuleList([torch.nn.Linear(2*int_channels,int_channels) for _ in range(depth)])
        self.bn = torch.nn.ModuleList([torch_geometric.nn.BatchNorm(int_channels) for _ in range(depth)])
        self.finish = torch.nn.Linear(int_channels,out_channels)
        
    def forward(self,X,edge_index,edge_weight,batch):
        # Project to int_channels
        X = self.start(X)
        
        # Run through EdgeConv layers
        for idx,m in enumerate(self.intermediate):
            Z = torch.cat((X[edge_index[0]],X[edge_index[0]] - X[edge_index[1]]),dim=1)
            X = torch_scatter.scatter_sum(edge_weight[:,None] * m(Z), edge_index[0],dim=0)
            X = torch.nn.LeakyReLU()(self.bn[idx](X))
            
        # Project to out_channels
        return self.finish(X)

## Training Code

Graph centrality measures are used to quantify the structural properties of a network. By training GCNs to predict more and more complex centralities, we hope to gain insight into how well they incorporate topology and what limitations they posses, if any. Depending on our algorithm's performance, there may also be various practical applications. For example, path-based centralities (betweenness, closeness, etc) are broadly $\textit{O}(|V|^{3})$ and, at best, $\textit{O}(|V||E|)$ [12], so an accurate GCN approximation may be of interest in analyzing larger networks. 

Following the example of [5], we define our loss as the L1 Norm between the normalized model output, $\vec{x}'$, and the targeted centrality scores, $\vec{y}$. Both quantities are min-max scaled.  
$$\vec{x}' = \frac{\vec{x} - min(\vec{x})}{max(\vec{x}) - min(\vec{x})}$$

$$L = ||\vec{x}' - \vec{y}||_{1}$$

We also define the rank displacement. Let $\vec{u}$ and $\vec{s}$ be vectors in $R^{k}$. $f$ is the *argsort* function i.e: the mapping $f(x_{i},\vec{x})\rightarrow{}r$ where $r=|\{x_{j}:x_{j} > x_{i} \forall{} x_{j}\in{}\vec{x}\}|$. $n$ is an arbitary constant. Then the rank displacement is given as follows:

$$r_{disp}(\vec{u},\vec{s})=\frac{1}{(1+f(s_{i},\vec{s}))^{n}k(k-1)}\sum_{i=0}^{k-1}|f(u_{i},\vec{u}) - f(s_{i},\vec{s})|$$

Centrality is often used to compare individual nodes and ascertain some manner of "relevance". To reflect this, we wanted to included a ranking measure as an added metric.


# L1 Norm w/ Min-Max normalization
def scaled_L1(X,Y,batch):
    X = normalize(X,batch)
    Y = normalize(Y,batch)
    return torch.nn.L1Loss()(X,Y)

# Computes Min-Max norm
def normalize(X,batch):
    Min = -torch_scatter.scatter_max(-X,batch,dim=0)[0][batch]
    Max = torch_scatter.scatter_max(X,batch,dim=0)[0][batch]
    return (X-Min)/(1e-12 + Max - Min)

# Gets rank (descending) of each element in X
def get_rank(X):
    val,inv_val = X.unique(return_inverse=True)
    return torch.argsort(torch.argsort(val,descending=True))[inv_val]

# Compute rank displacement
def rank_disp(X,Y,batch):
    L = 0
    for idx,b in enumerate(batch.unique()):
        X_rank,Y_rank = get_rank(X[batch==b]),get_rank(Y[batch==b])
        l = (X_rank.float() - Y_rank.float()).abs()/(1+Y_rank.float())**(.6)
        L += l.mean()/torch.numel(X_rank - 1)
    return L/(idx+1)

# Takes GCN model and data loaders.
def train_loop(model,train_loader,test_loader,epochs,lr=1e-3):
    train_loss = []
    test_loss = []
    rank = []

    try: opt = torch.optim.Adam(model.parameters(),lr=lr)
    except: pass;
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # Compute initial test loss and rank displacement
    model.eval()
    with torch.no_grad():
      ts,r = 0,0
      for idx,data in enumerate(test_loader):
          X,Y,edge_index,edge_weight = data.x.cuda(),data.y.cuda(),data.edge_index.cuda(),data.edge_weight.cuda()
          batch = data.batch.cuda()

          preds = model(X,edge_index,edge_weight,batch)

          loss = scaled_L1(preds.squeeze(),Y,batch)
          ts += loss.item()
          r += rank_disp(preds.squeeze(),Y,batch).item()

      rank.append(r/(idx+1))
      test_loss.append(ts/(idx+1))

    # Iterate over epochs
    for epoch in range(epochs):
        tr,ts,r = 0,0,0
        
        # Compute train error and backprop.
        model.train()
        for idx,data in enumerate(train_loader):
            X,Y,edge_index,edge_weight = data.x.cuda(),data.y.cuda(),data.edge_index.cuda(),data.edge_weight.cuda()
            batch = data.batch.cuda()

            preds = model(X,edge_index,edge_weight,batch)
            loss = scaled_L1(preds.squeeze(),Y,batch)

            try: 
              loss.backward()
              opt.step()
              opt.zero_grad()
            except: pass;

            tr += loss.item()
        train_loss.append(tr/(idx+1))

        # Compute test error and rank displacement
        model.eval()
        with torch.no_grad():
            for idx,data in enumerate(test_loader):
                X,Y,edge_index,edge_weight = data.x.cuda(),data.y.cuda(),data.edge_index.cuda(),data.edge_weight.cuda()
                batch = data.batch.cuda()

                preds = model(X,edge_index,edge_weight,batch)

                loss = scaled_L1(preds.squeeze(),Y,batch)

                ts += loss.item()
                r += rank_disp(preds.squeeze(),Y,batch).item()

        rank.append(r/(idx+1))
        test_loss.append(ts/(idx+1))
        
    # Return average values per epoch
    return train_loss,test_loss,rank


```{toctree}
:hidden:
:titlesonly:
:numbered: True

DegreeCentrality
KatzCentrality
DeeperGCNs
Normalization
HigherMatrixPowers
FutureWorkandBibliography
```
