from ipynb.fs.full.Introduction import *
from tqdm import tqdm
import time

# Normalization Techniques

Within the literature, the typical solution to oversmoothing is to apply some type of normalization. To this end, we propose orthogonalizing out vector $\vec{q}$ (essentially a "best guess" for the dominant eigenvector) from $X^{l}$ and then renormalizing each column. We incorporate additional scale parameters $0\leq{}s_{1}\leq{}1$ and $s_{2}$.

$$X^{l} = X^{l} - s_{1}\vec{q}\frac{\vec{q}^{T}X^{l}}{||\vec{q}||_{2}^{2}}$$
$$X_{:,i}^{l} = s_{2}\frac{X_{:,i}^{l}}{||X_{:,i}^{l}||_{2}}$$

If $s_{1}=1$ and $\vec{q}=\vec{1}$, this is equivalent to PairNorm [16].

$$X_{:,i}^{l} = X_{i}^{l} - E[X_{:,i}^{l}]$$
$$X_{:,i}^{l} = s \frac{X_{:,i}^{l}}{||X_{:,i}^{l}||_{2}}$$

We evaluate our normalization scheme with $\vec{q} = \frac{1}{\vec{d}_{degree}}$ and compare against both PairNorm and GraphSizeNorm [3].

## Dataset

num_graphs = 3000
d = []
for _ in range(num_graphs):
    # Set Cluster sizes and edge probabilities
    n = torch.randint(50,100,(5,))
    p = 1/(50*n) + (49/(50*n)) * torch.rand((5,5))
    p = .5 * (p + p.T)
    
    # Generate SBM
    x,edges = torch.ones((n.sum(),1)),torch_geometric.utils.remove_isolated_nodes(torch_geometric.utils.stochastic_blockmodel_graph(n,p))[0]
    adj = torch_sparse.SparseTensor(row=edges[0],col=edges[1])

    # Write to Data object
    d.append(torch_geometric.data.Data(x=x[:adj.size(0)],edge_index = edges))

for idx,G in enumerate(d):
    G.edge_weight = torch.ones(G.edge_index[0].shape)
    adj = torch_sparse.SparseTensor(row=G.edge_index[0],col=G.edge_index[1],value=G.edge_weight)
    
    # Compute Katz Centrality
    v = 1/(1.01*torch.norm(torch.eig(adj.to_dense())[0],dim=1).max())
    y = torch.sum(torch.inverse(torch.eye(adj.size(0)) - v*adj.to_dense().T) - torch.eye(adj.size(0)),dim=1)
    
    # Set as target
    G.y = y
    d[idx] = G
    
train,test = d[:2000],d[2000::]
train_loader = torch_geometric.data.DataLoader(train,batch_size=200,shuffle=True)
test_loader = torch_geometric.data.DataLoader(test,batch_size=200,shuffle=True)

## Model

# Compute the Mean Average Distance of inputs
def batched_MAD(X,edge_index,edge_weights):
    X = X/torch.norm(X,dim=1)[:,None]
    cosine = 1 - torch.sum(X[edge_index[0]] * X[edge_index[1]],dim=1)
    return 1/edge_weights.sum() * (edge_weights * cosine).sum()

# Compute the Aggregation Norm
def batched_agg(X,edge_index,edge_weights,batch):
    nX = torch_scatter.scatter_sum(edge_weights[:,None] * X[edge_index[1]], edge_index[0],dim=0)
    X,nX = X/torch_scatter.scatter_sum(X**2,batch,dim=0).sqrt()[batch],\
              nX/torch_scatter.scatter_sum(nX**2,batch,dim=0).sqrt()[batch]
    return torch.norm(X - nX,dim=1).mean()

# Class for the OrthNorm
class OrthNormL2(torch.nn.Module):
    def __init__(self):
        super(OrthNormL2,self).__init__()
        
        # Scale parameters
        self.s1 = torch.nn.Parameter(torch.ones(1))
        self.s2 = torch.nn.Parameter(torch.ones(1))

    def forward(self,X,edge_index,edge_weight,batch):
        
        # Compute q
        q = 1/torch_geometric.utils.degree(edge_index[0])
        q_norm = torch_scatter.scatter_sum(q**2,batch,dim=0)
        
        # Compute scalar projection of X onto q
        alpha = torch_scatter.scatter_sum(X * q[:,None],batch,dim=0)/(q_norm)[:,None]
        
        # Compute OrthNorm
        X = X - torch.sigmoid(self.s1) * (alpha)[batch] * q[batch][:,None]
        X = self.s2 * X/(torch_scatter.scatter_sum(X**2,batch,dim=0).sqrt())[batch]
        return X

# This is a GraphConv model which we can plug various normalization schemes into.
class DummyModel(torch.nn.Module):
    def __init__(self,in_channels,int_channels,out_channels,depth,norm,p=2):
        super(DummyModel,self).__init__()
        self.start = torch.nn.Linear(in_channels,int_channels)
        self.intermediate = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Linear(int_channels,int_channels),\
                                                                      torch.nn.Linear(int_channels,int_channels)])\
                                                 for _ in range(depth)])
        self.norm = torch.nn.ModuleList([norm() for _ in range(depth)])
        self.finish = torch.nn.Linear(int_channels,out_channels)

    def forward(self,X,edge_index,edge_weight,batch):
        X = self.start(X)
        for idx,m in enumerate(self.intermediate):
            X = X + m[0](X) + torch_scatter.scatter_sum(edge_weight[:,None] * m[1](X)[edge_index[1]], edge_index[0],dim=0)
            X = torch.nn.LeakyReLU()(self.norm[idx](X,edge_index,edge_weight,batch))
            if torch.isnan(X).any(): raise ValueError
        return self.finish(X)

## OrthNorm

graph_results = []
model_mad = []
model_agg = []

for k in [4,8,16,32,64]:
    torch.manual_seed(0)
    graph = DummyModel(1,32,1,k,OrthNormL2).cuda()
    
    graph_results.append(train_loop(graph,train_loader,test_loader,150,lr=1e-1))
    torch.cuda.empty_cache()

    # Iterate through network layers and compute the MAD/Agg at each
    MAD,Agg = torch.zeros(k),torch.zeros(k)
    for idx,data in enumerate(test_loader):
        X = data.x.cuda()
        row,col = data.edge_index[0].cuda(),data.edge_index[1].cuda()
        batch = data.batch.cuda()

        graph.eval()
        X = graph.start(X)
        
        for jdx,m in enumerate(graph.intermediate):
            X = X + m[0](X) + torch_scatter.scatter_sum(m[1](X)[col], row.cuda(),dim=0)
            X = torch.nn.LeakyReLU()(graph.norm[jdx](X,data.edge_index.cuda(),data.edge_weight.cuda(),batch))
            MAD[jdx] += batched_MAD(X,data.edge_index.cuda(),data.edge_weight.cuda()).mean().item()
            Agg[jdx] += batched_agg(X,data.edge_index.cuda(),data.edge_weight.cuda(),batch).item()
        
    model_mad.append(MAD/(idx+1))
    model_agg.append(Agg/(idx+1))

### Results

plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
for i in range(5):
  plt.semilogy(graph_results[i][0])

plt.title('Train Error')
plt.ylabel('L1 Error')
plt.xlabel('Iterations')

plt.subplot(1,3,2)
for i in range(5):
  plt.semilogy(graph_results[i][1])

plt.title('Test Error')
plt.xlabel('Iterations')

plt.subplot(1,3,3)
for idx,alpha in enumerate([4,8,16,32,64]):
  plt.semilogy(graph_results[idx][2],label=alpha)
plt.title('Ranking Error')
plt.ylabel('Avg. Displacement')
plt.xlabel('Iterations')
plt.legend()

plt.tight_layout();

for idx,d in enumerate(model_mad):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.plot(d,label='MAD')
  plt.xlabel('Layer')
  plt.ylabel('Mean Average Distance')

  plt.subplot(1,2,2)
  plt.plot(model_agg[idx])
  plt.xlabel('Layer')
  plt.ylabel('Aggregation Norm')
  plt.show()

## GraphSizeNorm

# Class for GraphSizeNorm. This is essentially dividing each feature vector by |V|
class GraphSizeNorm(torch.nn.Module):
    def __init__(self):
        super(GraphSizeNorm,self).__init__()
        self.norm = torch_geometric.nn.GraphSizeNorm()
    def forward(self,X,edge_index,edge_weight,batch):
        return self.norm(X,batch)

graph_results = []
model_mad = []
model_agg = []

for k in [4,8,16,32,64]:
    torch.manual_seed(0)
    graph = DummyModel(1,32,1,k,GraphSizeNorm).cuda()
    
    graph_results.append(train_loop(graph,train_loader,test_loader,150,lr=1e-1))
    torch.cuda.empty_cache()

    # Iterate through network layers and compute the MAD/Agg at each
    MAD,Agg = torch.zeros(k),torch.zeros(k)
    for idx,data in enumerate(test_loader):
        X = data.x.cuda()
        row,col = data.edge_index[0].cuda(),data.edge_index[1].cuda()
        batch = data.batch.cuda()

        graph.eval()
        X = graph.start(X)
        
        for jdx,m in enumerate(graph.intermediate):
            X = X + m[0](X) + torch_scatter.scatter_sum(m[1](X)[col], row.cuda(),dim=0)
            X = torch.nn.LeakyReLU()(graph.norm[jdx](X,data.edge_index.cuda(),data.edge_weight.cuda(),batch))
            MAD[jdx] += batched_MAD(X,data.edge_index.cuda(),data.edge_weight.cuda()).mean().item()
            Agg[jdx] += batched_agg(X,data.edge_index.cuda(),data.edge_weight.cuda(),batch).item()
        
    model_mad.append(MAD/(idx+1))
    model_agg.append(Agg/(idx+1))

### Results

plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
for i in range(5):
  plt.semilogy(graph_results[i][0])

plt.title('Train Error')
plt.ylabel('L1 Error')
plt.xlabel('Iterations')

plt.subplot(1,3,2)
for i in range(5):
  plt.semilogy(graph_results[i][1])

plt.title('Test Error')
plt.xlabel('Iterations')

plt.subplot(1,3,3)
for idx,alpha in enumerate([4,8,16,32,64]):
  plt.semilogy(graph_results[idx][2],label=alpha)
plt.title('Ranking Error')
plt.ylabel('Avg. Displacement')
plt.xlabel('Iterations')
plt.legend()

plt.tight_layout();

for idx,d in enumerate(model_mad):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.plot(d,label='MAD')
  plt.xlabel('Layer')
  plt.ylabel('Mean Average Distance')

  plt.subplot(1,2,2)
  plt.plot(model_agg[idx])
  plt.xlabel('Layer')
  plt.ylabel('Aggregation Norm')
  plt.show()

## PairNorm

# PairNorm implementation
class PairNorm(torch.nn.Module):
    def __init__(self):
        super(PairNorm,self).__init__()
        self.norm = torch_geometric.nn.PairNorm(scale_individually=True)
    def forward(self,X,edge_index,edge_weight,batch):
        return self.norm(X,batch)

graph_results = []
model_mad = []
model_agg = []

for k in [4,8,16,32,64]:
    torch.manual_seed(0)
    graph = DummyModel(1,32,1,k,PairNorm).cuda()
    
    graph_results.append(train_loop(graph,train_loader,test_loader,150,lr=1e-1))
    torch.cuda.empty_cache()

    # Iterate through network layers and compute the MAD/Agg at each
    MAD,Agg = torch.zeros(k),torch.zeros(k)
    for idx,data in enumerate(test_loader):
        X = data.x.cuda()
        row,col = data.edge_index[0].cuda(),data.edge_index[1].cuda()
        batch = data.batch.cuda()

        graph.eval()
        X = graph.start(X)
        
        for jdx,m in enumerate(graph.intermediate):
            X = X + m[0](X) + torch_scatter.scatter_sum(m[1](X)[col], row.cuda(),dim=0)
            X = torch.nn.LeakyReLU()(graph.norm[jdx](X,data.edge_index.cuda(),data.edge_weight.cuda(),batch))
            MAD[jdx] += batched_MAD(X,data.edge_index.cuda(),data.edge_weight.cuda()).mean().item()
            Agg[jdx] += batched_agg(X,data.edge_index.cuda(),data.edge_weight.cuda(),batch).item()
        
    model_mad.append(MAD/(idx+1))
    model_agg.append(Agg/(idx+1))

### Results

plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
for i in range(5):
  plt.semilogy(graph_results[i][0])

plt.title('Train Error')
plt.ylabel('L1 Error')
plt.xlabel('Iterations')

plt.subplot(1,3,2)
for i in range(5):
  plt.semilogy(graph_results[i][1])

plt.title('Test Error')
plt.xlabel('Iterations')

plt.subplot(1,3,3)
for idx,alpha in enumerate([4,8,16,32,64]):
  plt.semilogy(graph_results[idx][2],label=alpha)
plt.title('Ranking Error')
plt.ylabel('Avg. Displacement')
plt.xlabel('Iterations')
plt.legend()

plt.tight_layout();

for idx,d in enumerate(model_mad):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.plot(d,label='MAD')
  plt.xlabel('Layer')
  plt.ylabel('Mean Average Distance')

  plt.subplot(1,2,2)
  plt.plot(model_agg[idx])
  plt.xlabel('Layer')
  plt.ylabel('Aggregation Norm')
  plt.show()

## Overview

Our new normalization scheme, which we are calling OrthNorm (i.e Orthagonal Normalization, get it?), improves upon the unnormalized case, even without having fully converged. It surpasses PairNorm by around an order of magnitude given the same depth and number of epochs. We suspect this is because $d_{Katz}$ is close enough to $v_{1}$, and so PairNorm reduces the smoothness *too* much. Indeed, PairNorm returns large MAP and AggNorm values for each layer, whereas those of OrthNorm are generally smaller and more variable. This is due to $s_{1}$ regulating the magnitude of the orthogonalization; without it, MAP consistently hovers around $.20$ and StepAgg continually decreases. 

As for GraphSizeNorm, there is certainly some benefit, but it does still converge to a larger value than OrthNorm (which isn't even convergent yet). Oddly enough, while the MAP is otherwise quite low, GraphSizeNorm spikes in the last few layers for both the $l=32$ and $l=64$ models. We do not have an explanation for why this occurs.