import os.path as osp
import random
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn  import global_add_pool
from Clustering import Clustering
import numpy as np
torch.manual_seed(0)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS')
dataset = dataset.shuffle()
n = len(dataset) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=60)
train_loader = DataLoader(train_dataset, batch_size=60)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GraphConv(dataset.num_features, 128)
        self.convk = GraphConv(500, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)
        self.k = 500
        self.Uw = nn.Parameter(torch.zeros(size=(int(self.k), int(128))))
    # nn.init.xavier_uniform_(self.Uw.data, gain=1.414)
    #     nn.init.xavier_uniform_(self.Uw.data, gain=1.414)

    def forward(self, data,epoch,j):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print("model.parameters()",model.parameters())


        x = F.relu(self.conv1(x, edge_index))
        # if  epoch==1 and j==0:
        #     kmeans = KMeans(n_clusters=500, n_init=20)
        #     kmeans.fit_predict(x.cpu().detach().numpy())
        #     self.Uw.data = torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)
        # print("x",x.shape)
        kl_loss, q = Clustering(x, self.Uw.cuda())
        # print("q",  q .shape)
        q= F.relu(self.convk(q, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(q, edge_index, None, batch)
        # print("x",x.shape   )
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        # x = x2
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.dropout(x, p=0.8, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x,kl_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
lr=0.0005
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

optimizer = torch.optim.Adam([
    # {'params': model.parameters(), 'lr': 0.01},
{'params':model.Uw, 'lr': 0},
{'params':model.conv1.lin_rel.weight, 'lr': lr},
{'params': model.conv1.lin_rel.bias, 'lr': lr},
{'params': model.conv1.lin_root.weight, 'lr': lr},
{'params': model.convk.lin_rel.weight, 'lr': lr},
{'params': model.convk.lin_rel.bias, 'lr': lr},
{'params': model.convk.lin_root.weight, 'lr': lr},
{'params': model.pool1.weight, 'lr': lr},
{'params': model.conv2.lin_rel.weight, 'lr':  lr},
{'params': model.conv2.lin_rel.bias, 'lr':  lr},
{'params': model.conv2.lin_root.weight , 'lr': lr},
{'params': model.pool2.weight, 'lr': lr},
{'params': model.conv3.lin_rel.weight, 'lr':  lr},
{'params': model.conv3.lin_rel.bias, 'lr':  lr},
{'params': model.conv3.lin_root.weight , 'lr': lr},
{'params': model.pool3.weight, 'lr': lr},
{'params': model.lin1.weight, 'lr': lr},
{'params': model.lin1.bias, 'lr': lr},
{'params': model.lin2.weight, 'lr': lr},
{'params': model.lin2.bias, 'lr': lr},
{'params': model.lin3.weight, 'lr': lr},
{'params': model.lin3.bias, 'lr': lr},
#block0



                        ], )


def train(epoch):
    model.train()

    loss_all = 0
    j=0
    for data in train_loader:
        # for name, param in model.named_parameters():
        #     print(name, '-->', param.shape)

        data = data.to(device)
        optimizer.zero_grad()
        output,klloss = model(data,epoch,j)
        loss = F.nll_loss(output, data.y)
        loss = loss+klloss
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        j=j+1
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    j=0
    for data in loader:
        data = data.to(device)
        pred = model(data,epoch,j)[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        j = j + 1
    return correct / len(loader.dataset)



# random.seed(0)
# np.random.seed(0)
torch.manual_seed(0)
for epoch in range(1, 201):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
          f'Test Acc: {test_acc:.5f}')
    file_handle1 = open('0716_slim_Protein_withloss-500-exp-w-U0.0005-seed0-zero2.txt', mode='a')
    print(str(epoch), file=file_handle1)
    print(str(test_acc), file=file_handle1)
    print(str(train_acc ), file=file_handle1)