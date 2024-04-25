import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
import os


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.checkpoint_dir = None

    def save(self, path, model_name, epoch):
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), path/f"{model_name}-{epoch}.pth")
        
    #def load(self, path):
    #    self.load_state_dict(torch.load(path))

    def load(self, checkpoint_dir):
        if not checkpoint_dir: return
        self.checkpoint_dir = checkpoint_dir
        
        # List all files in the checkpoint directory
        files = os.listdir(checkpoint_dir)
        
        # Filter files that match the model's naming pattern
        model_files = [f for f in files if f.startswith(f'{self.model_name}Weights+Optimizer')]
        
        # Extract epochs from the filtered filenames
        epochs = [int(f.split('Weights+Optimizer')[1]) for f in model_files]
        
        if not epochs:  # If no matching files were found
            print(f"No checkpoints found for {self.model_name} in {checkpoint_dir}")
            return
        
        # Find the file with the maximum epoch
        last_epoch = max(epochs)
        path = os.path.join(checkpoint_dir, f'{self.model_name}Weights+Optimizer{last_epoch}')
        
        # Load the state dicts for both the model and the optimizer
        model_state, optimizer_state = torch.load(path)
        self.load_state_dict(model_state)
        
        print(f"Loaded {self.model_name} from checkpoint '{path}'")

        return self
    


class SVCNN(Model):

    def __init__(self, nclasses, pretrained=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__()
        self.nclasses = nclasses
        self.pretrained = pretrained
        self.cnn_name = cnn_name
        self.model_name = "SVCNN"
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretrained)
                self.net.fc = nn.Linear(512, nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretrained)
                self.net.fc = nn.Linear(512, nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretrained)
                self.net.fc = nn.Linear(2048, nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretrained).features
                self.net_2 = models.alexnet(pretrained=self.pretrained).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretrained).features
                self.net_2 = models.vgg11(pretrained=self.pretrained).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretrained).features
                self.net_2 = models.vgg16(pretrained=self.pretrained).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096, nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model): #    mvcnn_model = MVCNN(nclasses=num_classes, pretrained=True, cnn_name='vgg11')  # Adjust parameters


    def __init__(self, model, nclasses, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__()
        self.model_name = "MVCNN"
        self.nclasses = nclasses
        self.num_views = num_views

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views), self.num_views, *y.shape[-3:]))
        y = torch.max(y, 1)[0]
        y = y.view(y.shape[0], -1)
        y = self.net_2(y)
        return y


class ArcGNN(Model):
    def __init__(self, num_node_features, num_classes):
        super(ArcGNN, self).__init__()
        self.model_name = "ArcGNN"
        self.conv1 = GCNConv(num_node_features, 128) #TODO
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global mean pooling
        x = global_mean_pool(x, batch) 
        return x


class HybridModel(Model):
    def __init__(self, arcgnn_model, mvcnn_model, num_classes):
        super(HybridModel, self).__init__()
        self.model_name = "HybridModel"
        self.arcgnn_model = arcgnn_model
        self.mvcnn_model = mvcnn_model
        # Example: Assuming both models' outputs are of size 64 before classification
        self.classifier = nn.Linear(num_classes * 2, num_classes)
        
    def forward(self, x, edge_index, batch, mvcnn_data):
        gnn_output = self.arcgnn_model(x, edge_index, batch)                
        mvcnn_output = self.mvcnn_model(mvcnn_data)
        combined_features = torch.cat((gnn_output, mvcnn_output), dim=1)
        classification = self.classifier(combined_features)
        return classification
    


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


