import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from src.models.Trainer import Trainer
from src.data.dataset import GNNArchitecturalDataset, MultiviewImgDataset, HybridDataset
from src.models.models import ArcGNN, SVCNN, MVCNN, HybridModel
from src.models.mvcnn import _get_mvcnn_datasets, train_mvcnn
from src.models.arcgnn import train_arc_gnn
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

def hybrid_collate_fn(batch):
    gnn_data_list = []
    mvcnn_data_list = []
    mvcnn_labels = []

    for gnn_data, mvcnn_data in batch:
        if gnn_data is not None and mvcnn_data is not None:
            gnn_data_list.append(gnn_data)
            mvcnn_data_list.append(mvcnn_data[0])  # Assuming mvcnn_data is a tuple of (data, label)
            mvcnn_labels.append(mvcnn_data[1])

    # Batch GNN data
    gnn_batch = Batch.from_data_list(gnn_data_list)

    mvcnn_batch = torch.cat(mvcnn_data_list, dim=0)  # Concatenate along the batch dimension
    mvcnn_labels = torch.tensor(mvcnn_labels, dtype=torch.long)  # Repeat labels for each view

    return gnn_batch, (mvcnn_batch, mvcnn_labels)



def _get_hybrid_loaders(data_root_gnn, data_root_cnn, batch_size, category_mapping): 
    train_dataset = HybridDataset(data_root_gnn, data_root_cnn, category_mapping, partition="train")
    val_dataset = HybridDataset(data_root_gnn, data_root_cnn, category_mapping, partition="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=hybrid_collate_fn)

    return train_loader, val_loader



def train_hybrid(data_root_gnn, data_root_cnn, category_mapping, config_hybrid, config_mvcnn, config_arcgnn, checkpoint_dir_mvcnn, checkpoint_dir_arcgnn , checkpoint_dir_hybrid, pretrained = True):
    num_classes = len(category_mapping)
    batch_size = config_hybrid["batch_size"]
    learning_rate = config_hybrid["learning_rate"]
    weight_decay = config_hybrid["weight_decay"]
    epochs = config_hybrid["epochs"]
    num_views=config_mvcnn['num_views']
    cnn_name=config_mvcnn['cnn_name']


    if pretrained:
        # Initialize both models (assuming they are pre-trained or to be trained)
        arcgnn_model = ArcGNN(num_node_features=81, num_classes=len(category_mapping)).load(checkpoint_dir_arcgnn)  # Adjust parameters
        svcnn_model = SVCNN(num_classes, pretrained=True, cnn_name=cnn_name) # .load(checkpoint_dir_mvcnn)
        mvcnn_model = MVCNN(svcnn_model, num_classes, num_views=num_views, cnn_name=cnn_name).load(checkpoint_dir_mvcnn)
    else:
        arcgnn_model = train_arc_gnn(
            data_root = data_root_gnn, 
            config = config_arcgnn,
            category_mapping = category_mapping,
            checkpoint_dir= checkpoint_dir_arcgnn
            )
        mvcnn_model=train_mvcnn(
            data_root=data_root_cnn,
            category_mapping = category_mapping,
            config=config_mvcnn,
            checkpoint_dir= checkpoint_dir_mvcnn
            )

    hybrid_model = HybridModel(arcgnn_model, mvcnn_model, num_classes=num_classes)
    
    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = _get_hybrid_loaders(data_root_gnn, data_root_cnn, batch_size, category_mapping)
    
    trainer = Trainer(hybrid_model, train_loader, val_loader, list(category_mapping.keys()), optimizer, loss_fn, checkpoint_dir_hybrid, "HybridModel") 
    trainer.train(epochs)

