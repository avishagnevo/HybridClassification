import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from src.models.Trainer import Trainer
from torch_geometric.data import DataLoader as PyGDataLoader
from src.data.dataset import GNNArchitecturalDataset, MultiviewImgDataset
from src.models.models import ArcGNN
from src.models.mvcnn import _get_mvcnn_datasets


def _get_gnn_loaders(data_root, batch_size, category_mapping): 
    class_names = category_mapping.keys() # TODO check if fine
    #mvcnn_dataset_train, mvcnn_dataset_val=_get_mvcnn_datasets(data_root, class_names, batch_size, num_views=12, pretrained=True, eval_on_test=False)

    train_dataset = GNNArchitecturalDataset(data_root, class_names, partition="train")
    val_dataset = GNNArchitecturalDataset(data_root, class_names, partition="val")  # Assume you have a validation partition
    print("Number of training samples:", len(train_dataset))

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_arc_gnn(data_root, category_mapping, config, checkpoint_dir):
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    epochs = config["epochs"]

    num_classes = len(category_mapping)
    num_node_features = 81  # Assuming each node has 81 features as defined earlier
    class_names = list(category_mapping.keys())

    model = ArcGNN(num_node_features=num_node_features, num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader, val_loader = _get_gnn_loaders(data_root, batch_size, category_mapping)

    trainer = Trainer(model, train_loader, val_loader, class_names, optimizer, nn.CrossEntropyLoss(), checkpoint_dir, "ArcGNN")
    trainer.train(epochs)
    arcgnn = trainer.model
    return arcgnn
