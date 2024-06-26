import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import json
from pathlib import Path
from datetime import datetime
from src.data import SingleImgDataset, MultiviewImgDataset
from src.models.models import MVCNN, SVCNN
from src.models.Trainer import Trainer


def _get_train_val_transforms(pretrained=True):
    train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    val_transform = [
        transforms.ToTensor()
    ]

    if pretrained:
        # Use ImageNet stats
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        # Use IFCNet stats
        norm = transforms.Normalize(mean=[0.911, 0.911, 0.911], std=[0.148, 0.148, 0.148])
    
    train_transform.append(norm)
    val_transform.append(norm)

    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    return train_transform, val_transform


def cnn_collate_fn(batch):
    cnn_data_list = []
    cnn_labels = []


    for cnn_data in batch:
        if cnn_data is not None:
            cnn_data_list.append(cnn_data[0])  # Assuming mvcnn_data is a tuple of (data, label)
            cnn_labels.append(cnn_data[1])

    # Batch MVCNN data
    if len(cnn_data_list) == 0:
        return None

    cnn_batch = torch.stack(cnn_data_list)
    print('cnn_batch',cnn_batch)
    print('cnn_batch.shape',cnn_batch.shape)
    cnn_labels = torch.tensor(cnn_labels, dtype=torch.long)
    print('cnn_labels',cnn_labels)

    return cnn_batch, cnn_labels


def _get_svcnn_loaders(data_root, class_names, batch_size, pretrained=True, eval_on_test=False):
    train_transform, val_transform = _get_train_val_transforms(pretrained=pretrained)

    if eval_on_test:
        train_dataset = SingleImgDataset(data_root, class_names, partition="train", transform=train_transform)
        val_dataset = SingleImgDataset(data_root, class_names, partition="test", transform=val_transform)
    else:
        train_dataset = SingleImgDataset(data_root, class_names, partition="train", transform=train_transform)
        val_dataset = SingleImgDataset(data_root, class_names, partition="train", transform=val_transform)

        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7 * len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])

    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=cnn_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=cnn_collate_fn)
    return train_loader, val_loader


def _pretrain_single_view(data_root, class_names, epochs, batch_size,
                        learning_rate, weight_decay, checkpoint_dir,
                        pretrained=True, cnn_name="vgg11", eval_on_test=False):
    model = SVCNN(nclasses=len(class_names), pretrained=pretrained, cnn_name=cnn_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader = _get_svcnn_loaders(data_root, class_names, batch_size,
                                                    pretrained=pretrained, eval_on_test=eval_on_test)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, nn.CrossEntropyLoss(), checkpoint_dir, "SVCNN")
    trainer.train(epochs)
    return model


def _get_mvcnn_loaders(data_root, class_names, batch_size, num_views=12, pretrained=True, eval_on_test=False):

    train_dataset, val_dataset =_get_mvcnn_datasets(data_root, class_names, batch_size, num_views=12, pretrained=True, eval_on_test=False)

    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=cnn_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, collate_fn=cnn_collate_fn)
    return train_loader, val_loader


def _get_mvcnn_datasets(data_root, class_names, batch_size, num_views=12, pretrained=True, eval_on_test=False):
    train_transform, val_transform = _get_train_val_transforms(pretrained=pretrained)
    
    if eval_on_test:
        train_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="train", transform=train_transform)
        val_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="test", transform=val_transform)
    else:
        train_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="train", transform=train_transform)
        val_dataset = MultiviewImgDataset(data_root, class_names, num_views, partition="train", transform=val_transform)

        np.random.seed(42)
        perm = np.random.permutation(range(len(train_dataset)))
        train_len = int(0.7 * len(train_dataset))
        train_dataset = Subset(train_dataset, perm[:train_len])
        val_dataset = Subset(val_dataset, perm[train_len:])

    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")
        
    return train_dataset, val_dataset




def _train_multi_view(svcnn, data_root, class_names, epochs, batch_size,
                    learning_rate, weight_decay, checkpoint_dir,
                    pretrained=True, cnn_name="vgg11", num_views=12, eval_on_test=False):
    model = MVCNN(svcnn, nclasses=len(class_names), num_views=num_views, cnn_name=cnn_name)

    # Can remove SVCNN after layers have been copied into MVCNN to safe memory
    del svcnn

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, val_loader = _get_mvcnn_loaders(data_root, class_names, batch_size, num_views=12,
                                                    pretrained=pretrained, eval_on_test=eval_on_test)

    trainer = Trainer(model, train_loader, val_loader, class_names,
        optimizer, nn.CrossEntropyLoss(), checkpoint_dir, "MVCNN",
        after_load_cb=lambda x: x.view(-1, *x.shape[-3:]))
    trainer.train(epochs)
    mvcnn = trainer.model
    return mvcnn


def train_mvcnn(config, checkpoint_dir=None, data_root=None, category_mapping=None, eval_on_test=False):        
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    pretrained = config["pretrained"]
    cnn_name = config["cnn_name"]
    num_views = config["num_views"]
    epochs = config["epochs"]
    class_names = list(category_mapping.values())
    print('class_names', class_names)

    svcnn = _pretrain_single_view(data_root, class_names, epochs, batch_size,
                                learning_rate, weight_decay, checkpoint_dir,
                                pretrained=pretrained, cnn_name=cnn_name, eval_on_test=eval_on_test)
    mvcnn =_train_multi_view(svcnn, data_root, class_names, epochs, int(batch_size/num_views),
                            learning_rate, weight_decay, checkpoint_dir,
                            pretrained=pretrained, cnn_name=cnn_name, num_views=num_views, eval_on_test=eval_on_test)
    return mvcnn
