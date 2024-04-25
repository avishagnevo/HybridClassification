import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score)
#from ray import tune
from pathlib import Path
import numpy as np
import pickle
import os
import time
import logging
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Trainer:

    def __init__(self, model, train_loader, val_loader, class_names, optimizer,
                loss_fn, checkpoint_dir, model_name, after_load_cb=None):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.class_names = class_names
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir 
        self.after_load_cb = after_load_cb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.load(checkpoint_dir)



    def train(self, epochs, global_step=0):
        print('Training with train')
        
        for epoch in range(global_step, epochs + global_step):
            print(f'EPOCH {epoch}/{epochs}')
            self.model.train()
            all_probs = []
            all_labels = []
            running_loss = 0.0

            # Use this function with your DataLoader
            #check_data_loader_shapes(self.train_loader)


            for batch_idx, batch in enumerate(self.train_loader):
                
                            
                if batch_idx == 1000:
                    break
            
                if not batch:
                    continue

                if batch_idx  % 1 == 0:
                    #print(stop)
                    print(f'batch {batch_idx}/{len(self.train_loader)}')
                
                if self.model_name == 'ArcGNN':
                    # For GNN
                    x, edge_index, labels = batch.x.to(self.device), batch.edge_index.to(self.device), batch.y.to(self.device)
                    labels -=1 # to fit with class indices
                    outputs = self.model(x, edge_index.t(), batch.batch.to(self.device))
                elif self.model_name == 'MVCNN' or self.model_name == 'SVCNN':
                    # For CNN
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    if self.after_load_cb:
                        data = self.after_load_cb(data)
                    outputs = self.model(data)    
                else : #Hybrid Model 
                    gnn_batch, (mvcnn_data, mvcnn_labels) = batch
                    x, edge_index, labels = gnn_batch.x.to(self.device), gnn_batch.edge_index.to(self.device), gnn_batch.y.to(self.device)
                    labels -=1 # to fit with class indices
                    mvcnn_data, mvcnn_labels = mvcnn_data.to(self.device), mvcnn_labels.to(self.device)
                    outputs = self.model(x, edge_index.t(), gnn_batch.batch.to(self.device), mvcnn_data)



                self.optimizer.zero_grad()

                print(outputs)
                print(outputs.shape)
                print(labels)
                print(labels.shape)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
                # Assuming outputs and labels are tensors
                probs = F.softmax(outputs, dim=1).cpu().detach().numpy()
                labels = labels.cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels)

                print()

                

                

            # Concatenate all probabilities and labels from the epoch
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            train_metrics = self.calc_metrics(all_probs, all_labels, running_loss / len(self.train_loader.dataset), "train")
            print(train_metrics)
            val_metrics = self.evaluate()  # Make sure to adjust evaluate() similarly
            print(val_metrics)

            self.save(epoch, self.checkpoint_dir)
        return self.model
            # Log or print your metrics here

    def evaluate(self):
        print('Evaluating with validation')
        self.model.eval()
        all_probs = []
        all_labels = []
        running_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):

                if not batch:
                    continue
                
                if batch_idx + 1 % 10 == 0:
                    print(stop)
                    print(f'batch {batch_idx}/{len(self.val_loader)}')
                
                if self.model_name == 'ArcGNN':
                    # For GNN
                    x, edge_index, labels = batch.x.to(self.device), batch.edge_index.to(self.device), batch.y.to(self.device)
                    labels -=1 # to fit with class indices
                    outputs = self.model(x, edge_index.t(), batch.batch.to(self.device))
                elif self.model_name == 'MVCNN' or self.model_name == 'SVCNN':
                    # For CNN
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    if self.after_load_cb:
                        data = self.after_load_cb(data)
                    outputs = self.model(data)    
                else : #Hybrid Model 
                    gnn_batch, (mvcnn_data, mvcnn_labels) = batch
                    x, edge_index, labels = gnn_batch.x.to(self.device), gnn_batch.edge_index.to(self.device), gnn_batch.y.to(self.device)
                    labels -=1 # to fit with class indices
                    mvcnn_data, mvcnn_labels = mvcnn_data.to(self.device), mvcnn_labels.to(self.device)
                    outputs = self.model(x, edge_index.t(), gnn_batch.batch.to(self.device), mvcnn_data)



                #outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels)

                if batch_idx == 10:  # Limit number of batches for evaluation
                    break

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        metrics = self.calc_metrics(all_probs, all_labels, running_loss / len(self.val_loader.dataset), "val")
        return metrics
        


    def calc_metrics(self, probabilities, labels, loss, tag):
        predictions = np.argmax(probabilities, axis=1)

        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        
        return {
            f"{tag}_loss": loss,
            f"{tag}_accuracy_score": acc,
            f"{tag}_balanced_accuracy_score": balanced_acc,
            f"{tag}_precision_score": precision,
            f"{tag}_recall_score": recall,
            f"{tag}_f1_score": f1
        }


    def save(self, epoch, checkpoint_dir):
        target_path = f'{checkpoint_dir}/{self.model_name}Weights+Optimizer{epoch}'
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), target_path) 



    def load(self, checkpoint_dir):
        if not checkpoint_dir: return
        
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
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        print(f"Loaded {self.model_name} from checkpoint '{path}'")


def check_data_loader_shapes(data_loader):
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(stop)
                
        print(f"Batch {batch_idx}")
        if hasattr(batch, 'x'):
            print(f"  Feature tensor shape (x): {batch.x.shape}")
        if hasattr(batch, 'y'):
            print(f"  Label tensor shape (y): {batch.y.shape}")
        if hasattr(batch, 'edge_index'):
            print(f"  Edge index shape: {batch.edge_index.shape}")
        if hasattr(batch, 'batch'):
            print(f"  Batch tensor shape: {batch.batch.shape}")
        # Add any other attributes you expect in your batches


