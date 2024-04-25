import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from src.data.util import read_ply
from torch_geometric.data import Data, Dataset 
import json 
import os 
import random
import pandas as pd



class IFCNetPly(Dataset):

    def __init__(self, data_root, class_names, partition="train", transform=None):
        self.transform = transform
        self.data_root = data_root
        self.class_names = class_names
        self.partition = partition
        self.files = sorted(data_root.glob(f"**/{partition}/*.ply"))

        self.cache = {}

    def __getitem__(self, idx):
        if idx in self.cache:
            pointcloud, label = self.cache[idx]
        else:
            f = self.files[idx]
            df = read_ply(f)
            pointcloud = df["points"].to_numpy()
            class_name = f.parts[-3]
            label = self.class_names.index(class_name)
            self.cache[idx] = (pointcloud, label)

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud, label

    def __len__(self):
        return len(self.files)


class IFCNetNumpy(Dataset):

    def __init__(self, data_root, max_faces, class_names, partition='train'):
        self.data_root = data_root
        self.max_faces = max_faces
        self.partition = partition
        self.files = sorted(data_root.glob(f"**/{partition}/*.npz"))
        self.class_names = class_names

    def __getitem__(self, idx):
        path = self.files[idx]
        class_name = path.parts[-3]
        label = self.class_names.index(class_name)
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face)
        neighbor_index = torch.from_numpy(neighbor_index)
        target = torch.tensor(label, dtype=torch.long)
        data = torch.cat([face, neighbor_index], dim=1)

        return data, target

    def __len__(self):
        return len(self.files)


class MultiviewImgDataset(Dataset):

    def __init__(self, root_dir, classnames, num_views, partition="train", transform=None):
        self.classnames = classnames
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()]) #
        self.num_views = num_views

        self.filepaths = sorted(root_dir.glob(f"**/{partition}/*.png"))
        self.filepaths = np.array(self.filepaths).reshape(-1, self.num_views)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        
        paths = self.filepaths[idx]
        class_name = paths[0].parts[-3]

        if class_name not in self.classnames:
            return None

        label = self.classnames.index(class_name)

        imgs = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)  

        return torch.stack(imgs), label


class SingleImgDataset(Dataset):

    def __init__(self, root_dir, classnames, partition="train", transform=None):
        self.classnames = classnames
        self.transform = transform
        self.root_dir = root_dir

        self.filepaths = sorted(root_dir.glob(f"**/{partition}/*.png"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.parts[-3]

        if class_name not in self.classnames:
            return None
        
        label = self.classnames.index(class_name)

        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label
        

class GNNArchitecturalDataset(Dataset): #        

    def __init__(self, root_dir, classnames, partition="train", transform=None, pre_transform=None): 
        self.root_dir = root_dir
        self.classnames = classnames # Assuming category_mapping is defined globally or passed in
        self.processed_file_count = 0
        self.bin_path = None #TODO 
        # Mapping category names to integers
        self.category_to_idx = {category: idx + 1 for idx, category in enumerate(sorted(self.classnames))}  # Ensure this matches with how you handle classes elsewhere

        super(GNNArchitecturalDataset, self).__init__(root_dir, transform, pre_transform)
    

    @property
    def raw_file_names(self):
        #raw_file_names = [file for file in os.listdir(f'{self.root_dir}/raw') if file.endswith('json')]
        raw_file_names = [file for file in os.listdir(f'{self.root_dir}/subgraphs') if file.endswith('json')]
        return raw_file_names

    @property
    def processed_file_names(self):
        # This example assumes pre-processing is done externally,
        # and processed files are ready to be loaded. Adjust accordingly.
        return [f'processed_subgraph_{i}.pt' for i in range(self.processed_file_count)]

    def download(self):
        # Download to self.raw_dir (optional)
        pass

    def process(self):
        ''''''
        processed_subgraphs_path = os.path.join(self.root_dir, 'processed_subgraphs')
        directory_empty = not os.listdir(processed_subgraphs_path)
        if not directory_empty:
            self.processed_file_count = self.count_files_in_directory(processed_subgraphs_path)
            print(f"There are {self.processed_file_count} files to use in {processed_subgraphs_path}")
            return #no need to process the files all over again


        for path_idx, raw_path in enumerate(self.raw_file_names):
            print('Processing:', raw_path)
            if raw_path.endswith('.json'):
                full_path = os.path.join(self.root_dir, 'subgraphs', raw_path)
                print('File path:', full_path)
                with open(full_path, 'r') as f:
                    json_data = json.load(f)

                    for subgraph in json_data:  # Assuming each file might contain multiple subgraphs
                        data = self.process_subgraph(subgraph, self.bin_path)
                    
                        if data:
                            # Saving the processed data
                            save_path = os.path.join(self.root_dir, 'processed_subgraphs', f'processed_subgraph_{self.processed_file_count}.pt')
                            torch.save(data, save_path)
                            self.processed_file_count += 1
                            #print('Saved:', save_path)

                        
                    print(f'Saved all subgraphs in file {raw_path}')    



    def process_subgraph(self, subgraph, bin_path):
        if subgraph['category'] not in self.classnames:
            return None

        encoding = GenerateOneHotEncoding(subgraph['nodes'], bin_path)
        #return encoding 
        cat_encoding = self.encode_categories(subgraph)
        encoding = pd.concat([encoding, cat_encoding], axis=1)

        x = torch.tensor(encoding.values, dtype=torch.float32)
        y = torch.tensor([self.category_to_idx.get(subgraph['nodes'][0]['category_name'])], dtype=torch.long) #give the whole subgraph the label of the first node
        edge_index = torch.tensor(subgraph['adj'], dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index, y=y)

    def encode_categories(self, graph):

        nodes = graph['nodes'] 
        node_categories = [0] + [self.category_to_idx.get(node['category_name'] , -1) for node in nodes[1:]]
        encoding = torch.nn.functional.one_hot(torch.tensor(node_categories, dtype=torch.long), num_classes=len(self.category_to_idx) + 1)
        return pd.DataFrame(encoding.numpy(), columns=['Root'] + list(self.classnames)).astype(bool)


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        #data = torch.load(os.path.join(self.root_dir,  'processed_subgraphs' , self.processed_file_names[idx])) #maybe chnage to more informative names ?
        # Adjust get method to match the file naming convention
        filename = f'processed_subgraph_{idx}.pt'  # Adjust this according to actual naming convention
        filepath = os.path.join(self.root_dir, 'processed_subgraphs', filename)
        data = torch.load(filepath)
        return data     
    
    def count_files_in_directory(self, directory_path):
        try:
            # List all entries in the directory given by "directory_path"
            directory_contents = os.listdir(directory_path)
            
            # Count how many of these entries are files using os.path.isfile
            file_count = sum(os.path.isfile(os.path.join(directory_path, entry)) for entry in directory_contents)
            return file_count
        except FileNotFoundError:
            print(f"Directory not found: {directory_path}")
            return 0
        except PermissionError:
            print(f"Permission denied accessing the directory: {directory_path}")
            return 0
        

class HybridDataset(Dataset):
    def __init__(self, gnn_data_root, mvcnn_data_root, category_mapping, partition="train", transform=None):
        """
        Initialize the hybrid dataset.
        :param gnn_data_root: Path to the GNN architectural data.
        :param mvcnn_data_root: Path to the Multiview CNN data.
        :param category_mapping: A dictionary mapping GNN categories to MVCNN categories.
        :param transform: Optional transform to be applied on a sample.
        """
        self.gnn_dataset = GNNArchitecturalDataset(gnn_data_root, classnames=list(category_mapping.keys()),  transform=transform)
        self.mvcnn_dataset = MultiviewImgDataset(mvcnn_data_root, classnames=list(category_mapping.values()), num_views=12, transform=transform)
        self.category_mapping = category_mapping
        self.transform = transform
        self.category_to_idx = {category: idx + 1 for idx, category in enumerate(sorted(category_mapping.keys()))}
        self.idx_to_category = {idx: category for category, idx in self.category_to_idx.items()}

        # Precompute indices for each MVCNN class
        self.mvcnn_indices_by_class = {category: [] for category in category_mapping.values()}
        for idx, item in enumerate(self.mvcnn_dataset):
            if not item : 
                continue
            mvcnn_class = self.mvcnn_dataset.classnames[item[1]]
            self.mvcnn_indices_by_class[mvcnn_class].append(idx)  

    def __len__(self):
        return len(self.gnn_dataset)

    def __getitem__(self, idx):
        # Fetch the GNN data point
        gnn_data = self.gnn_dataset[idx]
        category_name = self.idx_to_category.get(gnn_data.y.item())
        
        # Map the GNN category to MVCNN class and retrieve precomputed indices
        mvcnn_class = self.category_mapping.get(category_name, None)
        if mvcnn_class is None:
            raise ValueError(f"Category {category_name} not found in category mapping.")

        mvcnn_indices = self.mvcnn_indices_by_class.get(mvcnn_class, [])
        if not mvcnn_indices:
            raise ValueError(f"No matching MVCNN data found for category {category_name}.") #
        mvcnn_idx = random.choice(mvcnn_indices)
        mvcnn_data = self.mvcnn_dataset[mvcnn_idx]

        return gnn_data, mvcnn_data


def OneHotDynamic(frame, col_name, num_bins, extremes, bin_edges=(0,1)):
    col = pd.to_numeric((frame[col_name].clip(lower=0) - extremes[0]) / (extremes[1] - extremes[0]), errors='coerce')

    # Use pd.cut to ensure consistency in the number of bins
    binned = pd.cut(col, bins=np.linspace(bin_edges[0], bin_edges[1], num_bins + 1), labels=False, include_lowest=True)

    dummies = pd.get_dummies(binned, prefix=col_name)
    #print('dummies', dummies)
    
    # Ensure all possible bins are represented even if some bins have no data
    expected_columns = [f"{col_name}_{i}" for i in range(num_bins)]
    for col in expected_columns:
        if col not in dummies:
            dummies[col] = False
    
    return dummies[expected_columns]  # Return in a consistent order


def GenerateOneHotEncoding(elements, bin_path):
    df = pd.DataFrame(elements)

    pd.set_option('display.max_rows', None)  # or use a specific large number if 'None' is too slow
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)  # Adjusts the width of each column to show full content

    # extremes   
    BB_X_dim_ex, BB_Y_dim_ex, BB_Z_dim_ex, BB_volume_ex, solid_volume_ex, relative_height_ex, num_of_components_ex = [0, 15356], [0, 31778], [0, 3652], [0, 2052687602], [0, 1623027792], [0.0, 1.0], [1, 428]

    BB_X_dim = OneHotDynamic(df,'BB_X_dim',10, BB_X_dim_ex)
    BB_Y_dim = OneHotDynamic(df,'BB_Y_dim',10, BB_Y_dim_ex)
    BB_Z_dim = OneHotDynamic(df,'BB_Z_dim',10, BB_Z_dim_ex)
    BB_volume = OneHotDynamic(df,'BB_volume',10, BB_volume_ex)
    solid_volume = OneHotDynamic(df,'solid_volume',10, solid_volume_ex)
    relative_height = OneHotDynamic(df,'relative_height',10, relative_height_ex)
    num_of_components = OneHotDynamic(df,'num_of_components', 10, num_of_components_ex)
    encoding = pd.concat([BB_X_dim, BB_Y_dim,BB_Z_dim,BB_volume,solid_volume, relative_height,num_of_components],axis=1)

    return encoding

