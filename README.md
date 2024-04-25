# HybridClassification
Hybrid classification model for BIM objects, combining CNN and GNN

## Overview
This project integrates Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) to enhance the classification of Building Information Modeling (BIM) objects. By combining geometric feature extraction with contextual relationship understanding, this hybrid approach aims to significantly improve classification accuracy and robustness.

## Project Structure

- `mvcnn.py` - Implements the Multi-View Convolutional Neural Network for processing 3D objects with 2D images from sevral angles.
- `arcgnn.py` - Implements the Architecture Graph Neural Network for processing graph-based representations.
- `hybrid.py` - Combines MVCNN and ArcGNN outputs using a fusion layer for final classification.
- `Trainer.py` - Core training script for managing the training process of the models.
- `datasets.py` - Handles data preprocessing and loading for training and evaluation.
- `models.py` - Contains the definitions of the MVCNN, SVCNN, ArcGNN, and Hybrid models.
- `train_model.py` - Entry script that sets up and initiates the training process and configurations.

## Methodology

### Data Collection and Preprocessing
- **Data Source:** 3D models are sourced from BIM repositories supplied by prof. Guy Austern and public IFC objects data.
- **IFCNetCore Dataset:** Utilizes a subset of the IFCNet dataset, focusing on 20 classes out of 19,613 objects for balanced training.
- **Preprocessing:** Uses `make_mvcnn_data.py` for rendering 2D images from multiple views for the CNN part and `graph_splitter.ipynb` along with `helpers.py` for creating subgraph representations for the GNN model.

### Model Development
- **CNN and GNN Architectures:** Defined in `models.py`, these include specialized layers and configurations tailored for BIM object features.
- **Training:** Managed by `Trainer.py`, incorporating cross-entropy loss, Adam optimizer, and data augmentation strategies.

### Integration and Evaluation
- **Hybrid Model:** Combines features from both MVCNN and ArcGNN using a fusion layer to enhance classification capabilities.
- **Metrics:** Accuracy, precision, recall, and F1-score are tracked to evaluate model performance against training, validation, and test sets.

## Environment Setup and Execution

A dedicated Python environment is provided to manage dependencies and ensure reproducibility. Follow these steps to activate the environment and run the models:

```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/yourgithub/repo.git
cd repo/ifcnet-models-master

# Activate the Python environment
source ifcnet-env/bin/activate

# Run the training script for a specific model
# Replace {Enum of the model you want to train} with MVCNN, ArcGNN, or HybridModel
python src/models/train_model.py {Enum of the model you want to train}
```

For example:

```bash
(base) avishagnevo@Avishags-MBP archi_project % cd ifcnet-models-master
(base) avishagnevo@Avishags-MBP ifcnet-models-master % source ifcnet-env/bin/activate
```

Download the data for the MVCNN model you want to train and place them in the corresponding data folder:
* [MVCNN](https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePng.7z)

```bash
mkdir -p data/processed/MVCNN
cd data/processed/MVCNN
wget https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePng.7z
7z x IFCNetCorePng.7z
```

Get data for the ArcGNN model (BIM jsons) you want to train and place them in the corresponding data folder in a dirctory called 'raw':

```bash
mkdir -p data/processed/ArcGNN/RevitBuildings
cd data/processed/ArcGNN/RevitBuildings
```
Now run the graph_splitter.ipynb of the raw jsons and place the created subgraphs (by building) in the corresponding data folder in a dirctory called 'subgraphs' for exmple:

```bash
json_path = '/Users/avishagnevo/Desktop/archi_project/ifcnet-models-master/data/processed/ArcGNN/RevitBuildings/raw'
output_path = '/Users/avishagnevo/Desktop/archi_project/ifcnet-models-master/data/processed/ArcGNN/RevitBuildings/subgraphs'

file_list = os.listdir(json_path) 

for file_name in file_list:
  file = os.path.join(json_path, file_name)
  print ('opening ' + file)
  with open(file, encoding="utf8") as f:
    json_data = json.load(f)
    edges = json_data['edges'][0]['relations'] #### the 2nd index depicts the type of relation
    elements = json_data['elements_data']
    G = helpers.CreateGraph(elements,edges)
    subgraphs = CreateSubgraphs(elements, G)
    #### the 2nd index depicts the type of relation, 0  is intersections 
    f.close()

  outfile = os.path.join(output_path, file_name[:-5]+ '_subgraphs.json')
  with open(outfile, 'w', encoding="utf8") as fout:
    json.dump(subgraphs, fout)
    fout.close()
```

When building the ArchitecturalDataset the program would automaticlly create a directory named 'processed_subgraphs', available (after running once) at:
```bash
cd data/processed/ArcGNN/RevitBuildings/processed_subgraphs
```

## Installation

```bash
cd ifcnet-models-master
source ifcnet-env/bin/activate
pip install -r requirements.txt
```

## Conclusion

This project demonstrates the potential of integrating CNNs with GNNs to improve 3D object classification with contextual connections.
