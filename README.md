# Hybrid Model Classification
Hybrid classification model for BIM objects, combining CNN and GNN

## Overview
This project integrates Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) to enhance the classification of Building Information Modeling (BIM) objects. By combining geometric feature extraction with contextual relationship understanding, this hybrid approach aims to significantly improve classification accuracy and robustness.

## Project Structure
### /src 
- `mvcnn.py` - Implements the Multi-View Convolutional Neural Network for processing 3D objects with 2D images from sevral angles.
- `arcgnn.py` - Implements the Architecture Graph Neural Network for processing graph-based representations.
- `hybrid.py` - Combines MVCNN and ArcGNN outputs using a fusion layer for final classification.
- `Trainer.py` - Core training script for managing the training process of the models.
- `datasets.py` - Handles data preprocessing and loading for training and evaluation.
- `models.py` - Contains the definitions of the MVCNN, SVCNN, ArcGNN, and Hybrid models.
- `train_model.py` - Entry script that sets up and initiates the training process and configurations.

### /notebooks
- `MVCNN-Evaluation.ipynb` - evaluate pretrained model and visulize results
- `ArcGNN-Evaluation.ipynb` - evaluate pretrained model and visulize results
- `HybridModel-Evaluation.ipynb` - evaluate pretrained model and visulize results

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
git clone https://github.com/avishagnevo/HybridClassification.git
cd HybridClassification

# Create a virtual enviorment for the project (recommended)
python3 -m venv ifcnet-env

# Activate the Python environment and install required dependencies
source ifcnet-env/bin/activate
pip install -r requirements.txt

# Run the training script for a specific model after you have inserted the data
# Replace {Enum of the model you want to train} with MVCNN, ArcGNN, or HybridModel
python src/models/train_model.py {Enum of the model you want to train}
```

For example (after creatign the virtual env):

```bash
(base) avishagnevo@Avishags-MBP archi_project % cd ifcnet-models-master
(base) avishagnevo@Avishags-MBP ifcnet-models-master % source ifcnet-env/bin/activate
(ifcnet-env) (base) avishagnevo@Avishags-MBP ifcnet-models-master % python src/models/train_model.py HybridModel
```

# Download and set the project data to put in /data directory

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

## Citation
If you use the IFCNet dataset or code please cite:
```
@inproceedings{emunds2021ifcnet,
  title={IFCNet: A Benchmark Dataset for IFC Entity Classification},
  author={Emunds, Christoph and Pauen, Nicolas and Richter, Veronika and Frisch, Jérôme and van Treeck, Christoph},
  booktitle = {Proceedings of the 28th International Workshop on Intelligent Computing in Engineering (EG-ICE)},
  year={2021},
  month={June},
  day={30}
}
```

## Acknowledgements
The code for the MVCNN and ArcGNN neural networks is based on the implementations of the original publications:
* [RWTH-E3D](https://github.com/RWTH-E3D/ifcnet-models) 
* [jongchyisu](https://github.com/jongchyisu/mvcnn_pytorch) 
* [Guy Austern](https://www.mdpi.com/2075-5309/14/2/527)

The structure of this repository is loosely based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/)

## Conclusion
This project demonstrates the potential of integrating CNNs with GNNs to improve 3D object classification with contextual connections.
