import json
from argparse import ArgumentParser
from functools import partial
from enum import Enum
from pathlib import Path
from datetime import datetime
from src.models.mvcnn import train_mvcnn
from src.models.arcgnn import train_arc_gnn
from src.models.hybrid import train_hybrid


'''
To run this file from TERMINAL 
cd {your path to ifcnet-models-master directory}/ifcnet-models-master
source ifcnet-env/bin/activate
which python # just for checking if the enviroment is activated proparly
python src/models/train_model.py {Enum of the model you want to train}

like this: 
(base) avishagnevo@Avishags-MBP archi_project % cd ifcnet-models-master
(base) avishagnevo@Avishags-MBP ifcnet-models-master % source ifcnet-env/bin/activate
'''

class Model(str, Enum):
    MVCNN = "MVCNN"
    ArcGNN = "ArcGNN"
    HybridModel = "HybridModel"


def main(args):
    model = args.model
    config_file = args.config_file
    log_dir = Path(f"./logs/{model.value}")
    log_dir.mkdir(exist_ok=True, parents=True)
    #data_root = Path(f"./data/processed/{model.value}").absolute()
    data_root_cnn = Path(f"./data/processed/MVCNN/IFCNetCore").absolute()
    data_root_gnn = Path(f"./data/processed/ArcGNN/RevitBuildings").absolute() #/subgraphs
    checkpoint_dir = Path(f"./models/{model.value}_pretrained").absolute()

    category_mapping = { # to sample matching IFCNetCore classes and ArcGNN categories
    "Walls": "IfcWall",
    "Furniture": "IfcFurniture",
    "Doors": "IfcDoor",
    "Windows": "IfcDoor", #not in IFC dataset
    "Floors": "IfcWall", #not in IFC dataset
    "Plumbing Fixtures": "IfcSanitaryTerminal",
    "Structural Columns": "IfcBeam",
    "Railings": "IfcRailing",
    "Structural Framing": "IfcBeam", #not in IFC dataset
    "Stairs": "IfcStair"
    }


    if model == Model.MVCNN:
        
        config = {
        "batch_size": 64,
        "cnn_name": "vgg11",
        "epochs": 10,
        "learning_rate": 1.3529557841712963e-05,
        "num_views": 12,
        "pretrained": True,
        "weight_decay": 0.00019094898290886048
        }

        train_mvcnn(
            data_root=data_root_cnn,
            category_mapping= category_mapping,
            config=config,
            eval_on_test=config_file is not None,
            checkpoint_dir= checkpoint_dir
        )


    elif model == Model.ArcGNN :
        # Configuration parameters
        config = {
            "batch_size": 1,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
            "epochs": 3,
            # Add other necessary configuration parameters here
        }
        
        # Start the training process
        train_arc_gnn(
            data_root = data_root_gnn, 
            config = config,
            category_mapping = category_mapping,
            checkpoint_dir= checkpoint_dir
            )


    elif model == Model.HybridModel :
        # Configuration parameters
        config_hybrid = {
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 5e-4,
            "epochs": 3,
            # Add other necessary configuration parameters here
        }

        config_mvcnn = {
            "batch_size": 64,
            "cnn_name": "vgg11",
            "epochs": 30,
            "learning_rate": 1.3529557841712963e-05,
            "num_views": 12,
            "pretrained": True,
            "weight_decay": 0.00019094898290886048
        }

        config_arcgnn = {
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 5e-4,
            "epochs": 30,
            # Add other necessary configuration parameters here
        }

        train_hybrid(data_root_gnn = data_root_gnn, 
                     data_root_cnn = data_root_cnn, 
                     category_mapping = category_mapping, 
                     config_hybrid = config_hybrid,
                     config_mvcnn = config_mvcnn,
                     config_arcgnn = config_arcgnn,
                     checkpoint_dir_arcgnn= Path(f"./models/ArcGNN_pretrained").absolute(),
                     checkpoint_dir_mvcnn = Path(f"./models/MVCNN_pretrained").absolute(),
                     checkpoint_dir_hybrid = checkpoint_dir
                     )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=Model, choices=list(Model))
    parser.add_argument("--config_file", default=None, type=Path)
    args = parser.parse_args()

    main(args)
