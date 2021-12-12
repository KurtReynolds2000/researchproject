from rdkit import Chem
import pandas as pd
import deepdock
from deepdock.models import *
from deepdock.DockingFunction import dock_compound_research, dock_compound
from Algorithms import *
from hybrid_algorithms import *

import numpy as np
import torch
import os

""" Add your relative path to DeepDock here, for me Files Fred and DeepDock are at the same level """

abs_path = (os.path.abspath(os.path.join(os.path.dirname(__file__),'..','DeepDock\\data')))

class save_csv():
    def __init__(self,csv_id):
        self.data = dict()
        self.csv_id = csv_id
    def prepare_data(self,my_object,curr_alg):
            self.data[curr_alg] = [my_object.feval,my_object.n_feval]

    def store_data(self):
        df_rates = pd.DataFrame.from_dict(self.data)
        df_rates.to_csv(self.csv_id,index=False) 

# set the random seeds for reproducibility
np.random.seed(123)
torch.cuda.manual_seed_all(123)

# Defining DeepDock model to use

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ligand_model = LigandNet(28, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(4, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10, dist_threhold=7.).to(device)

# Load pretrained model 
checkpoint = torch.load(deepdock.__path__[0]+'/../Trained_models/DeepDock_pdbbindv2019_13K_minTestLoss.chk', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict']) 

idents = ["1z6e","2br1","2wtv","2yge","4f2w","4ivd"]
for k in idents:
    # Specify which target to load with indent
    ident = k
    protein_id = ident + "_protein.ply"
    ligand_id = ident + "_ligand.mol2"
    csv_id = ident + "_target.csv"


    # Specify the ligand and target to optimise 
    target_ply = os.path.join(abs_path,protein_id)
    real_mol = Chem.MolFromMol2File(os.path.join(abs_path,ligand_id),sanitize=False, cleanupSubstructures=False)

    my_algorithms = {"HGA":hybrid_genetic,"MA":mayfly_alg,"HDE":hybrid_differential}

    save_data = save_csv(csv_id)
    for name,alg in my_algorithms.items():
        algorithm = my_algorithms[name]
        print("running")
        result = dock_compound_research(real_mol, target_ply, model, algorithm, device=device,maxfeval=75000,seed=123)
        save_data.prepare_data(result,name)

    save_data.store_data()           