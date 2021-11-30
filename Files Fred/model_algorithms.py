from rdkit import Chem
import pandas as pd
import deepdock
from deepdock.models import *
from deepdock.DockingFunction import dock_compound_research, dock_compound_research
from Algorithms import *
from hybrid_algorithms import *

import numpy as np
import torch
import os

""" Add your relative path to DeepDock here, for me Files Fred and DeepDock are at the same level """

abs_path = (os.path.abspath(os.path.join(os.path.dirname(__file__),'..','DeepDock\\data')))

# Specify which target to load with indent
ident = "1z6e"
protein_id = ident + "_protein.ply"
ligand_id = ident + "_ligand.mol2"
csv_id = ident + "_target.csv"


# Specify the ligand and target to optimise 
target_ply = os.path.join(abs_path,protein_id)
real_mol = Chem.MolFromMol2File(os.path.join(abs_path,ligand_id),sanitize=False, cleanupSubstructures=False)



# set the random seeds for reproducibility
np.random.seed(123)
torch.cuda.manual_seed_all(123)

# Def
# ining DeepDock model to use

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ligand_model = LigandNet(28, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(4, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10, dist_threhold=7.).to(device)

# Load pretrained model 
checkpoint = torch.load(deepdock.__path__[0]+'/../Trained_models/DeepDock_pdbbindv2019_13K_minTestLoss.chk', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict']) 

class save_csv():
    def __init__(self,csv_id):
        self.data = dict()
        self.csv_id = csv_id
    def prepare_data(self,my_object,curr_alg):
            self.data[curr_alg] = [my_object.feval,my_object.n_feval]

    def store_data(self):
        df_rates = pd.DataFrame.from_dict(self.data)
        df_rates.to_csv(self.csv_id,index=False) 

my_algorithms = {"ABC":mayfly_alg}

save_data = save_csv(csv_id)
for name,alg in my_algorithms.items():
    algorithm = my_algorithms[name]
    print("running")
    result = dock_compound_research(real_mol, target_ply, model, algorithm, seed=123,device=device,maxfeval=50000)
    print(result.feval[-1])
    save_data.prepare_data(result,name)

save_data.store_data()           