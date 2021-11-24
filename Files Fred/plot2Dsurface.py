import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from deepdock.DockingFunction import dock_compound_research, optimze_conformation
from rdkit import Chem
from deepdock.models import *
import deepdock
import matplotlib.pyplot as plt

from deepdock.utils.data import *

def dock_c(mol, target_ply, model, dist_threshold=3., seed=None, device='cpu'):
    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
            mol=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            
        # Prepare ligand and target to be used by pytorch geometric
        ligand = from_networkx(mol2graph.mol_to_nx(mol))
        ligand = Batch.from_data_list([ligand])
            
    else:
        raise Exception('mol should be an RDKIT molecule')

    if not isinstance(target_ply, Batch):
        if isinstance(target_ply, str):
            # Prepare ligand and target to be used by pytorch geometric
            target = Cartesian()(FaceToEdge()(read_ply(target_ply)))
            target = Batch.from_data_list([target])
        else:
            raise Exception('target should be a string with the ply file paht or a Batch instance')

    # Use the model to generate distance probability distributions
    model.eval()
    ligand, target = ligand.to(device), target.to(device)
    pi, sigma, mu, dist, atom_types, bond_types, batch = model(ligand, target)

    #Set optimization function
    opt = optimze_conformation(mol=mol, target_coords=target.pos.cpu(), n_particles=1,pi=pi.cpu(), mu=mu.cpu(), sigma=sigma.cpu(), dist_threshold=dist_threshold, seed=seed)
    print(target.pos.cpu().min(0)[0].numpy(), target.pos.cpu().max(0)[0].numpy())

    return opt

np.random.seed(123)
torch.cuda.manual_seed_all(123)


abs_path = (os.path.abspath(os.path.join(os.path.dirname(__file__),'..','DeepDock\\data')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ligand_model = LigandNet(28, residual_layers=10, dropout_rate=0.10)
target_model = TargetNet(4, residual_layers=10, dropout_rate=0.10)
model = DeepDock(ligand_model, target_model, hidden_dim=64, n_gaussians=10, dropout_rate=0.10, dist_threhold=7.).to(device)

checkpoint = torch.load(deepdock.__path__[0]+'/../Trained_models/DeepDock_pdbbindv2019_13K_minTestLoss.chk', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict']) 

ident = "1z6e"
protein_id = ident + "_protein.ply"
ligand_id = ident + "_ligand.mol2"


# Specify the ligand and target to optimise 
target_ply = os.path.join(abs_path,protein_id)
real_mol = Chem.MolFromMol2File(os.path.join(abs_path,ligand_id),sanitize=False, cleanupSubstructures=False)
array_size = 80

opt = dock_c(real_mol,target_ply,model)

x = np.array([[-0.99374445, -2.56988637, -3.12112576, 10.60745797, 27.68033189,
       56.43378115, -1.93031307,  0.15773386, -0.30088624, -1.9763511 ,
       -0.61640145,  0.54661285,  2.04593077,  0.16918374]]*array_size)

y = x


x_coords = np.linspace(-np.pi,np.pi,array_size)
y_coords = x_coords
y[:,7] = y_coords

z_coords = np.empty((array_size,array_size))

for j in range(array_size):
    y[:,6] = x_coords[j]
    for i in range(array_size):
        z_coords[j,i] = opt.score_conformation(y[i])

x_coords,y_coords = np.meshgrid(x_coords,y_coords)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x_coords,y_coords, z_coords, rstride=1, cstride=1,cmap='jet', edgecolor='none')
ax.set_xlabel('Rotatable Bond 1')
ax.set_ylabel('Rotatable Bond 2')
ax.grid(False)
plt.show()
