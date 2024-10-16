import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import numpy as np

from schnetpack.data import AtomsDataModule
from numpy import linalg as LA

from ase import Atoms

import time

torch.set_float32_matmul_precision('high')

# размеры обучающей, валидационный, тестовой выборок
num_train=80000
num_val=15000
num_test=5000
num_workers = 8   # параметр, отвечающий за распараллеливание загрузки данных в датасет
pin_memory=False  # True, если расчеты ведутся на GPU
device_type = 'cpu'
cutoff = 5.0      # !!!!
batch_size = 1000 # !!!!

dataset_file_path = 'dataset_100000_pbc.db'
model_file_path = 'phosphorene_model_256_7_4000'

# формируем файл split.npz, в котором храним 3 файла:
# индексы элементов основной обучающей выборки (train_idx.npy)
# индексы элементов валидационной выборки (val_idx.npy)
# индексы элементы тестовой выборки (test_idx.npy)
if os.path.exists('./split.npz'):
    os.remove('./split.npz')
train_idx = np.array(range(num_train))
val_idx = np.array(range(num_train, num_train+num_val))
test_idx = np.array(range(num_train+num_val, num_train+num_val+num_test))
np.savez("./split.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

phosphorene_data = AtomsDataModule(
    os.path.join('./forcetut', dataset_file_path),
    batch_size=batch_size,
    num_train=num_train, # длина массива из файла train_idx.npy
    num_val=num_val,    # длина массива из файла val_idx.npy
    num_test=num_test,   # длина массива из файла test_idx.npy
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units = {'energy':'eV', 'forces':'eV/Ang'},
    num_workers = num_workers,
    split_file = "./split.npz",
    pin_memory=pin_memory, # set to false, when not using a GPU
    load_properties = ['energy', 'forces']
)
phosphorene_data.prepare_data()
phosphorene_data.setup()

# датасет загружен

# проверка модели
model_path = os.path.join('./forcetut', model_file_path)
best_model = torch.load(model_path, map_location=torch.device(device_type))

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.TorchNeighborList(cutoff=cutoff), dtype=torch.float32
)

temp = np.zeros(num_test*16)
temp = temp.reshape(num_test,16)
start_time = time.time()
for i in range(num_test):
    if i%100 == 0:
        print("%d %s" % (i,(time.time() - start_time)))
    structure = phosphorene_data.dataset[i+num_train+num_val]
    atoms = Atoms(
        numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
    )
    # convert atoms to SchNetPack inputs and perform prediction
    inputs = converter(atoms)
    results = (best_model(inputs)['forces']).detach().numpy()
    for j in range(16):
        temp[i][j] = LA.norm(results[j])
print("--- Done %s seconds ---\n" % (time.time() - start_time))
lst_model_data_all_by_atoms=temp.transpose()
with open('lst_model_data_all_by_atoms_2000.npy', 'wb') as f:
    np.save(f, lst_model_data_all_by_atoms)
