import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import numpy as np

# использует ASE database format
from schnetpack.data import ASEAtomsData
from ase import Atoms
from schnetpack.data import AtomsDataModule

torch.set_float32_matmul_precision('high')

# размеры обучающей, валидационный, тестовой выборок
num_train=80000
num_val=15000
num_test=5000
num_workers = 8   # параметр, отвечающий за распараллеливание загрузки данных в датасет
pin_memory = False  # True, если расчеты ведутся на графическом процессоре
device_type = 'cpu' #'gpu', сли расчеты ведутся на графическом процессоре
cutoff = 5.0      # радиус обрезки
batch_size = 1000 # размер батча

dataset_file_path = 'dataset_100000_pbc.db'
model_file_path = 'phosphorene_model_256_9_500_100000_pbc_gpu'

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

# ASE автоматически пересчитывает энергию в электрон-вольты (эВ),
# расстояние - а Ангстремы (А)
# силы - в эВ/А
phosphorene_data = AtomsDataModule(
    os.path.join('./forcetut',dataset_file_path),
    batch_size=batch_size,
    num_train=num_train, # длина массива из файла train_idx.npy
    num_val=num_val,    # длина массива из файла val_idx.npy
    num_test=num_test,   # длина массива из файла test_idx.npy
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    #property_units = {'energy':'eV', 'forces':'eV/Ang'},
    #distance_unit = 'Ang',
    num_workers = num_workers,
    split_file = "./split.npz",
    pin_memory = False # True, если используем графический процессор
    load_properties = ['energy', 'forces']
)
phosphorene_data.prepare_data()
phosphorene_data.setup()

# вычисляем среднее значение и стандартное отклонение среднего значения
means, stddevs = phosphorene_data.get_stats(
    property='energy', divide_by_atoms=True, remove_atomref=False
)
print('Mean atomization energy / atom:', means.item(), flush=True)
print('Std. dev. atomization energy / atom:', stddevs.item(), flush=True)

# В SchNetPack потенциал, вычисляемый нейронной сетью (neural network potential),
# состоит из 3 частей:
# 1) Список входных модулей, которые подготавливают разделенные по пакетам данные
# (batched_data) перед построением представления. Этот список включает в себя,
# в том числе, вычисление попарных расстояний между атомами на основе индексов
# соседей или добавление вспомогательных входов для свойств отклика
# 2) Представление, при помощи какой сети (SchNet или PaiNN) будут строиться
# свойства атомов
# 3) Одна или более выходных моделей для предсказания свойств.

# Используем представление с 9 слоями взаимодействия, с параметром cosine cutoff,
# равным 5 Ангстрем с попарным расстоянием, расширенным на 300 Гауссианов 
# и 256 поатомным свойствами и сверточными фильтрами

n_atom_basis = 256

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=300, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=9,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy')
pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)
    ]
)

output_energy = spk.task.ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MSE": torchmetrics.MeanSquaredError()
    }
)

output_forces = spk.task.ModelOutput(
    name='forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MSE": torchmetrics.MeanSquaredError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

logger = pl.loggers.TensorBoardLogger(save_dir='./forcetut')
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join('./forcetut', model_file_path),
        save_top_k=1,
        monitor="val_loss"
    )
]

print("Start train", flush = True)
trainer = pl.Trainer(
    # при наличии графического процессора нужно раскомментировать следующие две строки
    
    # accelerator="gpu",
    # devices=[0],
    
    log_every_n_steps=1,
    callbacks=callbacks,
    logger=logger,
    default_root_dir='./forcetut',
    max_epochs=500,
)
print("Finish train", flush = True)
trainer.fit(task, datamodule=phosphorene_data)
