# Данные о координатах, силах и энергиях хранятся в файлах
# './out_cp/Ph_pos.npy'
# './out_cp/Ph_for.npy'
# './out_cp/Ph_energy.npy'

# Формируем базу данных
# './forcetut/dataset_100000_pbc.db'

# использует ASE database format
from schnetpack.data import ASEAtomsData
import os
import numpy as np
from ase import Atoms
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

# В руководстве https://www.quantum-espresso.org/Doc/user_guide_PDF/cp_user_guide.pdf
# написано, что энергия измеряется в единицах Хартри (для cp.x) и Ридберга (для pw.x)
# расстояния - в единицах первого боровского радиуса.
# В исходном коде https://wiki.fysik.dtu.dk/ase/_modules/ase/io/espresso.html ,
# преобразующем выходные файлы pw.x к формату ASE, видно, что силы измеряются в единицах Rydberg/Bohr.
# Делаем вывод, что, скорее всего, единицы Hartree/Bohr.

# Загрузка из датасета 100000
# Координаты
r_data=np.load(r'./out_cp/Ph_pos.npy')*Bohr
print(r_data[0])

# Силы
f_data = np.load(r'./out_cp/Ph_for.npy')*Hartree/Bohr
print(f_data[0])

# Энергии
en_data = np.load(r'./out_cp/Ph_energy.npy')*Hartree
print(en_data[0])

# Список зарядовых чисел атомов, входящих в исследуемую молекулу:
# всего 16 атомов фосфора (зарядовое число 15)
n_atoms = 16
numbers = 15*np.ones((n_atoms), dtype = int)

# учет периодичности - описание элементарной ячейки
lattice_scale = 8.86

a1 = np.array([0.993809735,   0.0,   0.0])
a2 = np.array([0.0,   0.742697495,   0.0])
a3 = np.array([0.0,   0.0,  2.4627])

cell = np.array([a1*lattice_scale, a2*lattice_scale, a3*lattice_scale])

# Создаем список атомов (тип ASE.Atoms) вида
# [{property_name1: property1_molecule1}, {property_name1: property1_molecule2}, ..]
atoms_list = []
property_list = []
for positions, energies, forces in zip(r_data, en_data, f_data):
    ats = Atoms(
        positions=positions,
        numbers=numbers,
        cell = cell,
        pbc = True
    )
    properties = {'energy': energies, 'forces': forces}
    property_list.append(properties)
    atoms_list.append(ats)
#with np.printoptions(precision=2, suppress=True):
#    print('Properties:', property_list[0])
#    print('Positions:', atoms_list[0].positions)
#    print('Numbers:', atoms_list[0].numbers)


# Создаем базу данных и загружаем в нее полученные из датасета значения
forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)
if os.path.exists('./forcetut/dataset_100000_pbc.db'):
    os.remove('./forcetut/dataset_100000_pbc.db')
new_dataset = ASEAtomsData.create(
    './forcetut/dataset_100000_pbc.db',
    distance_unit='Ang',
    property_unit_dict={'energy':'eV', 'forces':'eV/Ang'}
)
new_dataset.add_systems(property_list, atoms_list)

# выводим формат данных, хранящихся в датасете
print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

for p in new_dataset.available_properties:
    print('-', p)
print()

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)

print(example)
