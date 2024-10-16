import time
from numpy import linalg as LA
import numpy as np
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

# Вычисляем модули сил, действующих на каждый из атомов, для всех записей набора данных, используя стандартную
# функцию пакета numpy.linalg
# Функция возвращает список значений модулей сил, действующих на каждый атом:
# [[0.62801302 0.6307762  0.63622954 ... 0.36716072 0.37727941 0.38339295]
# [1.29915505 1.29822888 1.29273672 ... 1.69248764 1.70490134 1.71561141]
# ........................................................................
# [1.42605078 1.43417797 1.43716007 ... 1.41029354 1.40126231 1.39578814]]
# 16 списков (по числу атомов), в каждом списке - значения модуля силы,
# действующей на каждый из атомов, вычисленной на данном кадре выборки
def get_abs_force_value_by_atoms(list_forces):
    LLL = len(list_forces[0])
    temp = np.zeros(LLL*len(list_forces))
    temp = temp.reshape(len(list_forces),LLL)
    start_time = time.time()
    for i in range(len(list_forces)):
        for j in range(LLL):
            temp[i][j] = LA.norm(list_forces[i][j])
    temp = temp.transpose()
    #for i in range(len(temp)):
    #    temp[i] = np.sort(temp[i])
    print("--- Done %s seconds ---\n" % (time.time() - start_time))
    return(temp)

# Силы
f_data = np.load(r'./out_cp/Ph_for.npy')*Hartree/Bohr # эВ/Ангстрем
print(f_data[0])

num_train = 80000
num_val=15000
num_test=5000
lst_initial_data_all_by_atoms = get_abs_force_value_by_atoms(f_data[num_train+num_val:])
print(lst_initial_data_all_by_atoms)
with open('lst_initial_data_all_by_atoms.npy', 'wb') as f:
    np.save(f, lst_initial_data_all_by_atoms)
