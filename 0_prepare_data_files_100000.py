# Формируем файлы Ph_pos.npy, Ph_for.npy, Ph_energy.npy
# которые в дальнейшем будем использовать для формирования датасета

import os
import numpy as np
import time
# В папке out_cp есть файлы Ph.pos и Ph.for.
# Формат этих файлов одинаковый: в одном хранятся позиции для всех 16 атомов,
# в другом - силы, дествующие на каждый атом:
# 700001 84.66107236
#    -0.79928119349662E-02    -0.88981686708501E-02    -0.24683389851526E-02
#     0.18990419809663E-01    -0.97411267244073E-02    -0.13519303240801E-01
#    ........................................................................
#     -0.30898007230874E-02    -0.37616972155489E-02    -0.10024453783450E-02
# Нам нужно выбрость строки-заголовки (700001 84.66107236).
# Эти данные сохраняем во временном файле f_without_spaces.
# Затем при помощи метода genfromtxt() данные из этого временного файла загружаются в
# плоский NumPy-массив, после чего переформатируются в список списков, каждый из элементов
# которого содержит 16 записей по 3 вещественных числа (для 16 атомов модели и 3 компонент
# координаты/силы.
# Временные файлы удаляются, результат сохраняется в файле, имя которого передается
# в качестве второго параметра в функцию prepare_file_3()
def prepare_file_3(fname_in, fname_in_npy):
    fname_without_spaces = fname_in+'_wos'
    f_in  = open(fname_in, 'r')
    f_without_spaces = open(fname_without_spaces, 'w')
    f_lines = f_in.readlines()

    print('Delete spaces ... ')
    start_time = time.time()
    i = 0
    for item in f_lines:
        if i%17 != 0:
            f_without_spaces.write(' '.join(item.split())+'\n')
        i = i + 1
    f_without_spaces.close()
    print("--- Done %s seconds ---" % (time.time() - start_time))

    print('Convert to numpy format... ')
    start_time = time.time()
    data_np = np.genfromtxt(fname_without_spaces, delimiter=" ", usemask=True).flatten()
    data_np = data_np.reshape(len(data_np)//48,16,3)
    os.remove(fname_without_spaces)
    print("--- Done %s seconds ---" % (time.time() - start_time))
    
    print('Save to .npy format... ')
    start_time = time.time()
    np.save(fname_in_npy, data_np.data)
    print("--- Done %s seconds ---\n" % (time.time() - start_time))
    

prepare_file_3(r'./out_cp/Ph.pos', r'./out_cp/Ph_pos.npy')
prepare_file_3(r'./out_cp/Ph.for', r'./out_cp/Ph_for.npy')

# файл out_cp/Ph.evp имеет следующий формат:
# nfi     time(ps)      ekinc         Tcell(K)      Tion(K)            etot                enthal
# 700001  8.466107E+01  1.419833E-03  0.000000E+00  2.695491E+02       -112.15392184       -112.15392184
# econs               econt          Volume             Pressure(GPa)
# -112.13343515       -112.13635105  8.531480E+03       1.75963
# Здесь:
# ekink  - фиктивная кинетическая энергия электронов (K_electrons)
# enthal - энтальпия (E_DFT + PV)
# etot   - DFT(потенциальная) энергия системы (E_DFT+PV)
# econs  - физически значимая константа движения (E_DFT+K_nuclei) в пределе нулевой массы электронов
# econt  - константа движения лагранжиана

# Нам для обучения сети нужен параметр etot, поэтому мы берем 5 элемент каждой строки
evp = np.loadtxt(r'./out_cp/Ph.evp')
energy = evp[:, 5]
energy = energy.reshape(len(energy),1)
np.save(r'./out_cp/Ph_energy.npy',energy.data)
energy = np.load(r'./out_cp/Ph_energy.npy')
print(len(energy))