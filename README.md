# SchNetPhosphorene
Supplementary materials for article ... 

Установка и настройка пакета SchNet осуществляется согласно руководству
https://schnetpack.readthedocs.io/
Сам пакет schnetpack устанавливается в окружение schnetenv среды conda.
В этом окружении настраиваются все пакеты, необходимые для обучения сети.
Для использования обученной сети для вычисления потенциала межатомного взаимодействия
в ходе классической молекулярной динамики настраивается окружение spk_lammps.

1) В папке /out_cp должны лежать исходные данные для обучения - результаты моделирования черного фосфорена методом квантовой
молекулярной динамики Кара-Парринелло. Выборка содержит 100000 записей. Моделируется система из 16 атомов.
Эти данные размещены на Яндекс-диске, ссылка и перечень файлов - в текстовом файле.

2) Файл 0_prepare_data_files_100000.py содержит код для преобразования данных из файлов
    Ph.pos (координаты атомов),
    Ph.for (силы, действующие на каждый из атомов),
    Ph.evp (энергия системы атомов)
в формат NumPy (файлы Ph_pos.npy, Ph_for.npy, Ph_energy.npy). Эти файлы будут использоваться в дальнейшем
для формирования датасета.
Для запуска файла нужно активировать окружение schnetenv:
    conda activate schnetenv
Собственно запуск файла на исполнение осуществляется командой
    python 0_prepare_data_files_100000.py
   
3) Файл 1_prepare_database_100000.py с использованием данных из файлов Ph_pos.npy, Ph_for.npy, Ph_energy.npy
   базу данных dataset_100000_pbc.db, которая сохраняется в папке /forcetut.
   
4) 

