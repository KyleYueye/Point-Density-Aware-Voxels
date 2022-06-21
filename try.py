path = '/media/disk/02drive/05yueye/code/PDV/output/wanji_models/pdv/default/eval/eval_all_default/default/log_eval_20220525-154546.log'

bus = [[[], []],
       [[], []]]
car = [[[], []],
       [[], []]]
bicycle = [[[], []],
            [[], []]]
pedestrain = [[[], []],
                [[], []]]

index = 0
with open(path, 'r') as f:
    for line in f.readlines():
        if line[0:2] == '3d':
            index += 1
            if index % 8 == 1:
                bus[0][0].append(eval(line[8:15]))
                bus[1][0].append(eval(line[21:]))
            elif index % 8 == 2:
                bus[0][1].append(eval(line[8:15]))
                bus[1][1].append(eval(line[21:]))
            elif index % 8 == 3:
                car[0][0].append(eval(line[8:15]))
                car[1][0].append(eval(line[21:]))
            elif index % 8 == 4:
                car[0][1].append(eval(line[8:15]))
                car[1][1].append(eval(line[21:]))
            elif index % 8 == 5:
                bicycle[0][0].append(eval(line[8:15]))
                bicycle[1][0].append(eval(line[21:]))
            elif index % 8 == 6:
                bicycle[0][1].append(eval(line[8:15]))
                bicycle[1][1].append(eval(line[21:]))
            elif index % 8 == 7:
                pedestrain[0][0].append(eval(line[8:14]))
                pedestrain[1][0].append(eval(line[20:]))
            elif index % 8 == 0:
                pedestrain[0][1].append(eval(line[8:14]))
                pedestrain[1][1].append(eval(line[20:]))

import numpy as np
import pandas as pd
bus = np.asarray(bus)
car = np.asarray(car)
bicycle = np.asarray(bicycle)
pedestrain = np.asarray(pedestrain)

for i in range(len(bus[0,0])):
    print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            bus[0,0,i], bus[0,1,i], bus[1,0,i], bus[1,1,i], 
            car[0,0,i], car[0,1,i], car[1,0,i], car[1,1,i], 
            bicycle[0,0,i], bicycle[0,1,i], bicycle[1,0,i], bicycle[1,1,i], 
            pedestrain[0,0,i], pedestrain[0,1,i], pedestrain[1,0,i], pedestrain[1,1,i], 
        ))
        
