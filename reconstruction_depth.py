import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/ha618/Desktop/ICL-NUIM/Script/depth.dat'

with open(PATH, 'r') as f:
    n = 0
    for line in f:
        n+=1
        elements = line.split(',')
        LINES = []
        for element in elements:
            if element == '\n':
                continue
            else:
                LINES.append(element)

        value = np.asarray(LINES, dtype='f')
        value = np.expand_dims(value, axis=0)
        try:
            values = np.concatenate([value, values], axis=0)
        except:
            values = value

plt.pcolor(values)
plt.show()
