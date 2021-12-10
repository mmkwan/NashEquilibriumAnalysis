import pandas as pd
import numpy as np
from math import log2, e
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

f = open("CSVs/interior_point_times_rps.csv")
avg = []
sizes = []
for line in f:
    data = line.split(",")
    sizes.append(float(data[0]))
    total = 0
    for item in data[1:]:
        total += float(item)
    avg.append(total/10)

#df_uniform = pd.read_csv("CC.csv", header=None, names=["Size", "CC"])
AlgorithmName = 'Interior Point on generalized Rock-Paper-Scissors.'

y = avg
x = sizes

print(y)

p = plt.loglog(x, y, '.', markersize = 12,)

logx, logy = np.log(x), np.log(y)
m, b = np.polyfit(logx, logy, 1)
fit = np.poly1d((m,b))
expected_logy = fit(logx)

r2 = r2_score(logy, expected_logy)

sort_name = AlgorithmName
perm_name = ""

fit_p = plt.semilogx(x[::len(x)-1], (e ** expected_logy)[::len(y)-1], '--', base = 2,
                   label = f'{sort_name}: {m:0.5}logx + {b: .5}, r^2 = {r2: .5}',
                   markersize = 6, color = p[-1].get_color())
plt.legend( loc = 'upper left')



plt.show()