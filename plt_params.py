'''
plot the param comparison of the model parameters
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'Times New Roman'

x = np.arange(1,5)  # 4  models
# name              params (m) ,    GFLOPs
# Sun, et al [41],   34.29           13.097
# Zhou, et al [50],  34.00m ,        12.001
# SPA                8.52           0.004
y1 = x*34.29        # sun
y2 = 34.29 + (x-1)*8.52
y3 = x*34
y4 = 34 + (x-1)*8.52

plt.plot(x, y1, 'r', label='Sun, et al [41]')
plt.plot(x, y2, 'r--', label='Sun, et al [41] + SPA')
plt.plot(x, y3, 'b', label='Zhou, et al [50]')
plt.plot(x, y4, 'b--', label='Zhou, et al [50] + SPA')

plt.legend()
plt.xlabel('datasets')
plt.ylabel('# of Parameters (M)')
plt.tight_layout()

plt.savefig('imgs/SPA_params.pdf')
plt.xlim([1, 1.2])
plt.ylim([33.9, 40])
# plt.show()
plt.savefig('imgs/SPA_params_zm.pdf')
# x datasets
# y params(M)
plt.show()