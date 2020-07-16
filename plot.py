import numpy as np
import matplotlib.pyplot as plt
inert=np.loadtxt("./DQN_inert")
print(np.shape(inert))
aver_inert=np.average(inert,axis=0)
plt.plot(range(len(aver_inert)),aver_inert)
raw=np.loadtxt("./DQN_raw")
aver_raw=np.average(raw,axis=0)
print(np.shape(raw))

plt.plot(range(len(aver_raw)),aver_raw)
plt.savefig("./copare.png")