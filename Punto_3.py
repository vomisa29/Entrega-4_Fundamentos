import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("data/datosNN.txt")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

fig, ax = plt.subplots(1,1, layout="constrained",figsize=(10, 4))

w=np.array([-2,1])
b=-1

x = np.linspace(-1.36,0.36,100)
f = (-w[0]/w[1])*x + (-b/w[1])
plt.plot(x, f,color="green",label="Regresión Logistica",linewidth=3)

rojo_list_X=[]
rojo_list_Y=[]
azul_list_X=[]
azul_list_Y=[]

for elem in data_scaled:
    clasificador = w@elem+b
    if clasificador > 0:
        rojo_list_X.append(elem[0])
        rojo_list_Y.append(elem[1])
    else:
        azul_list_X.append(elem[0])
        azul_list_Y.append(elem[1])

ax.scatter(rojo_list_X,rojo_list_Y,color="red",marker=".",label="Clase 1")
ax.scatter(azul_list_X,azul_list_Y,color="blue",marker=".", label="Clase 0")
ax.legend(framealpha=1)

plt.show()