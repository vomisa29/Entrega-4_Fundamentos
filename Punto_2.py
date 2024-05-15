import matplotlib.pyplot as plt
import numpy as np

#------------------------------- Definición de funciones ---------------------------------------

def crear_H(n,m,X):
  H=[]
  for i in range(0,m):
    fila=[1]
    for j in range(0,n):
      fila.append(X[i]**(j+1))
    H.append(fila)

  H = np.array(H)
  return H

def sol_analitica(H,Y):
  Ht= np.transpose(H)
  return np.linalg.inv(Ht@H)@Ht@Y

def hallar_b_con_X(n,m,X):
  H=crear_H(n,m,X)
  b=sol_analitica(H,Y)
  return b, H

def dar_Y_a_partir_b(b,var):
  rta=[]
  for elem in var:
    num=0
    for i in range(0,len(b)):
      num+=b[i]*(elem**i)
    rta.append(num)
  return rta

#------------------------------- Inicio del programa -----------------------------------

data = np.loadtxt("data/datosToy.txt")

X=[]
Y=[]
for elem in data:
    x=elem[0]
    y=elem[1]
    X.append(x)
    Y.append(y)

b_1, H_1 = hallar_b_con_X(1,data.shape[0],X)
b_2, H_2 = hallar_b_con_X(2,data.shape[0],X)
b_3, H_3 = hallar_b_con_X(3,data.shape[0],X)

fig, ax = plt.subplots(1,1, layout="constrained",figsize=(10, 4))

ax.scatter(X,Y,color="orange", marker=".")
ax.set(xlabel="X", ylabel="Y", title="Grafico 1")

r1 =dar_Y_a_partir_b(b_1,X)
ax.plot(X,r1,color="blue",label="Polinomio de grado 1",linewidth=3)

r2 =dar_Y_a_partir_b(b_2,X)
ax.plot(X,r2,color="green",label="Polinomio de grado 2",linewidth=3)

r3 =dar_Y_a_partir_b(b_3,X)
ax.plot(X,r3,color="red",label="Polinomio de grado 3",linewidth=3)

ax.legend()
plt.show()