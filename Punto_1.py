import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

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

def lista_estimacion(b,X):
  est=[]
  for k in range(0,len(X)):
    est.append(estimacion(b,X,k))
  return est

def estimacion(b,X,k):
  rta=0
  for i in range(0,len(b)):
    rta+=b[i]*(X[k]**i)
  return rta

def RMSE(X,Y):
  sumatoria=0
  for k in range(0,len(Y)):
    sumatoria+= (Y[k] - X[k])**2
  return math.sqrt(sumatoria/len(Y))

#------------------------------- Inicio del programa -----------------------------------

data = np.loadtxt("data/greenhouse.txt")

X=[]
Y=[]
for elem in data:
    x=elem[0]
    y=elem[1]
    X.append(x)
    Y.append(y)

b, H = hallar_b_con_X(1,data.shape[0],X)

new_data = np.loadtxt("data/datosNuevos.txt")

new_X=[]
new_Y=[]
for elem in new_data:
    x=elem[0]
    y=elem[1]
    new_X.append(x)
    new_Y.append(y)

new_b, new_H = hallar_b_con_X(1,data.shape[0],new_X)


fig, ax = plt.subplots(1,3, layout="constrained",figsize=(15, 4))

#----------------------------- Primer Grafico -------------------------------------

ax[0].scatter(X,Y,color="red", marker="o")
ax[0].set(xlabel="X", ylabel="Y", title="Grafico 1")

recta_regresion =dar_Y_a_partir_b(b,X)
ax[0].plot(X,recta_regresion,color="blue",label="Regresión Lineal - Minimos Cuadrados")
ax[0].legend()

#----------------------------- Segundo Grafico -------------------------------------

ax[1].set(xlabel="X", ylabel="Y", title="Grafico 2")
predicciones = lista_estimacion(b,X)
error_prom = RMSE(predicciones,Y)
rojo=True
verde=True


for i in range(0,len(predicciones)):

    error_actual = abs(Y[i] - predicciones[i])

    if error_actual<error_prom:
        if rojo:
            ax[1].scatter(X[i],Y[i],color="red", marker="o",label="Datos con error por debajo del promedio")
            rojo=False
        else:
            ax[1].scatter(X[i],Y[i],color="red", marker="o")
    else:
        if verde:
            ax[1].scatter(X[i],Y[i],color="green", marker="o",label="Datos con error mayor al promedio")
            verde=False
        else:
            ax[1].scatter(X[i],Y[i],color="green", marker="o")
       
ax[1].plot(X,recta_regresion,color="blue")
ax[1].legend()

#----------------------------- Tercer Grafico -------------------------------------

ax[2].set(xlabel="X", ylabel="Y", title="Grafico 3")

recta_regresion =dar_Y_a_partir_b(new_b,new_X)
predicciones = lista_estimacion(new_b,new_X)
error_prom = RMSE(predicciones,new_Y)
rojo=True
verde=True


for i in range(0,len(predicciones)):

    error_actual = abs(new_Y[i] - predicciones[i])

    if error_actual<error_prom:
        if rojo:
            ax[2].scatter(new_X[i],new_Y[i],color="red", marker="o",label="Datos con error por debajo del promedio")
            rojo=False
        else:
            ax[2].scatter(new_X[i],new_Y[i],color="red", marker="o")
    else:
        if verde:
            ax[2].scatter(new_X[i],new_Y[i],color="green", marker="o",label="Datos con error mayor al promedio")
            verde=False
        else:
            ax[2].scatter(new_X[i],new_Y[i],color="green", marker="o")
       
ax[2].plot(new_X,recta_regresion,color="blue")
ax[2].legend()

plt.show()