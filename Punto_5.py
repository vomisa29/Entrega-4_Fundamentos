import matplotlib.pyplot as plt
import numpy as np
import random

data = np.loadtxt("data/coordenadasPacientes.txt")

def centro_masa(conjunto):
    rta=np.array([0.0,0.0])
    for elem in conjunto:
        rta+=elem
    return rta/len(conjunto)

def K_means(data,k,numIteraciones):
    lista_k=[]
    elem_k={}
    for i in range(0,k):
        
        indice=random.randint(0,len(data))
        lista_k.append(data[indice])
        
        elem_k[i]=[]

    for j in range(0,numIteraciones):
        k_cercano(lista_k,elem_k)
        
        n=len(lista_k)
        lista_k=[]
        for i in range(0,n):
            lista=elem_k[i]
            lista_k.append(centro_masa(lista))
        
                
    return lista_k, elem_k

def k_cercano(lista_k,elem_k):
    for elem in data:
        min_dist=9999999
        k_menor_dist=0
        for i in range(0,len(lista_k)):
            coord_k=lista_k[i]
            dist_actual=np.linalg.norm(elem-coord_k)
            
            if dist_actual<min_dist:
                min_dist=dist_actual
                k_menor_dist=i
            
        elem_k[k_menor_dist].append(elem)


#----------------------------------- Inicio Programa ----------------------------------------------
print("Ingrese el número de iteraciones del algoritmo: ")
nIteraciones=int(input())
print("Calculando...")
lista_k, elem_k =K_means(data,4,100)

colores=["red","green","blue","orange"]

fig, ax = plt.subplots(1,1, layout="constrained",figsize=(10, 4))

for i in range(0,len(lista_k)):
    lista = elem_k[i]
    X=[]
    Y=[]
    for elem in lista:
        x=elem[0]
        y=elem[1]
        X.append(x)
        Y.append(y)
    ax.scatter(X,Y,color=colores[i],label="K " + str(i), marker=".")
    
print("Cálculo finalizado.")  
ax.legend()
plt.show()
