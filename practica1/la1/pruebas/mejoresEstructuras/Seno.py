# Generacion del grafico de estudio del error de entrenamiento de la mejor arquitectura seno

import matplotlib.pyplot as plt
# Biblioteca para eliminar ficheros
import os
semilla1 = []; semilla2 = []; semilla3 = []; semilla4 = []; semilla5 = []
iteraciones = []

# Guardamos las 1000 iteraciones

for i in range(1,1001):
    iteraciones.append(i)

valor = 0

# Creamos un nuevo fichero para cada semilla

# Delimitador de los datos SEED k
with open('seno_2,64', 'r') as f:
    # Semilla 1
    for line in f:
        # Almacenamos los datos de cada SEED k en un fichero diferente para poder leerlos
        if "SEED" in line:
            valor = valor + 1
            if valor == 1:
                with open('semilla1', 'w') as f1:
                    f1.write(line)
                    # Escribimos en el fichero semilla1 todas las filas del fichero xor desde SEED 1 hasta NETWORK WEIGHTS
                    for line in f:
                        if "NETWORK WEIGHTS" in line:
                            break
                        f1.write(line)
            if valor == 2:
                with open('semilla2', 'w') as f2:
                    f2.write(line)
                    # Escribimos en el fichero semilla2 todas las filas del fichero xor desde SEED 2 hasta NETWORK WEIGHTS
                    for line in f:
                        if "NETWORK WEIGHTS" in line:
                            break
                        f2.write(line)
            if valor == 3:
                with open('semilla3', 'w') as f3:
                    f3.write(line)
                    # Escribimos en el fichero semilla3 todas las filas del fichero xor desde SEED 3 hasta NETWORK WEIGHTS
                    for line in f:
                        if "NETWORK WEIGHTS" in line:
                            break
                        f3.write(line)
            if valor == 4:
                with open('semilla4', 'w') as f4:
                    f4.write(line)
                    # Escribimos en el fichero semilla4 todas las filas del fichero xor desde SEED 4 hasta NETWORK WEIGHTS
                    for line in f:
                        if "NETWORK WEIGHTS" in line:
                            break
                        f4.write(line)
            if valor == 5:
                with open('semilla5', 'w') as f5:
                    f5.write(line)
                    # Escribimos en el fichero semilla5 todas las filas del fichero xor desde SEED 5 hasta NETWORK WEIGHTS
                    for line in f:
                        if "NETWORK WEIGHTS" in line:
                            break
                        f5.write(line)

# Leemos los datos de cada fichero y los almacenamos en una lista

with open('semilla1', 'r') as f1:
    for line in f1:
        if "SEED" in line:
            continue
        # Leemos las lineas que empiecen con "Iteration"
        if "Iteration" in line:
            # Leemos la linea hasta un :
                linea = line.split(":")
                # Almacenamos el valor a continuacion del : en la semilla 1
                semilla1.append(float(linea[1])) 

with open('semilla2', 'r') as f2:
    for line in f2:
        if "SEED" in line:
            continue
        if "Iteration" in line:
                linea = line.split(":")
                semilla2.append(float(linea[1]))

with open('semilla3', 'r') as f3:
    for line in f3:
        if "SEED" in line:
            continue
        if "Iteration" in line:
                linea = line.split(":")
                semilla3.append(float(linea[1]))

with open('semilla4', 'r') as f4:
    for line in f4:
        if "SEED" in line:
            continue
        if "Iteration" in line:
                linea = line.split(":")
                semilla4.append(float(linea[1]))

with open('semilla5', 'r') as f5:
    for line in f5:
        if "SEED" in line:
            continue
        if "Iteration" in line:
                linea = line.split(":")
                semilla5.append(float(linea[1]))


# Eliminamos los ficheros que no nos interesan
os.remove("semilla1"); os.remove("semilla2"); os.remove("semilla3"); os.remove("semilla4"); os.remove("semilla5")

# Creamos el grafico
plt.plot(iteraciones, semilla1, label = "Semilla 1")
plt.plot(iteraciones, semilla2, label = "Semilla 2")
plt.plot(iteraciones, semilla3, label = "Semilla 3")
plt.plot(iteraciones, semilla4, label = "Semilla 4")
plt.plot(iteraciones, semilla5, label = "Semilla 5")
plt.xlabel("Iteraciones")
plt.ylabel("Error de entrenamiento")
plt.title("Error de entrenamiento de la mejor arquitectura SENO")
plt.legend()
plt.savefig("seno_2,64.png")
