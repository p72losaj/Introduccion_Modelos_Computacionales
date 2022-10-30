# Biblioteca matplotlib
import matplotlib.pyplot as plt

# Fichero seed_0.txt
f1 = open("seed_0.txt", "r")
# Fichero seed_1.txt
f2 = open("seed_1.txt", "r")
# Fichero seed_2.txt
f3 = open("seed_2.txt", "r")
# fichero seed_3.txt
f4 = open("seed_3.txt", "r")
# fichero seed_4.txt
f5 = open("seed_4.txt", "r")

# Semilla 1
seed1 = []; seed1_TrainError = []; seed1_TrainCCR = []; seed1_TestCCR=[]
# Semilla 2
seed2 = []; seed2_TrainError = []; seed2_TrainCCR = []; seed2_TestCCR=[]
# Semilla 3
seed3 = []; seed3_TrainError = []; seed3_TrainCCR = []; seed3_TestCCR=[]
# Semilla 4
seed4 = []; seed4_TrainError = []; seed4_TrainCCR = []; seed4_TestCCR=[]
# Semilla 5
seed5 = []; seed5_TrainError = []; seed5_TrainCCR = []; seed5_TestCCR=[]


# Abrimos el fichero f1
for line in f1:
    # Si la linea contiene Epoch -> La ignoramos
    if "Epoch" in line:
        continue
    # Si la linea contiene Confusion -> Finalizamos la lectura del fichero
    if "Confusion" in line:
        break
    # Separamos la linea por espacios
    linea = line.split()
    # La linea no puede estar vacia
    if len(linea) > 0:
        # Añadimos los valores a las listas
        seed1.append(int(linea[0]))
        seed1_TrainError.append(float(linea[1]))
        seed1_TrainCCR.append(float(linea[2]))
        seed1_TestCCR.append(float(linea[3]))

# Abrimos el fichero f2
for line in f2:
    # Si la linea contiene Epoch -> La ignoramos
    if "Epoch" in line:
        continue
    # Si la linea contiene Confusion -> Finalizamos la lectura del fichero
    if "Confusion" in line:
        break
    # Separamos la linea por espacios
    linea = line.split()
    # La linea no puede estar vacia
    if len(linea) > 0:
        # Añadimos los valores a las listas
        seed2.append(int(linea[0]))
        seed2_TrainError.append(float(linea[1]))
        seed2_TrainCCR.append(float(linea[2]))
        seed2_TestCCR.append(float(linea[3]))

# Abrimos el fichero f3
for line in f3:
    # Si la linea contiene Epoch -> La ignoramos
    if "Epoch" in line:
        continue
    # Si la linea contiene Confusion -> Finalizamos la lectura del fichero
    if "Confusion" in line:
        break
    # Separamos la linea por espacios
    linea = line.split()
    # La linea no puede estar vacia
    if len(linea) > 0:
        # Añadimos los valores a las listas
        seed3.append(int(linea[0]))
        seed3_TrainError.append(float(linea[1]))
        seed3_TrainCCR.append(float(linea[2]))
        seed3_TestCCR.append(float(linea[3]))

# Abrimos el fichero f4
for line in f4:
    # Si la linea contiene Epoch -> La ignoramos
    if "Epoch" in line:
        continue
    # Si la linea contiene Confusion -> Finalizamos la lectura del fichero
    if "Confusion" in line:
        break
    # Separamos la linea por espacios
    linea = line.split()
    # La linea no puede estar vacia
    if len(linea) > 0:
        # Añadimos los valores a las listas
        seed4.append(int(linea[0]))
        seed4_TrainError.append(float(linea[1]))
        seed4_TrainCCR.append(float(linea[2]))
        seed4_TestCCR.append(float(linea[3]))

# Abrimos el fichero f5
for line in f5:
    # Si la linea contiene Epoch -> La ignoramos
    if "Epoch" in line:
        continue
    # Si la linea contiene Confusion -> Finalizamos la lectura del fichero
    if "Confusion" in line:
        break
    # Separamos la linea por espacios
    linea = line.split()
    # La linea no puede estar vacia
    if len(linea) > 0:
        # Añadimos los valores a las listas
        seed5.append(int(linea[0]))
        seed5_TrainError.append(float(linea[1]))
        seed5_TrainCCR.append(float(linea[2]))
        seed5_TestCCR.append(float(linea[3]))


# Cerramos los ficheros
f1.close(); f2.close(); f3.close(); f4.close(); f5.close()

# Generamos el grafico de la semilla 1
plt.plot(seed1, seed1_TrainCCR, label="Seed1_TrainCCR")
plt.plot(seed1, seed1_TestCCR, label="Seed1_TestCCR")
# Eje x -> Numero de iteraciones
plt.xlabel("Numero de iteraciones")
# Eje y -> Valor de CCR
plt.ylabel("CCR_Train/CCR_Test")
plt.legend()
plt.savefig("Seed1_CCR.png")
plt.clf()

# Generamos el grafico de la semilla 2
plt.plot(seed2, seed2_TrainCCR, label="Seed2_TrainCCR")
plt.plot(seed2, seed2_TestCCR, label="Seed2_TestCCR")
# Eje x -> Numero de iteraciones
plt.xlabel("Numero de iteraciones")
# Eje y -> Valor de CCR
plt.ylabel("CCR_Train/CCR_Test")
plt.legend()
plt.savefig("Seed2_CCR.png")
plt.clf()

# Generamos el grafico de la semilla 3
plt.plot(seed3, seed3_TrainCCR, label="Seed3_TrainCCR")
plt.plot(seed3, seed3_TestCCR, label="Seed3_TestCCR")
# Eje x -> Numero de iteraciones
plt.xlabel("Numero de iteraciones")
# Eje y -> Valor de CCR
plt.ylabel("CCR_Train/CCR_Test")
plt.legend()
plt.savefig("Seed3_CCR.png")
plt.clf()

# Generamos el grafico de la semilla 4
plt.plot(seed4, seed4_TrainCCR, label="Seed4_TrainCCR")
plt.plot(seed4, seed4_TestCCR, label="Seed4_TestCCR")
# Eje x -> Numero de iteraciones
plt.xlabel("Numero de iteraciones")
# Eje y -> Valor de CCR
plt.ylabel("CCR_Train/CCR_Test")
plt.legend()
plt.savefig("Seed4_CCR.png")
plt.clf()

# Generamos el grafico de la semilla 5
plt.plot(seed5, seed5_TrainCCR, label="Seed5_TrainCCR")
plt.plot(seed5, seed5_TestCCR, label="Seed5_TestCCR")
# Eje x -> Numero de iteraciones
plt.xlabel("Numero de iteraciones")
# Eje y -> Valor de CCR
plt.ylabel("CCR_Train/CCR_Test")
plt.legend()
plt.savefig("Seed5_CCR.png")
plt.clf()




