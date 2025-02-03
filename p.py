

import time
import random

def generar_matriz(filas, columnas, min_val=0, max_val=10):
    return [[random.randint(min_val, max_val) for _ in range(columnas)] for _ in range(filas)]

# Ejemplo: Matriz 1000x1000 con valores entre 0 y 100
a = generar_matriz(1000, 1000, 0, 100)
b = generar_matriz(1000, 1000, 0, 100)

start = time.time()
c = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(b))]
end = time.time()
print("Time taken:", end - start)


start = time.time()
d = list(map(lambda fila: list(map(lambda x: x[0] + x[1], zip(*fila))), zip(a, b)))
end = time.time()
print("Time taken:", end - start)