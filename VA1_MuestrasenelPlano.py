################################################################################
# Muestras en el Plano
# Autor: Elías González Nieto
# Afil : Facultad de Ciencias - UNAM
# Curso : Probabilidad II
################################################################################

################################################################################
# Librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import math
from mpl_toolkits.mplot3d import Axes3D
################################################################################

################################################################################
# Entradas independientes

N = 1000

# Ambas muestras
u1 = np.random.uniform(0,1,N)
u2 = np.random.uniform(0,1,N)

# Graficamos las muestras
plt.figure()
plt.scatter(u1,u2, color = 'indigo',  alpha = 0.6)
plt.show()

# Ejemplo 2
# Ambas muestras
u3 = np.random.binomial(1, 0.5, N)
u4 = np.random.uniform(0,1,N)

# Graficamos las muestras
plt.figure()
plt.scatter(u3,u4, color = 'indigo', alpha = 0.6)
plt.show()

# Ejemplo 3
# Tamaño de la muestra
N = 100

# Ambas muestras
u5 = np.random.normal(0,1,N)
u6 = np.random.binomial(50, 0.5, N)

# Graficamos las muestras
plt.figure()
plt.scatter(u5,u6, color = 'indigo', alpha = 0.6)
plt.show()

# Función que lo junta todo
def graficar_vector(N, dist1, dist2, args1=(), args2=()):
    # Generamos muestras
    u1 = dist1(*args1, N)
    u2 = dist2(*args2, N)
    # Graficamos ambas muestras
    plt.figure(figsize=(6, 6))
    plt.scatter(u1, u2, color='indigo', alpha=0.6)
    plt.xlabel('Dist 1')
    plt.ylabel('Dist 2')
    plt.title('Muestra conjunta de dos distribuciones')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Ejemplos
graficar_vector(1000, np.random.normal, np.random.normal, args1=(0, 1), args2=(0, 1))
graficar_vector(1000, np.random.exponential, np.random.uniform, args1=(3,), args2=(0, 1)) # Si la distribución tiene un solo parámetro se pone (lambda, )
graficar_vector(1000, np.random.poisson, np.random.geometric, args1=(3,), args2=(0.01,)) # Si la distribución tiene un solo parámetro se pone (lambda,
graficar_vector(1000, np.random.exponential, np.random.geometric, args1=(0.5,), args2=(0.1,))

############################
# Con entradas no independientes

# Ambas muestras
muestra1 = np.random.normal(0, 1, 1000)
muestra2 = muestra1 + np.ones(1000)
# Gráfico
plt.figure(figsize=(6, 6))
plt.scatter(muestra1, muestra2, color='indigo', alpha=0.6)
plt.xlabel('Muestra 1')
plt.ylabel('Muestra 2')
plt.title('Muestra conjunta de dos distribuciones')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Función para graficar dos muestras
def dos_muestras(muestra1, muestra2):
  # Graficamos ambas muestras
  plt.figure(figsize=(6, 6))
  plt.scatter(muestra1, muestra2, color='indigo', alpha=0.6)
  plt.xlabel('Muestra 1')
  plt.ylabel('Muestra 2')
  plt.title('Muestra conjunta de dos distribuciones')
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.show()
  return None

# Ejemplos
muestra2 = np.random.normal(0, 1, 1000)
muestra3 = muestra2 + np.random.uniform(-1, 1, 1000)
dos_muestras(muestra3, muestra2)

muestra4 = np.random.exponential(0.9, 1000)
muestra5 = np.cumsum(muestra4)
dos_muestras(muestra5, muestra4)

#############################
# Vectores en el espacio

# Función para graficar una muestra de vectores aleatorios en R3
def graficar_vector_3D(N, distx, disty, distz, argsx=(), argsy=(), argsz=()):
  # Generamos ambas muestras
  x = distx(*argsx, N)
  y = disty(*argsy, N)
  z = distz(*argsz, N)

  # Gráfico 3D
  fig = plt.figure(figsize=(7, 7))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, color='indigo', alpha=0.6, s=30)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Muestra conjunta de tres distribuciones')
  ax.grid(True, linestyle='--', alpha=0.5)
  plt.show()
  return None

# Ejemplos
graficar_vector_3D(1000, np.random.uniform, np.random.uniform, np.random.uniform, argsx=(-1, 1), argsy=(-1, 1), argsz=(-1, 1))
graficar_vector_3D(1000, np.random.normal, np.random.normal, np.random.normal, argsx=(0, 1), argsy=(0, 1), argsz=(0, 1))
graficar_vector_3D(1000, np.random.exponential, np.random.binomial, np.random.geometric, argsx=(0.5,), argsy=(100, 0.3), argsz=(0.2,))
