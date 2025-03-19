import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('atividade_enzimatica.csv', delimiter=',')
temp = data[:, 0]
ph = data[:, 1]
ativ_enz = data[:, 2]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(temp, ativ_enz, color='red')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('Temperatura x Atividade Enzimática')

plt.subplot(1, 2, 2)
plt.scatter(ph, ativ_enz, color='blue')
plt.xlabel('pH')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('pH x Atividade Enzimática')


plt.show()



