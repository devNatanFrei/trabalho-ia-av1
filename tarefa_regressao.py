import numpy as np
import matplotlib.pyplot as plt

# Carregando os dados
data = np.loadtxt('atividade_enzimatica.csv', delimiter=',')


temp = data[:, 0]
ph = data[:, 1]
ativ_enz = data[:, 2]


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(temp, ativ_enz, color='red', marker='x')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('Temperatura x Atividade Enzimática')

plt.subplot(1, 2, 2)
plt.scatter(ph, ativ_enz, color='blue', marker='x')
plt.xlabel('pH')
plt.ylabel('Atividade Enzimática (U/mg)')
plt.title('pH x Atividade Enzimática')

plt.show()


X = data[:, :2]  
y = ativ_enz     


N = X.shape[0]
X = np.hstack((np.ones((N, 1)), X))


beta_mqo = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"MQO Tradicional (OLS): {beta_mqo}")


lambdas = [0.25, 0.5, 0.75, 1]  
betas = {}

for lambda_ in lambdas:
    I = np.eye(X.shape[1])  
    beta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
    betas[lambda_] = beta


print("Estimativas do vetor β para diferentes valores de λ:")
for lambda_, beta in betas.items():
    print(f"λ = {lambda_}: {beta}")


media_y = np.mean(y)
print(f"Média dos valores observáveis: {media_y}")

    

def monte_carlo_validation(X, Y, model_func, R=500, test_size=0.2):
    errors = []
    nsamples = len(Y)
    
    for _ in range(R):
        indices = np.random.permutation(nsamples)
        X_shuffled, Y_shuffled = X[indices], Y[indices]

        split = int(nsamples * (1 - test_size))
        X_train, Y_train = X_shuffled[:split], Y_shuffled[:split]
        X_test, Y_test = X_shuffled[split:], Y_shuffled[split:]


        beta = model_func(X_train, Y_train)
        Y_pred = X_test @ beta
        
      
        rss = np.sum((Y_pred - Y_test) ** 2)
        errors.append(rss)
    
    return errors


def modelo_mqo(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def modelo_tikhonov(X, y, lambda_):
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y

def media(X, y):
    return np.mean(y) * np.ones(X.shape[1])

results = {
    "MQO": monte_carlo_validation(X, y, modelo_mqo),
    "Media": monte_carlo_validation(X, y, media),
}

for lambda_ in lambdas:
    def modelo_lambda(X, y):
        return modelo_tikhonov(X, y, lambda_)
    
    results[f"Tikhonov λ={lambda_}"] = monte_carlo_validation(X, y, modelo_lambda)
    
statics = {}
for name, errors in results.items():
    statics[name] = [np.mean(errors), np.std(errors), np.min(errors), np.max(errors)]
    
print("\nTabela de Estatísticas do RSS")
print("=" * 60)
print(f"{'Modelo':<20}{'Média':<12}{'Desvio Padrão':<15}{'Mínimo':<12}{'Máximo':<12}")
print("=" * 60)
for nome, stats in statics.items():
    print(f"{nome:<20}{stats[0]:<12.4f}{stats[1]:<15.4f}{stats[2]:<12.4f}{stats[3]:<12.4f}")
