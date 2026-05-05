import random
import math
import matplotlib.pyplot as plt
import numpy as np

## Definição das Variáveis de Decisão

funcao_objetivo="sphere"

## Definição das Variáveis da Particula
n_execucoes=30
n_particulas=30
n_dimensoes=30
n_iteracoes=30
var_min=-100
var_max=100

## Definição da Inércia (Linear ou Constante)
tipo_inercia="linear"

## SE CONSTANTE=
w_constante=0.7

## SE LINEAR=
w_max=0.9
w_min=0.4

## Componente Cognitivo
c1=2.05

## Componente Social
c2=2.05

## Velocidade máxima (por partícula)
v_max=0.2

## Tipo de cooperação (Global ou Local)
tipo="global"

## Definição da função objetivo

def sphere(x):
    """
    Sphere Function
    Mínimo global: f(0, ..., 0) = 0
    """
    return sum(xi**2 for xi in x)


def rastrigin(x):
    """
    Rastrigin Function
    Mínimo global: f(0, ..., 0) = 0
    """
    n = len(x)
    return 10*n + sum(xi**2 - 10*math.cos(2 * math.pi * xi) for xi in x)

def rosenbrock(x):
    """
    Rosenbrock Function
    Mínimo global: f(1, ..., 1) = 0
    """
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1))


def avaliar(x):
  if funcao_objetivo == "sphere":
    return sphere(x)
  if funcao_objetivo == "rastrigin":
    return rastrigin(x)
  if funcao_objetivo == "rosenbrock":
    return rosenbrock(x)
  else:
    return TypeError("Função fora do escopo")
  
def executar_pso():

    historico_melhor = []
    historico_media = []

    posicoes = [
        [random.uniform(var_min, var_max) for _ in range(n_dimensoes)]
        for _ in range(n_particulas)
    ]

    print("\nPopulação inicial:")
    for i, p in enumerate(posicoes):
        print(f"Partícula {i}: {p[:5]}...")

    velocidades = [
        [random.uniform(-v_max, v_max) for _ in range(n_dimensoes)]
        for _ in range(n_particulas)
    ]

    # pbest
    pbest = [p[:] for p in posicoes]
    pbest_val = [avaliar(p) for p in posicoes]

    # gbest
    gbest_index = pbest_val.index(min(pbest_val))
    gbest = pbest[gbest_index][:]
    gbest_val = pbest_val[gbest_index]

    # Loop principal
    for t in range(n_iteracoes):

        if tipo_inercia == "linear":
          w = w_max - (w_max - w_min) * (t / n_iteracoes)
        elif tipo_inercia == "constante":
          w = w_constante
        else:
          raise ValueError("Tipo de inércia inválido")

        valores = []

        for i in range(n_particulas):

            if tipo == "local":
                vizinhos = [(i-1)%n_particulas, i, (i+1)%n_particulas]
                melhor_vizinho = min(vizinhos, key=lambda idx: pbest_val[idx])
                best_social = pbest[melhor_vizinho]
            else:
                best_social = gbest

            for j in range(n_dimensoes):
                r1 = random.random()
                r2 = random.random()

                velocidades[i][j] = (
                    w * velocidades[i][j]
                    + c1 * r1 * (pbest[i][j] - posicoes[i][j])
                    + c2 * r2 * (best_social[j] - posicoes[i][j])
                )

                velocidades[i][j] = max(-v_max, min(v_max, velocidades[i][j]))

                posicoes[i][j] += velocidades[i][j]
                posicoes[i][j] = max(var_min, min(var_max, posicoes[i][j]))

            valor = avaliar(posicoes[i])
            valores.append(valor)

            if valor < pbest_val[i]:
                pbest[i] = posicoes[i][:]
                pbest_val[i] = valor

                if valor < gbest_val:
                    gbest = posicoes[i][:]
                    gbest_val = valor

        # salvar histórico
        historico_melhor.append(gbest_val)
        historico_media.append(sum(valores)/len(valores))

    return gbest, gbest_val, historico_melhor, historico_media

def experimento():

    resultados = []
    historico_global_melhor = []
    historico_global_media = []

    for _ in range(n_execucoes):
        _, melhor_valor, hist_melhor, hist_media = executar_pso()

        resultados.append(melhor_valor)
        historico_global_melhor.append(hist_melhor)
        historico_global_media.append(hist_media)

    print("\nResultados:")
    print(f"Melhor: {min(resultados)}")
    print(f"Média: {sum(resultados)/len(resultados)}")
    print(f"Pior: {max(resultados)}")

    return resultados, historico_global_melhor, historico_global_media

def plotar_resultados(resultados, historico_global_melhor, historico_global_media):

    melhor_array = np.array(historico_global_melhor)
    media_array = np.array(historico_global_media)

    melhor_medio = np.mean(melhor_array, axis=0)
    media_media = np.mean(media_array, axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Convergência média
    axs[0].plot(melhor_medio, label="Melhor global")
    axs[0].plot(media_media, label="Média população")
    axs[0].set_title("Convergência do PSO")
    axs[0].set_xlabel("Iteração")
    axs[0].set_ylabel("Fitness")
    axs[0].legend()

    # Boxplot
    axs[1].boxplot(resultados)
    axs[1].set_title("Distribuição dos Resultados")
    axs[1].set_ylabel("Fitness")

    # Execução 1
    axs[2].plot(historico_global_melhor[0], label="Melhor")
    axs[2].plot(historico_global_media[0], label="Média")
    axs[2].set_title("Execução 1")
    axs[2].set_xlabel("Iteração")
    axs[2].set_ylabel("Fitness")
    axs[2].legend()

    # ✅ Espaçamento automático entre gráficos
    plt.tight_layout()

    plt.show()

resultados, hist_melhor, hist_media = experimento()

plotar_resultados(resultados, hist_melhor, hist_media)