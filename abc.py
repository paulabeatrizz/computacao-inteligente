import random
import math
import matplotlib.pyplot as plt
import numpy as np

## Definição das Variáveis de Decisão

funcao_objetivo="rastrigin"

## Definição das Váriaveis das Fontes de Alimentação
dimensoes=10
tamanho_populacao=20
ciclos=100
limite_abandono=20
limite_inferior=-5
limite_superior=5

n_execucoes=30

## Definição da função objetivo

def sphere(x):
    """
    Sphere Function
    Mínimo global: f(0,...,0) = 0
    """
    return sum(xi**2 for xi in x)


def rastrigin(x):
    """
    Rastrigin Function
    Mínimo global: f(0,...,0) = 0
    """
    n = len(x)

    return 10*n + sum(
        xi**2 - 10 * math.cos(2 * math.pi * xi)
        for xi in x
    )


def rosenbrock(x):
    """
    Rosenbrock Function
    Mínimo global: f(1,...,1) = 0
    """

    return sum(
        100 * (x[i+1] - x[i]**2)**2 +
        (x[i] - 1)**2
        for i in range(len(x)-1)
    )


def avaliar(x, funcao_objetivo):

    if funcao_objetivo == "sphere":
        return sphere(x)

    elif funcao_objetivo == "rastrigin":
        return rastrigin(x)

    elif funcao_objetivo == "rosenbrock":
        return rosenbrock(x)

    else:
        raise ValueError("Função objetivo inválida")

## Defição da Fonte de Alimento (Solução em potencial)

class FonteAlimento:

    def __init__(
        self,
        dimensoes,
        limite_inferior,
        limite_superior,
        funcao_objetivo
    ):

        # Inicialização aleatória
        self.posicao = np.random.uniform(
            limite_inferior,
            limite_superior,
            dimensoes
        )

        # Avaliar Fitness
        self.fitness = avaliar(
            self.posicao,
            funcao_objetivo
        )

        # Número de tentativas sem melhora
        self.tentativas = 0

def gerar_vizinho(
    fontes,
    indice,
    limite_inferior,
    limite_superior
):

    nova_posicao = np.copy(fontes[indice].posicao)

    dimensoes = len(nova_posicao)

    i = random.randint(0, dimensoes - 1)


    k = random.randint(0, len(fontes) - 1)

    while k == indice:
        k = random.randint(0, len(fontes) - 1)

    phi = random.uniform(-1, 1)

    nova_posicao[i] = (
        fontes[indice].posicao[i]
        +
        phi * (
            fontes[indice].posicao[i]
            -
            fontes[k].posicao[i]
        )
    )

    nova_posicao[i] = np.clip(
        nova_posicao[i],
        limite_inferior,
        limite_superior
    )

    return nova_posicao



def calcular_probabilidades(fontes):

    fitnesses = np.array([
        fonte.fitness for fonte in fontes
    ])

    aptidoes = 1 / (1 + fitnesses)

    probabilidades = aptidoes / np.sum(aptidoes)

    return probabilidades

def algoritmo_abc():

    num_fontes = tamanho_populacao // 2

    fontes = [

        FonteAlimento(
            dimensoes,
            limite_inferior,
            limite_superior,
            funcao_objetivo
        )

        for _ in range(num_fontes)
    ]

    melhor_solucao = min(
        fontes,
        key=lambda f: f.fitness
    )

    historico = []

    for ciclo in range(ciclos):

        ## Passo 1 - Fase das Abelhas Empregadas

        for i in range(num_fontes):

            nova_posicao = gerar_vizinho(
                fontes,
                i,
                limite_inferior,
                limite_superior
            )

            novo_fitness = avaliar(
                nova_posicao,
                funcao_objetivo
            )


            if novo_fitness < fontes[i].fitness:

                fontes[i].posicao = nova_posicao
                fontes[i].fitness = novo_fitness
                fontes[i].tentativas = 0

            else:
                fontes[i].tentativas += 1

        ## Passo 2 - Fase das abelhas observadoras

        probabilidades = calcular_probabilidades(fontes)

        for _ in range(num_fontes):

            ## Seleção por roleta
            i = np.random.choice(
                range(num_fontes),
                p=probabilidades
            )

            nova_posicao = gerar_vizinho(
                fontes,
                i,
                limite_inferior,
                limite_superior
            )

            novo_fitness = avaliar(
                nova_posicao,
                funcao_objetivo
            )

            if novo_fitness < fontes[i].fitness:

                fontes[i].posicao = nova_posicao
                fontes[i].fitness = novo_fitness
                fontes[i].tentativas = 0

            else:
                fontes[i].tentativas += 1

        ## Passo 3 - Fase das Abelhas Exploratórias (Se necessário*)

        for i in range(num_fontes):

            # abandono da fonte
            if fontes[i].tentativas >= limite_abandono:

                fontes[i] = FonteAlimento(
                    dimensoes,
                    limite_inferior,
                    limite_superior,
                    funcao_objetivo
                )

      ## Melhor resultado Global

        melhor_atual = min(
            fontes,
            key=lambda f: f.fitness
        )

        if melhor_atual.fitness < melhor_solucao.fitness:
            melhor_solucao = melhor_atual

        historico.append(melhor_solucao.fitness)

        print(
            f"Ciclo {ciclo+1:3d} | "
            f"Melhor Fitness: "
            f"{melhor_solucao.fitness:.10f}"
        )

    return melhor_solucao, historico


resultados_finais = []

historicos_execucoes = []

for execucao in range(n_execucoes):

    melhor, historico = algoritmo_abc()

    resultados_finais.append(melhor.fitness)

    historicos_execucoes.append(historico)

    print(
        f"Execução {execucao+1:2d} | "
        f"Fitness Final: {melhor.fitness:.10f}"
    )

print("\n==============================")
print("MELHOR SOLUÇÃO ENCONTRADA")
print("==============================")

print("\nPosição:")
print(melhor.posicao)

print("\nFitness:")
print(melhor.fitness)

plt.figure(figsize=(12, 6))

for i, historico in enumerate(historicos_execucoes):

    plt.plot(
        historico,
        label=f'Execução {i+1}',
        alpha=0.6
    )

plt.title("Convergência das Execuções do ABC")

plt.xlabel("Ciclos")
plt.ylabel("Melhor Fitness")

plt.grid(True)

plt.show()

plt.figure(figsize=(14, 6))

plt.boxplot(historicos_execucoes)

plt.title("Boxplot das Execuções do ABC")

plt.xlabel("Execuções")
plt.ylabel("Fitness")

plt.grid(True)

plt.show()