# Funções
import math
import random
import matplotlib.pyplot as plt 

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

# Definir a Função Problema e Seus Limites
funcao = rosenbrock
var_min = -5.12
var_max = 5.12

# Parâmetros
n_populacao = 30
quant_genes = 30
geracoes = 20
taxa_crossover = 1.0
taxa_mutacao = 0.05

# Seleções
def selecao_torneio(populacao_avaliada):
    a, b = random.sample(populacao_avaliada, 2)
    return a if a[1] < b[1] else b

def selecao_roleta(populacao_avaliada):
    soma_fitness = sum(1/(1 + ind[1]) for ind in populacao_avaliada)

    r = random.uniform(0, soma_fitness)
    acumulado = 0

    for ind in populacao_avaliada:
        acumulado += 1/(1 + ind[1])
        if acumulado >= r:
            return ind
        
# Crossover (Cruzamentos)
def crossover_1ponto(pai1, pai2):
    ponto = random.randint(1, len(pai1)-1)

    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]

    return filho1, filho2

def crossover_2pontos(pai1, pai2):
    p1, p2 = sorted(random.sample(range(len(pai1)), 2))

    filho1 = pai1[:p1] + pai2[p1:p2] + pai1[p2:]
    filho2 = pai2[:p1] + pai1[p1:p2] + pai2[p2:]

    return filho1, filho2

# Mutação
def mutacao(individuo):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = random.uniform(var_min, var_max)
    return individuo

# Função para Executar o Algoritmo
def executar_ag():

    populacao = [
        [random.uniform(var_min, var_max) for _ in range(quant_genes)]
        for _ in range(n_populacao)
    ]

    melhores = []

    for g in range(geracoes):

        populacao_avaliada = [(ind, funcao(ind)) for ind in populacao]
        populacao_avaliada.sort(key=lambda x: x[1])

        melhores.append(populacao_avaliada[0][1])

        nova_populacao = []

        while len(nova_populacao) < n_populacao:

            if tipo_selecao == "torneio":
                p1 = selecao_torneio(populacao_avaliada)[0]
                p2 = selecao_torneio(populacao_avaliada)[0]
            else:
                p1 = selecao_roleta(populacao_avaliada)[0]
                p2 = selecao_roleta(populacao_avaliada)[0]

            if random.random() < taxa_crossover:
              if tipo_crossover == "1ponto":
                filho1, filho2 = crossover_1ponto(p1, p2)
              else:
                filho1, filho2 = crossover_2pontos(p1, p2)
            else:
               filho1, filho2 = p1[:], p2[:]

            filho1 = mutacao(filho1)
            filho2 = mutacao(filho2)

            nova_populacao.append(filho1)
            if len(nova_populacao) < n_populacao:
                nova_populacao.append(filho2)

        populacao = nova_populacao

    melhor_final = min(funcao(ind) for ind in populacao)

    return melhores, melhor_final

# Execução Automatizada para Cada Função
melhor_global = float("inf")
melhor_curva = None
melhor_config = ""

resultados_t1 = []
resultados_t2 = []
resultados_r1 = []
resultados_r2 = []

# Torneio + 1 ponto
tipo_selecao = "torneio"
tipo_crossover = "1ponto"
for _ in range(30):
    melhores, melhor = executar_ag()

    resultados_t1.append(melhor)

    if melhor < melhor_global:
        melhor_global = melhor
        melhor_curva = melhores
        melhor_config = "Torneio + 1 ponto"

# Torneio + 2 pontos
tipo_selecao = "torneio"
tipo_crossover = "2pontos"
for _ in range(30):
    melhores, melhor = executar_ag()

    resultados_t2.append(melhor)

    if melhor < melhor_global:
        melhor_global = melhor
        melhor_curva = melhores
        melhor_config = "Torneio + 2 pontos"

# Roleta + 1 ponto
tipo_selecao = "roleta"
tipo_crossover = "1ponto"
for _ in range(30):
    melhores, melhor = executar_ag()

    resultados_r1.append(melhor)

    if melhor < melhor_global:
        melhor_global = melhor
        melhor_curva = melhores
        melhor_config = "Roleta + 1 ponto"

# Roleta + 2 pontos
tipo_selecao = "roleta"
tipo_crossover = "2pontos"
for _ in range(30):
    melhores, melhor = executar_ag()

    resultados_r2.append(melhor)

    if melhor < melhor_global:
        melhor_global = melhor
        melhor_curva = melhores
        melhor_config = "Roleta + 2 pontos"

def estatisticas(nome, lista):
    media = sum(lista) / len(lista)
    melhor = min(lista)
    pior = max(lista)

    print(f"\n{nome}")
    print(f"Média: {media}")
    print(f"Melhor valor: {melhor}")
    print(f"Pior valor: {pior}")

# Mostrar resultados
estatisticas("Torneio + 1 ponto", resultados_t1)
estatisticas("Torneio + 2 pontos", resultados_t2)
estatisticas("Roleta + 1 ponto", resultados_r1)
estatisticas("Roleta + 2 pontos", resultados_r2)

# Gráfico de Convergência do Melhor Resultado
plt.plot(melhor_curva)
plt.title(f"Gráfico de Convergência - {melhor_config}")
plt.xlabel("Gerações")
plt.ylabel("Fitness")
plt.show()

# Gráfico BoxPlot das 4 Configurações
plt.boxplot([resultados_t1, resultados_t2, resultados_r1, resultados_r2])
plt.xticks([1, 2, 3, 4], [
    "Torneio+1p",
    "Torneio+2p",
    "Roleta+1p",
    "Roleta+2p"
])
plt.title(f"Sphere")
plt.ylabel("Valor Final")
plt.show()