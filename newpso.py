import random
import numpy as np
from monocomponent_isotherms import langmuir


class Particle:
    def __init__(self, param_min_max: list) -> None:
        
        """
        Inicializa a posição da partícula com valores aleatórios dentro dos limites fornecidos.

        - param_min_max[]: Lista de listas com os valores mínimos e máximos para cada parâmetro.
        - min_max[]: Cada min_max representa um intervalo [min, max]. Aqui, min_max[0] é o limite inferior (min), e min_max[1] é o limite superior (max).
        - min_max[0] + random.random() * (min_max[1] - min_max[0]):
            -> (min_max[1] - min_max[0]): Calcula a amplitude do intervalo.
            -> random.random() * (min_max[1] - min_max[0]): Gera um valor proporcional à amplitude, escalado para o intervalo [0, amplitude)
            -> min_max[0] + ...: Desloca o valor gerado para o intervalo [min, max)
            
        Exemplo:
        
        param_min_max = 
            [
                [0, 10], [1, 5], [2, 8],
            ],

        Para cada min_max em param_min_max[0]:

        -> 1
            min_max[0] = 0
            min_max[1] = 10 
            
            min_max[1] - min_max[0] = 10 - 0 = 10
            random.random() * (10) = 0.5 * 10 = 5
            min_max[0] + 5 = 0 + 5 = 5
        
        -> 2
            min_max[0] = 1
            min_max[1] = 5 
            
            min_max[1] - min_max[0] = 5 - 1 = 4
            random.random() * (4) = 0.5 * 4 = 2
            min_max[0] + 2 = 1 + 2 = 3
        
        -> 3
            min_max[0] = 2
            min_max[1] = 8
            
            min_max[1] - min_max[0] = 8 - 2 = 6
            random.random() * (6) = 0.5 * 6 = 3
            min_max[0] + 3 = 2 + 3 = 5
        
        Então:
        
        self.position = np.array([5, 3, 5])
        """
        
        self.position: list = np.array(
            [min_max[0] + random.random() * (min_max[1] - min_max[0]) for min_max in param_min_max]
        )

        # Inicializar a velovidade da partícula como um vetor nulo.
        self.velocity: float = np.zeros_like(self.position)

        # Define a melhor posição inicial como a posição atual da partícula.
        self.best_position: list = self.position.copy()

        # Define o fitness da melhor posição como infinito (pior valor possível).
        self.best_fitness: float = float('inf')

        # Define o fitness atual da partícula como infinito.
        self.fitness: float = float('inf')


def velocity(
    w: float,
    c1: float,
    c2: float,
    particle: Particle,
    swarm_best_position: list
) -> float:
    
    # Gera dos números aleatórios entre 0 e 1 para os fatores de aprendizado.
    r1: float = random.random()
    r2: float = random.random()

    # Obtém a velocidade, melhor posição individual e posição atual da partícula.
    v: float = particle.velocity
    best_position: list = particle.best_position
    position: list = particle.position

    """
    Calcula a nova velocidade com base na equação do PSO.
    
    - w * v: Componente de inércia
     -> w: Fator de inércia. Controla o peso da velocidade atual na nova velocidade.
     -> V: Velocidade atual da partícula.
    
    Efeito: Mantém a partícula "em movimento", aproveitando sua tendência anterior
     - Valores altos de w favorecem a exploração (movimento maior)
     - Valor baixos de w favorecem a exploração local (movimento menor)

    ----------
    
    c1 * r1 * (best_position - position): Compopnente Cognitivo
    
    - c1: Constante de aprendizado individual. Define o quanto a partícula confia na sua melhor posição pessoal.
    - r1: Número aleatório entre 0 e 1. Introduz variabilidade ao movimento.
    - best_position - position: Vetor que aponta da posição atual para a melhor posição individual encontrada pela partícula
    
    Efeito: Faz a partícula se mover em direção à sua própria experiência (melhor posição pessoal).
    
    ----------
    
    c2 * r2 * (swarm_best_position - position): Componente Social
    
    - c2: Constante de aprendizado social. Define o quanto a partícula confia na melhor posição do enxame.
    - r2: Número aleatório entre 0 e 1. Assim como r1, introduz variabilidade.
    - swarm_best_position - position: Vetor que aponta da posição atual para a melhor posição global encontrada pelo enxame.

    Efeito: Faz a partícula se mover em direção à melhor solução global, baseada na colaboração do grupo.
    
    ----------
    
    Exemplo:
    
    w = 0.5, c1 = 1.5, c2 = 1.5
    v = [2, -1]
    position = [3, 4]
    best_position = [5, 3]
    swarm_best_position = [6, 2]
    r1 = 0.8, r2 = 0.4
    
    Componente de inércia: 
    w * v = 0.5 * [2, -1] = [1, -0.5]
    
    Fator cognitivo:
    c1 * r1 * (best_position - position)
     -> best_position - position = [5, 3] - [3, 4] = [2, -1]
     -> 1.5 * 0.8 * [2, -1] = [2.4, -1.2]
    
    c2 * r2 * (swarm_best_position - position)
     -> swarm_best_position - position = [6, 2] - [3, 4] = [3, -2]
     -> 1.5 * 0.4 * [3, -2] = [1.8, -1.2]
    
    velocity = [1, -0.5] + [2.4, -1.2] + [1.8, -1.2] = [5.2, -2.9]
    """
    return w * v + c1 * r1 * (best_position - position) + c2 * r2 * (swarm_best_position - position)


def pso(p, qe, obj_func=langmuir, part_n=100, iter_n=100, param_min_max=[[0, 10], [1, 100], [1, 2]], comp_n=1, relative=False):
    
    # Lista para armazenar todas as partículas do enxame.
    particles_list = []

    # Inicializa o melhor fitness global como infinito.
    swarm_best_fitness = float('inf')

    # Cria as partículas e avalia o fitness inicial.
    for _ in range(part_n): # Para cada partícula no enxame.
        for _ in range(comp_n): # Para cada componente da partícula.

            # Cria uma partícula com os limites especificados.
            particle = Particle(param_min_max)

            # Calcula o fitness inicial da partícula usando a função objetivo.
            fitness = obj_func(p, qe, particle.position, relative)

            # Atualiza o fitness e as melhores posições da partícula
            particle.fitness = fitness
            particle.best_position = particle.position.copy()
            particle.best_fitness = fitness

            # Adiciona a partícula a lista
            particles_list.append(particle)

            # Atualiza o melhor fitness e posição do enxame se necessário.
            if _ == 0: # Para a primeira partícula.
                swarm_best_fitness = fitness
                swarm_best_position = particle.position.copy()

            if fitness < swarm_best_fitness: # Se o fitness atual for melhor.
                swarm_best_fitness = fitness
                swarm_best_position = particle.position.copy()

    # Define os parâmetros do PSO (inércia e fatores de aprendizado).
    w = 0.8 # Peso de inércia.
    c1 = 1.2 # Fator cognitivo (influência da melhor posição individual).
    c2 = 1.2 # Fator social (influência da melhor posição do enxame).

    # Iterações para atualizar as partículas.
    for _ in range(iter_n): # Para cada iteração.
        for particle in particles_list: # Para cada partícula no enxame.

            # Atualiza a velocidade da partícula.
            v = velocity(w, c1, c2, particle, swarm_best_position)
            particle.velocity = v
            
            # Atualiza a posição da partícula.
            particle.position += v
            
            # Avalia o fitness da nova posição.
            fitness = obj_func(p, qe, particle.position, relative)
            particle.fitness = fitness

            # Atualiza a melhor posição individual da partícula se necessário.
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Atualiza a melhor posição e fitness global do enxame se necessário.
            if fitness < swarm_best_fitness:
                swarm_best_fitness = fitness
                swarm_best_position = particle.position.copy()

    # Retorna a melhor posição e o melhor fitness encontrados.
    return swarm_best_position, swarm_best_fitness
