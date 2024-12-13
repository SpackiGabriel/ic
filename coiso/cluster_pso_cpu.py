import random
import numpy as np
from monocomponent_isotherms import langmuir
import time
from itertools import product


class Particle:
    def __init__(self, param_min_max: list) -> None:

        self.array = np.array([min_max[0] + random.random() * (min_max[1] - min_max[0]) for min_max in param_min_max])
        self.position: list = self.array

        # Inicializar a velocidade da partícula como um vetor nulo.
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

    return w * v + c1 * r1 * (best_position - position) + c2 * r2 * (swarm_best_position - position)


def divide_intervals(param_min_max, divisions):
    """Divide each interval into subintervals based on the number of divisions."""
    divided_intervals = []
    for min_val, max_val in param_min_max:
        step = (max_val - min_val) / divisions
        intervals = [(min_val + i * step, min_val + (i + 1) * step) for i in range(divisions)]
        divided_intervals.append(intervals)
    return divided_intervals


def pso(
    p,
    qe,
    obj_func=langmuir,
    part_n=100,
    iter_n=100,
    param_min_max=[[0, 10], [1, 100], [1, 2]],
    comp_n=1,
    relative=False,
    initial_positions=None
):
    #print(f"Rodando PSO com os limites: {param_min_max}")

    # Lista para armazenar todas as partículas do enxame.
    particles_list = []

    # Inicializa o melhor fitness global como infinito.
    swarm_best_fitness = float('inf')

    # Cria as partículas e avalia o fitness inicial.
    if initial_positions is None:
        for _ in range(part_n):  # Para cada partícula no enxame.
            for _ in range(comp_n):  # Para cada componente da partícula.

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
                if fitness < swarm_best_fitness:  # Se o fitness atual for melhor.
                    swarm_best_fitness = fitness
                    swarm_best_position = particle.position.copy()
    else:
        for pos in initial_positions:
            particle = Particle(param_min_max)
            particle.position = np.array(pos)
            particle.best_position = np.array(pos)
            particle.fitness = obj_func(p, qe, particle.position, relative)
            particle.best_fitness = particle.fitness
            particles_list.append(particle)

            if particle.fitness < swarm_best_fitness:
                swarm_best_fitness = particle.fitness
                swarm_best_position = particle.position.copy()

    # Define os parâmetros do PSO (inércia e fatores de aprendizado).
    w = 0.8  # Peso de inércia.
    c1 = 1.5  # Fator cognitivo (influência da melhor posição individual).
    c2 = 1.5  # Fator social (influência da melhor posição do enxame).

    # Iterações para atualizar as partículas.
    for _ in range(iter_n):  # Para cada iteração.
        for particle in particles_list:  # Para cada partícula no enxame.

            # Atualiza a velocidade da partícula.
            v = velocity(w, c1, c2, particle, swarm_best_position)
            particle.velocity = v

            # Atualiza a posição da partícula.
            particle.position += particle.velocity

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

    #print(f"Melhor posição encontrada: {swarm_best_position}, Melhor fitness: {swarm_best_fitness}")
    # Retorna a melhor posição e o melhor fitness encontrados.
    return swarm_best_position, swarm_best_fitness


def hierarchical_pso(
    p,
    qe,
    obj_func=langmuir,
    part_n=100,
    iter_n=100,
    param_min_max=[[0, 10000], [1, 10000]],
    comp_n=1,
    divisions=4,
    relative=False
):
    """Run PSO with hierarchical interval splitting."""
    # Divide intervals into subintervals.
    divided_intervals = divide_intervals(param_min_max, divisions)

    # Generate all Cartesian combinations of intervals.
    subinterval_combinations = list(product(*divided_intervals))

    # Store the best particles from each subinterval.
    best_particles = []

    for subinterval in subinterval_combinations:
        #print(f"Subintervalo atual: {subinterval}")
        # Run PSO for the current subinterval.
        best_position, best_fitness = pso(
            p,
            qe,
            obj_func=obj_func,
            part_n=part_n,
            iter_n=iter_n,
            param_min_max=subinterval,
            comp_n=comp_n,
            relative=relative
        )
        best_particles.append((best_position, best_fitness))

    # Print the best particles from each subinterval.
    print("Melhores partículas globais de cada subintervalo:")
    for i, (pos, fit) in enumerate(best_particles):
        print(f"Subintervalo {i + 1}: Melhor posição: {pos}, Fitness: {fit}")

    # Extract the positions of the best particles for the second round.
    global_positions = [pos for pos, _ in best_particles]

    # Run a second round of PSO using the global best positions as the starting swarm.
    final_position, final_fitness = pso(
        p,
        qe,
        obj_func=obj_func,
        part_n=len(global_positions),
        iter_n=iter_n,
        param_min_max=param_min_max,
        comp_n=comp_n,
        relative=relative,
        initial_positions=global_positions
    )

    print(f"Melhor partícula global final: {final_position}, Fitness: {final_fitness}")
    return final_position, final_fitness


# Example usage:
start_time = time.time()
result = hierarchical_pso(
    [0.271004, 1.44862, 2.70512, 3.94841, 5.13112, 6.61931, 8.60419, 11.0863, 13.5677, 16.068, 18.5552, 21.0393, 23.5223, 26.0124, 28.4806, 30.9418, 33.4053, 35.8564, 38.3088, 40.7621, 42.843, 43.8868, 44.409],
    [0.905796, 1.98353, 2.35874, 2.5484, 2.67333, 2.7765, 2.88334, 2.9774, 3.04683, 3.09766, 3.14708, 3.17517, 3.2241, 3.24116, 3.2549, 3.26587, 3.27946, 3.28687, 3.28825, 3.28971, 3.30512, 3.30683, 3.30725],
    obj_func=langmuir,
    part_n=100,
    iter_n=100,
    param_min_max=[[0, 10000], [1, 10000]],
    comp_n=2,
    divisions=10,
    relative=False
)
end_time = time.time()
print(f"Tempo de execução do PSO hierárquico: {end_time - start_time:.5f} segundos")
print("Resultado final:", result)
