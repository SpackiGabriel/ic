import cupy as cp
import numpy as np
import time

# Kernel para atualizar velocidade, posição e fitness
kernel_code = """
extern "C" __global__
void update_velocity_position(double *position, double *velocity, 
                              double *personal_best, double *global_best, 
                              double *fitness, double *personal_best_fitness, 
                              double *r1, double *r2, double w, double c1, double c2, 
                              int n_particles, int dim, int n_points, double *p, double *q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles * dim) {
        int particle_idx = idx / dim;
        int dim_idx = idx % dim;
        int pos = particle_idx * dim + dim_idx;

        // Atualizar velocidade
        velocity[pos] = w * velocity[pos] 
                      + c1 * r1[pos] * (personal_best[pos] - position[pos])
                      + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);

        // Atualizar posição
        position[pos] += velocity[pos];

        // Avaliar fitness com base na função de Langmuir
        if (dim_idx == 0) {
            double qmax = position[particle_idx * dim];
            double b = position[particle_idx * dim + 1];
            double fit = 0.0;
            for (int i = 0; i < n_points; ++i) {
                double q_calc = (qmax * b * p[i]) / (1.0 + b * p[i]);
                fit += (q[i] - q_calc) * (q[i] - q_calc);
            }
            fitness[particle_idx] = fit;

            // Atualizar melhor posição pessoal
            if (fit < personal_best_fitness[particle_idx]) {
                personal_best_fitness[particle_idx] = fit;
                for (int d = 0; d < dim; ++d) {
                    personal_best[particle_idx * dim + d] = position[particle_idx * dim + d];
                }
            }
        }
    }
}
"""

# Compilar o kernel
pso_kernel = cp.RawKernel(kernel_code, 'update_velocity_position')

# GPU-PSO Function
def gpu_pso(p, q, n_particles, dim, n_iterations, w, c1, c2, param_min_max):
    # Certificar-se de que p e q estão na GPU
    p = cp.asarray(p, dtype=cp.float64)
    q = cp.asarray(q, dtype=cp.float64)

    # Inicializar posições e velocidades
    position = cp.zeros((n_particles, dim), dtype=cp.float64)
    for i in range(dim):
        lower, upper = param_min_max[i]
        position[:, i] = cp.random.uniform(lower, upper, size=n_particles).astype(cp.float64)

    velocity = cp.zeros_like(position)
    personal_best = position.copy()
    fitness = cp.full(n_particles, cp.inf, dtype=cp.float64)
    personal_best_fitness = cp.full(n_particles, cp.inf, dtype=cp.float64)
    global_best = cp.zeros(dim, dtype=cp.float64)
    global_best_fitness = cp.inf

    # Configuração CUDA
    threads_per_block = 256
    blocks_per_grid = (n_particles * dim + threads_per_block - 1) // threads_per_block

    # Iterações do PSO
    for _ in range(n_iterations):
        # Gerar números aleatórios r1 e r2 fora do kernel
        r1 = cp.random.rand(n_particles, dim).astype(cp.float64)
        r2 = cp.random.rand(n_particles, dim).astype(cp.float64)

        # Chamar o kernel para atualizar partículas
        pso_kernel((blocks_per_grid,), (threads_per_block,),
                   (position.ravel(), velocity.ravel(),
                    personal_best.ravel(), global_best,
                    fitness, personal_best_fitness,
                    r1.ravel(), r2.ravel(), w, c1, c2, n_particles, dim,
                    len(p), p, q))

        # Atualizar global_best
        min_idx = cp.argmin(personal_best_fitness)
        if personal_best_fitness[min_idx] < global_best_fitness:
            global_best_fitness = personal_best_fitness[min_idx]
            global_best = personal_best[min_idx]

    return cp.asnumpy(global_best), global_best_fitness

# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada (p e q)
    p = [0.271004, 1.44862, 2.70512, 3.94841, 5.13112, 6.61931, 8.60419, 11.0863, 13.5677, 16.068,
         18.5552, 21.0393, 23.5223, 26.0124, 28.4806, 30.9418, 33.4053, 35.8564, 38.3088, 40.7621,
         42.843, 43.8868, 44.409]
    q = [0.905796, 1.98353, 2.35874, 2.5484, 2.67333, 2.7765, 2.88334, 2.9774, 3.04683, 3.09766,
         3.14708, 3.17517, 3.2241, 3.24116, 3.2549, 3.26587, 3.27946, 3.28687, 3.28825, 3.28971,
         3.30512, 3.30683, 3.30725]

    n_particles = 1000
    dim = 2  # Apenas [Qmax, b] por partícula
    n_iterations = 100
    w = 0.8
    c1 = 1.5
    c2 = 1.5
    param_min_max = [[0, 10], [0, 100]]  # Limites para Qmax e b

    start_time = time.time()
    best_position, best_fitness = gpu_pso(p, q, n_particles, dim, n_iterations, w, c1, c2, param_min_max)
    end_time = time.time()

    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Calcular fitness manualmente para validar
    qmax, b = best_position
    fitness_manual = sum([(q[i] - (qmax * b * p[i]) / (1 + b * p[i])) ** 2 for i in range(len(p))])
    print(f"Fitness calculado manualmente: {fitness_manual}")