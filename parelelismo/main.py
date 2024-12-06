import cupy as cp
import numpy as np
import time

# Definir o tamanho do vetor
n = 10**7

# Criar um vetor grande (em CPU)
a_cpu = np.random.rand(n)

# Transferir para GPU
a_gpu = cp.asarray(a_cpu)

# Kernel CUDA em uma string
kernel_code = """
extern "C" __global__
void multiply_by_scalar(float *a, float scalar, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] = a[idx] * scalar;
    }
}
"""

# Compilar o kernel
kernel = cp.RawKernel(kernel_code, 'multiply_by_scalar')

# Definir o escalar e o número de elementos
scalar = 5.0
num_elements = n

# Alocar memória na GPU para o vetor de entrada
a_gpu_result = cp.copy(a_gpu)

# Definir o número de threads e blocos
block_size = 1024  # número de threads por bloco
grid_size = (num_elements + block_size - 1) // block_size  # número de blocos necessários

# Executar o kernel
start_time = time.time()
kernel((grid_size,), (block_size,), (a_gpu_result, scalar, num_elements))
cp.cuda.Stream.null.synchronize()  # Esperar a conclusão do cálculo
end_time = time.time()

print(f"Tempo de execução do kernel na GPU: {end_time - start_time:.5f} segundos")

# Verifique os primeiros valores do resultado
result_cpu = cp.asnumpy(a_gpu_result)
print("Primeiros 5 valores do resultado:", result_cpu[:5])
