import cupy as cp
import numpy as np
import time

# Definir o tamanho do vetor
n = 10**9

# Criar um vetor grande em float32
a_cpu = np.ones(n, dtype=np.float32)

# Transferir para GPU
a_gpu = cp.asarray(a_cpu)

# Multiplicação vetorizada diretamente na GPU
scalar = 5.0

start_time = time.time()
a_gpu_result = a_gpu * scalar  # Operação vetorizada
cp.cuda.Stream.null.synchronize()  # Sincroniza o stream padrão
end_time = time.time()

print(f"Tempo de execução do cálculo vetorizado na GPU: {end_time - start_time:.5f} segundos")

# Transferir de volta para CPU para verificação (se necessário)
result_cpu = cp.asnumpy(a_gpu_result)
