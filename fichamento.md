# 📄 Fichamento Comparativo — Avaliação de Desempenho do PSO com Funções Benchmark (2024–2025)

## 1. Abbas et al. (2025) — *Optimizing Benchmark Functions using Particle Swarm Optimization (PSO)*

**Fonte:** Al-Salam Journal for Engineering and Technology  
**Link:** https://doi.org/10.55145/ajest.2025.04.01.019

### Objetivo:
Avaliar o PSO puro em funções benchmark clássicas, analisando tempo de execução, desempenho computacional e capacidade de encontrar ótimos globais.

### Funções Benchmark utilizadas:
- Rastrigin (multimodal)
- Sphere (unimodal)
- Rosenbrock ("valley" function)
- Ackley (paisagem complexa com múltiplos ótimos locais)

### Parâmetros PSO:
- Partículas: 30  
- Iterações: 100  
- Inércia: 0.5  
- Constantes cognitivas/social: 1.5  
- Avaliação em Python

### Métricas avaliadas:
- Posição ótima
- Valor ótimo obtido
- Tempo de execução (em segundos)

### Resultados:
| Função       | Valor ótimo | Tempo (s) |
|--------------|-------------|------------|
| Sphere       | 0.0         | 0.087      |
| Rastrigin    | 0.0         | 0.123      |
| Ackley       | 0.0         | 0.132      |
| Rosenbrock   | 0.0         | 0.156      |

### Conclusão:
PSO mostra bom desempenho na maioria das funções. Sphere é a mais simples; Rosenbrock exige mais tempo devido à topologia estreita.

---

## 2. Sakpere et al. (2025) — *Particle Swarm Optimization and Benchmark Functions: An Extensive Analysis*

**Fonte:** International Journal of Engineering Research in Computer Science and Engineering (IJERCSE)

**Link:** https://ijercse.com/article/1%20January%202025%20IJERCSE.pdf

### Objetivo:
Investigar como diferentes tamanhos populacionais afetam o desempenho do PSO em funções benchmark.

### Funções Benchmark utilizadas:
- Sphere (unimodal)
- Rosenbrock
- Quartic
- Rastrigin (multimodal)
- Schwefel 2.26
- Ackley

### Variações testadas:
- Tamanho do enxame: 20, 50, 70, 100, 120 partículas  
- Ferramenta: MATLAB  
- Iterações: 100  
- Parâmetros: inércia (0.4–1.2), c1=2.8, c2=1.3

### Avaliação:
- Curvas de convergência
- Tempo de execução médio
- Estabilidade dos resultados (30 execuções)

### Conclusão:
PSO apresenta desempenho altamente dependente da configuração de parâmetros e diversidade populacional. As funções multimodais testam mais fortemente a capacidade de exploração do algoritmo.

---

## 3. Yao et al. (2024) — *Research on Hybrid Strategy PSO Algorithm and Its Applications*

**Fonte:** Scientific Reports (Nature Portfolio)  
**Link:** https://doi.org/10.1038/s41598-024-76010-y

### Objetivo:
Propor e avaliar o HSPSO (Hybrid Strategy PSO), comparando com PSO padrão e outros algoritmos bioinspirados.

### Funções Benchmark utilizadas:
- Conjunto CEC 2005 e 2014 (amplas funções multimodais e multidimensionais)

### Avaliação:
- Desempenho médio em múltiplas funções CEC  
- Convergência, diversidade, robustez e acurácia  
- Comparação com: ACO, FA, BOA, GSA, entre outros

### Resultados:
HSPSO superou o PSO clássico em:
- Velocidade de convergência  
- Robustez em múltiplas execuções  
- Alcance de ótimos globais em cenários de alta complexidade

---

# 📌 Conclusão Geral

| Artigo          | Funções Benchmark | Comparação com outros algs | Testes de Tamanho Pop | Métrica de Tempo | Conjunto CEC |
|-----------------|-------------------|-----------------------------|------------------------|------------------|--------------|
| Abbas et al.    | 4 funções clássicas | ❌                          | ❌                     | ✅               | ❌           |
| Sakpere et al.  | 6 funções           | ❌                          | ✅                     | ✅               | ❌           |
| Yao et al.      | CEC 2005/2014      | ✅                          | ❌                     | ✅               | ✅           |

---
