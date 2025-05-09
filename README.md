# Problema da Mochila com Algoritmos Bio-Inspirados

**Autores:**

- Arthur Henrique Tscha Vieira
- Rafael Rodrigues Ferreira de Andrade

## Descrição do Projeto

Este projeto apresenta implementações de diferentes algoritmos bio-inspirados para resolver o clássico problema da Mochila 0/1 (Knapsack Problem), promovendo o entendimento de técnicas heurísticas e sua aplicação a problemas reais de otimização.

## Problema da Mochila 0/1

No problema da mochila 0/1, temos:

- Um conjunto de `n` itens, cada um com um peso `w[i]` e um valor `v[i]`
- Uma mochila com capacidade limitada `W`

O objetivo é escolher um subconjunto de itens para maximizar o valor total, sem exceder a capacidade da mochila.

### Exemplo

- Pesos = [2, 3, 4, 5]
- Valores = [3, 4, 5, 6]
- Capacidade = 5

A melhor combinação é selecionar os itens 1 (peso 2, valor 3) e 2 (peso 3, valor 4), totalizando valor 7.

## Algoritmos Bio-Inspirados Implementados

Este projeto implementa três algoritmos bio-inspirados para resolver o problema da mochila:

### 1. Algoritmo Genético (GA)

#### Principais características da implementação:

- **Representação**: Indivíduos codificados como vetores binários, onde 1 indica item selecionado e 0 item não selecionado.
- **Função de aptidão**: Soma dos valores dos itens selecionados, com penalização (aptidão zero) para soluções que excedem a capacidade.
- **Seleção**: Método de torneio, que seleciona aleatoriamente k indivíduos e escolhe o melhor.
- **Crossover**: Crossover de um ponto, com probabilidade controlada pelo parâmetro `crossover_rate`.
- **Mutação**: Chance de inverter cada bit da solução, controlada pelo parâmetro `mutation_rate`.
- **Evolução**: A cada geração, a população passa por seleção, crossover e mutação.

#### Pontos fortes e fracos:

- **Pontos fortes**: Boa exploração do espaço de busca, capacidade de escapar de ótimos locais, fácil paralelização.
- **Pontos fracos**: Muitos parâmetros para ajustar, convergência pode ser lenta, pode perder boas soluções ao longo das gerações.

### 2. Otimização por Enxame de Partículas (PSO)

#### Principais características da implementação:

- **Representação**: Partículas com posições em espaço contínuo, convertidas para binárias usando função sigmóide.
- **Velocidade e atualização de posição**: Cada partícula se move com base em três componentes:
  - Componente de inércia (`w`): mantém a tendência de movimento
  - Componente cognitivo (`c1`): atração para a melhor posição pessoal
  - Componente social (`c2`): atração para a melhor posição global
- **Binarização**: As posições contínuas são convertidas para binárias usando função sigmóide e comparação com valor aleatório.
- **Atualização**: Iterativamente avalia e atualiza as melhores posições pessoais e global.

#### Pontos fortes e fracos:

- **Pontos fortes**: Convergência rápida, menos parâmetros que GA, bom equilíbrio entre exploração e explotação.
- **Pontos fracos**: Pode convergir prematuramente para ótimos locais, sensível à inicialização das partículas, adaptação para problemas binários pode perder eficiência.

### 3. Otimização por Colônia de Formigas (ACO)

#### Principais características da implementação:

- **Representação**: Cada formiga constrói uma solução incrementalmente, item por item.
- **Heurística**: Utiliza a razão valor/peso como informação heurística.
- **Feromônio**: Mantém uma matriz de feromônio que influencia a probabilidade de seleção de cada item.
- **Construção de soluções**: Cada formiga seleciona itens com base em:
  - Nível de feromônio (`alpha`)
  - Informação heurística (`beta`)
  - Exploração vs. explotação (`q0`)
- **Atualização de feromônio**: Evaporação e depósito proporcional à qualidade das soluções.

#### Pontos fortes e fracos:

- **Pontos fortes**: Constrói soluções viáveis por design, incorpora facilmente informações específicas do problema, bom desempenho em problemas de otimização combinatória.
- **Pontos fracos**: Tempo de convergência pode ser longo, sensível à configuração de parâmetros, pode estagnar em soluções subótimas.

## Estrutura do Projeto

- `knapsack_genetic.py`: Implementação do Algoritmo Genético
- `knapsack_pso.py`: Implementação do PSO
- `knapsack_aco.py`: Implementação do ACO
- `comparacao_algoritmos.py`: Script para comparar os três algoritmos
- `main.py`: Interface de usuário para facilitar a execução dos algoritmos
- `requirements.txt`: Dependências do projeto

## Como Executar

### Instalação

1. Clone o repositório:

```bash
git clone https://github.com/arthurvieira2003/knapsack-problem-bio.git
cd knapsack-problem-bio
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

### Interface Unificada

Para iniciar a interface interativa que permite executar qualquer algoritmo ou comparação:

```bash
python main.py
```

### Executando os Algoritmos Individualmente

Para executar cada algoritmo individualmente:

```bash
python knapsack_genetic.py
python knapsack_pso.py
python knapsack_aco.py
```

### Comparando os Algoritmos

Para executar a comparação entre os três algoritmos:

```bash
python comparacao_algoritmos.py
```

## Análise Experimental

### Configuração dos Testes

Realizamos experimentos com 3 conjuntos de dados:

1. **Exemplo Pequeno**: 4 itens (exemplo da descrição do problema)
2. **Exemplo Médio**: 9 itens
3. **Exemplo Grande**: 1000 itens

Para cada conjunto, executamos os três algoritmos com parâmetros ajustados e registramos:

- Valor total obtido
- Peso total
- Número de itens selecionados
- Tempo de execução
- Evolução da aptidão ao longo das iterações

### Resultados e Discussão

#### Exemplo Pequeno (4 itens)

Para o exemplo fornecido no problema (capacidade = 5), os três algoritmos encontraram a solução ótima de valor 7 (itens 1 e 2). No entanto, o algoritmo GA convergiu mais rapidamente, seguido pelo PSO e ACO.

#### Exemplo Médio (9 itens)

Para o conjunto médio (capacidade = 200), notamos diferenças mais significativas:

- O GA encontrou a melhor solução (valor 342), utilizando 6 itens
- O PSO obteve uma solução próxima (valor 330)
- O ACO teve desempenho variável, dependendo dos parâmetros

Em termos de tempo de execução, o PSO foi o mais rápido, seguido pelo GA e ACO.

#### Exemplo Grande (1000 itens)

Para o conjunto grande, as diferenças ficaram mais evidentes:

- O GA continua encontrando boas soluções, mas o tempo de execução aumenta significativamente
- O PSO tem bom equilíbrio entre qualidade da solução e tempo de execução
- O ACO se destaca em encontrar soluções factíveis, mas tem maior custo computacional

## Complexidade dos Algoritmos

### Análise de Complexidade Temporal:

1. **Algoritmo Genético (GA)**:

   - O(G _ P _ n), onde:
     - G = número de gerações
     - P = tamanho da população
     - n = número de itens

2. **PSO**:

   - O(I _ P _ n), onde:
     - I = número de iterações
     - P = número de partículas
     - n = número de itens

3. **ACO**:
   - O(I _ A _ n²), onde:
     - I = número de iterações
     - A = número de formigas
     - n = número de itens
   - O fator n² vem do processo de construção de soluções, onde cada formiga avalia todos os itens restantes a cada passo

### Análise de Complexidade Espacial:

1. **Algoritmo Genético**: O(P \* n)
2. **PSO**: O(P \* n)
3. **ACO**: O(A \* n + n) (incluindo matriz de feromônio)

## Conclusões

### Quando usar cada algoritmo?

1. **Algoritmo Genético (GA)**:

   - Melhor quando o espaço de busca é grande e diversificado
   - Quando houver tempo suficiente para convergência
   - Quando a representação binária for natural para o problema

2. **PSO**:

   - Quando a convergência rápida for importante
   - Para problemas com menos itens
   - Quando for desejável um bom equilíbrio entre exploração e explotação

3. **ACO**:
   - Para problemas onde restrições devem ser respeitadas durante a construção da solução
   - Quando informações heurísticas específicas do problema estiverem disponíveis
   - Para problemas de otimização combinatória com estrutura sequencial

### Considerações Finais

Os três algoritmos bio-inspirados demonstraram capacidade de encontrar boas soluções para o problema da mochila 0/1, com diferentes características de desempenho:

- **GA**: Mais exploratório, boa qualidade de solução, mais parametrizado
- **PSO**: Mais equilibrado entre exploração e explotação, convergência mais rápida
- **ACO**: Incorpora facilmente heurísticas específicas do problema, bom para construção incremental de soluções

A escolha do algoritmo deve considerar as características específicas do problema, restrições de tempo computacional e a necessidade de exploração versus explotação do espaço de busca.

Para problemas de mochila muito grandes (milhares de itens), ajustes adicionais podem ser necessários em qualquer um dos algoritmos para manter performance adequada.

## Referências

1. Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley.
2. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In Proceedings of ICNN'95.
3. Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press.
