# Data Science Supply Chain Portfolio
# Portfólio de Ciência de Dados Aplicada: Otimização de Logística e Supply Chain 🚀

Olá! 👋 Bem-vindo(a) ao meu portfólio de projetos onde aplico **Ciência de Dados**, **Machine Learning** e **Otimização** para resolver desafios práticos no universo da **Logística e Supply Chain**.

Meu foco é transformar dados (sejam eles reais, públicos ou simulados) em insights acionáveis e soluções eficientes que podem levar a melhores decisões de estoque, rotas de transporte mais eficientes, previsões de demanda mais precisas e uma cadeia de suprimentos mais resiliente e sustentável.

Os projetos são desenvolvidos como **Jupyter Notebooks** (`.ipynb`) armazenados diretamente neste repositório GitHub.

Este portfólio está organizado nos seguintes pilares estratégicos:

1.  **[Pilar 1](#pilar-1-planejamento-e-previsão-de-demanda-o-ponto-de-partida-)**: Planejamento e Previsão de Demanda
2.  **[Pilar 2](#pilar-2-gestão-e-otimização-de-estoques-o-equilíbrio-custo-x-serviço-)**: Gestão e Otimização de Estoques
3.  **[Pilar 3](#pilar-3-otimização-de-transporte-e-rede-logística-movimentando-os-bens-)**: Otimização de Transporte e Rede Logística
4.  **[Pilar 4](#pilar-4-tópicos-avançados-e-visibilidade-end-to-end-o-futuro-e-a-integração-)**: Tópicos Avançados e Visibilidade End-to-End

---

## Como Navegar pelos Projetos 🧭

*   **Localização:** Todos os notebooks (`.ipynb`) e códigos associados residem diretamente neste repositório GitHub, organizados dentro de pastas correspondentes a cada projeto.
*   **Visualização:** Você pode visualizar o conteúdo dos notebooks diretamente na interface do GitHub.
*   **Execução Interativa (Opcional - Via Colab):** Para uma execução interativa, você pode abrir os notebooks `.ipynb` deste repositório no Google Colab (vá em `File > Open notebook > GitHub`, procure este repositório e selecione o notebook desejado). Isso permite rodar o código célula por célula em um ambiente hospedado.
*   **Dados e Dependências:** Instruções sobre fontes de dados e instalação de bibliotecas necessárias (`pip install ...`) estarão detalhadas no início de cada notebook.

**Importante:** O uso do Google Colab aqui é primariamente para fins de desenvolvimento, experimentação e visualização interativa. Uma etapa subsequente natural, representando o ciclo completo do trabalho de um Cientista de Dados, seria a **implementação e o deploy** destes modelos ou análises em um ambiente de nuvem (como AWS, GCP ou Azure) para torná-los operacionais (ex: APIs de previsão, dashboards automatizados, etc.). Esta etapa de deploy não está contemplada diretamente nos notebooks, mas é o objetivo final de muitos desses projetos em um cenário real.

---

## Pilar 1: Planejamento e Previsão de Demanda (O Ponto de Partida) <a name="pilar-1-planejamento-e-previsão-de-demanda-o-ponto-de-partida-"></a>

🎯 **Objetivo:** Prever com acurácia o que, quando e onde os clientes demandarão produtos, formando a base para decisões logísticas.

### Projetos:

1.  **Previsão de Vendas por SKU (Séries Temporais & ML):**
    *   **Descrição:** Desenvolvimento de modelos para prever vendas futuras de itens específicos usando dados históricos (públicos ou simulados).
    *   **Técnicas/Foco:** ARIMA, Prophet, Modelos de ML (Random Forest, XGBoost), Feature Engineering (tempo), Avaliação de Métricas (MAPE, RMSE), Análise de Sazonalidade e Tendências.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Statsmodels`, `Prophet`, `Scikit-learn`, `Matplotlib`, `Seaborn`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 1.1 - EM BREVE]`

2.  **Impacto de Promoções/Eventos na Demanda:**
    *   **Descrição:** Incorporar dados de promoções, feriados ou eventos externos (clima, notícias - simulados se necessário) como regressores para entender e quantificar seu impacto na demanda.
    *   **Técnicas/Foco:** Feature Engineering, Modelagem com Regressores Externos, Interpretabilidade do Modelo.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Scikit-learn`, `XGBoost`, `SHAP`, `Statsmodels`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 1.2 - EM BREVE]`

3.  **Análise de Erro de Previsão:**
    *   **Descrição:** Implementar um sistema para monitorar a acurácia das previsões ao longo do tempo, identificar bias (superestimação/subestimação consistente) e sugerir melhorias no modelo.
    *   **Técnicas/Foco:** Monitoramento de modelos, Análise Estatística de Erros, Visualização de Dados.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Matplotlib`, `Seaborn`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 1.3 - EM BREVE]`

4.  **Previsão Hierárquica:**
    *   **Descrição:** Prever a demanda em múltiplos níveis (Total da Categoria -> Tipo de Produto -> SKU específico) e reconciliar as previsões para garantir consistência.
    *   **Técnicas/Foco:** Estrutura de dados complexa, Técnicas de Reconciliação (Top-Down, Bottom-Up, etc.), Séries Temporais.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, bibliotecas de forecasting hierárquico (ex: `scikit-hts`) ou implementação customizada, `Statsmodels`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 1.4 - EM BREVE]`

5.  **Previsão para Novos Produtos (Cold Start):**
    *   **Descrição:** Simular o lançamento de um novo produto com poucos dados históricos, usando dados de produtos similares ou atributos do produto para fazer uma previsão inicial.
    *   **Técnicas/Foco:** Abordagens criativas para dados limitados, Modelagem baseada em atributos, Analogia com produtos similares, Machine Learning.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Scikit-learn`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 1.5 - EM BREVE]`

---

## Pilar 2: Gestão e Otimização de Estoques (O Equilíbrio Custo x Serviço) <a name="pilar-2-gestão-e-otimização-de-estoques-o-equilíbrio-custo-x-serviço-"></a>

⚖️ **Objetivo:** Definir níveis ótimos de estoque para atender à demanda sem incorrer em custos excessivos ou rupturas.

### Projetos:

1.  **Classificação ABC e Política de Estoque:**
    *   **Descrição:** Usar dados de vendas/custo para classificar produtos (Curva ABC) e definir políticas de contagem/revisão de estoque diferenciadas para cada classe.
    *   **Técnicas/Foco:** Análise de Pareto (Curva ABC), Análise de Dados, Segmentação, Definição de Regras de Negócio.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Matplotlib`, `Seaborn`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 2.1 - EM BREVE]`

2.  **Cálculo de Estoque de Segurança:**
    *   **Descrição:** Modelar a variabilidade da demanda e do lead time de fornecedores para calcular o estoque de segurança ideal para diferentes níveis de serviço desejados.
    *   **Técnicas/Foco:** Estatística Aplicada (Distribuições de Probabilidade), Fórmulas de Estoque de Segurança, Simulação Monte Carlo (opcional).
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `NumPy`, `SciPy.stats`, `Matplotlib`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 2.2 - EM BREVE]`

3.  **Otimização do Ponto de Reposição (ROP) e Quantidade (EOQ/Variantes):**
    *   **Descrição:** Implementar modelos clássicos (EOQ, ROP) e variantes (ex: com descontos por quantidade) para determinar quando e quanto pedir de cada item.
    *   **Técnicas/Foco:** Modelos de OR simples (Pesquisa Operacional), Análise de Trade-off (Custo de Pedido vs Custo de Manutenção), EOQ, ROP.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `NumPy`, `Math`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 2.3 - EM BREVE]`

4.  **Análise de Estoque Obsoleto/Lento:**
    *   **Descrição:** Criar um dashboard ou análise para identificar produtos com baixo giro ou risco de obsolescência, sugerindo ações (promoção, liquidação).
    *   **Técnicas/Foco:** Visualização de Dados, KPIs de Inventário (Giro de Estoque, Cobertura), Análise de Dados.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Matplotlib`, `Seaborn`, `Numpy`, (Potencialmente `Plotly`, `Dash`, `Streamlit`).
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 2.4 - EM BREVE]`

5.  **Simulação de Políticas de Estoque:**
    *   **Descrição:** Usar bibliotecas como SimPy (ou similar) para simular e comparar diferentes políticas de estoque (ex: revisão contínua vs. periódica) e seus custos totais (holding + shortage + ordering) sob diferentes cenários de demanda.
    *   **Técnicas/Foco:** Simulação de Eventos Discretos, Análise de Sensibilidade, Comparação de Políticas de Estoque.
    *   **Tecnologias/Bibliotecas Principais:** `SimPy`, `Pandas`, `NumPy`, `Matplotlib`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 2.5 - EM BREVE]`

---

## Pilar 3: Otimização de Transporte e Rede Logística (Movimentando os Bens) <a name="pilar-3-otimização-de-transporte-e-rede-logística-movimentando-os-bens-"></a>

🚚 **Objetivo:** Planejar rotas, modos de transporte e localizações de instalações para movimentar produtos de forma rápida, econômica e confiável.

### Projetos:

1.  **Otimização de Rota de Veículo (VRP - Básico):**
    *   **Descrição:** Usar bibliotecas como Google OR-Tools para resolver um problema de "caixeiro viajante" (TSP) ou VRP simples (um veículo, múltiplos clientes, minimizar distância/tempo).
    *   **Técnicas/Foco:** Introdução a OR-Tools (ou similar), Modelagem de Otimização, TSP, VRP.
    *   **Tecnologias/Bibliotecas Principais:** `Google OR-Tools`, `Pandas`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 3.1 - EM BREVE]`

2.  **VRP com Restrições:**
    *   **Descrição:** Adicionar complexidade ao projeto VRP básico: janelas de tempo de entrega (VRPTW), capacidade do veículo (CVRP), múltiplos veículos.
    *   **Técnicas/Foco:** Modelagem de restrições em Otimização, CVRP, VRPTW, Heurísticas e Metaheurísticas (se necessário).
    *   **Tecnologias/Bibliotecas Principais:** `Google OR-Tools`, `Pandas`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 3.2 - EM BREVE]`

3.  **Análise de Custos de Transporte:**
    *   **Descrição:** Analisar dados históricos de frete (simulados ou públicos) para comparar custos entre diferentes transportadoras, rotas ou modos (rodoviário, aéreo), identificando oportunidades de economia.
    *   **Técnicas/Foco:** Análise Exploratória de Dados (EDA), Visualização, Análise de Custos.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Matplotlib`, `Seaborn`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 3.3 - EM BREVE]`

4.  **Modelagem de Localização de Instalações (Simples):**
    *   **Descrição:** Usar dados de clientes (localização, demanda) para sugerir a localização ótima de um único Centro de Distribuição (CD) usando técnicas como centro de gravidade ou k-means clustering.
    *   **Técnicas/Foco:** Geoprocessamento básico, Clusterização (K-Means), Método do Centro de Gravidade, Otimização de Localização (P-Median, P-Center - conceitual).
    *   **Tecnologias/Bibliotecas Principais:** `GeoPandas`, `Scikit-learn` (para K-Means), `Pandas`, `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 3.4 - EM BREVE]`

5.  **Previsão de Tempo de Entrega (ETA):**
    *   **Descrição:** Criar um modelo de Machine Learning para prever o tempo de entrega de um pedido com base na origem, destino, distância, hora do dia, dia da semana, e talvez dados externos simulados (tráfego, clima).
    *   **Técnicas/Foco:** Modelagem Preditiva (Regressão), Feature Engineering, Machine Learning (Gradient Boosting, Redes Neurais), Análise de Dados Geoespaciais.
    *   **Tecnologias/Bibliotecas Principais:** `Scikit-learn`, `Pandas`, `XGBoost`, `Matplotlib`, `GeoPandas` (opcional), `Numpy`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 3.5 - EM BREVE]`

---

## Pilar 4: Tópicos Avançados e Visibilidade End-to-End (O Futuro e a Integração) <a name="pilar-4-tópicos-avançados-e-visibilidade-end-to-end-o-futuro-e-a-integração-"></a>

🔗 **Objetivo:** Explorar temas emergentes, integrar dados de ponta a ponta e aplicar técnicas avançadas para maior resiliência e inteligência na cadeia.

### Projetos:

1.  **Dashboard de Visibilidade da Cadeia:**
    *   **Descrição:** Integrar dados simulados de diferentes etapas (pedidos, estoque, transporte, entrega) para criar um painel que mostre o status geral da cadeia e KPIs chave.
    *   **Técnicas/Foco:** Integração de dados, Visualização Holística, Definição de KPIs, Business Intelligence.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `Plotly`/`Dash`/`Streamlit` (para interatividade), ou link para `Tableau Public`/`Power BI`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta, App Streamlit/Dash ou Dashboard Público do Projeto 4.1 - EM BREVE]`

2.  **Análise de Risco na Cadeia:**
    *   **Descrição:** Simular o impacto de eventos disruptivos (ex: fechamento de porto, falha de fornecedor) nos custos e prazos da cadeia. Ou usar análise de texto em notícias (simulado) para identificar riscos potenciais.
    *   **Técnicas/Foco:** Simulação, Análise de Cenários, Processamento de Linguagem Natural (NLP - básico), Avaliação de Risco.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `SimPy` (para simulação), `NLTK`/`spaCy`/`Scikit-learn` (para NLP), `Matplotlib`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 4.2 - EM BREVE]`

3.  **Otimização de Malha Logística (Avançado):**
    *   **Descrição:** Usar ferramentas de otimização para um problema mais complexo de decidir onde localizar múltiplos CDs e quais clientes cada um deve atender, considerando custos de transporte e instalação.
    *   **Técnicas/Foco:** Otimização em Larga Escala (requer mais estudo de OR), Modelos de Localização-Alocação, Pesquisa Operacional.
    *   **Tecnologias/Bibliotecas Principais:** `Google OR-Tools` (para problemas menores/heurísticas), solvers de otimização (interfaces `GurobiPy`, `CPLEX` - podem exigir licenças), `Pandas`, `GeoPandas`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 4.3 - EM BREVE]`

4.  **Detecção de Anomalias em Operações:**
    *   **Descrição:** Usar Machine Learning (ex: Isolation Forest, Autoencoders) para detectar padrões incomuns em dados de transporte (desvios de rota, atrasos excessivos) ou estoque (rupturas inesperadas).
    *   **Técnicas/Foco:** Anomaly Detection, Machine Learning Não Supervisionado (Isolation Forest, One-Class SVM), Autoencoders (Deep Learning), Análise de Séries Temporais.
    *   **Tecnologias/Bibliotecas Principais:** `Scikit-learn` (IsolationForest, OneClassSVM), `Pandas`, `TensorFlow`/`Keras` (para Autoencoders), `Matplotlib`.
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 4.4 - EM BREVE]`

5.  **Análise de Sustentabilidade:**
    *   **Descrição:** Calcular a pegada de carbono estimada das operações de transporte com base nas distâncias, modos e fatores de emissão (simulados).
    *   **Técnicas/Foco:** Cálculo de Métricas de ESG (Ambiental, Social e Governança), Análise de Trade-off (Custo vs. Emissão), Modelagem de Emissões.
    *   **Tecnologias/Bibliotecas Principais:** `Pandas`, `NumPy`, `GeoPandas` (para cálculos de distância).
    *   **(Status: Planejado)**
    *   **Notebook/Projeto:** `[Link para a Pasta ou Notebook .ipynb do Projeto 4.5 - EM BREVE]`

---

## Tecnologias e Ferramentas Gerais 🛠️

*   **Linguagem Principal:** Python
*   **Ambiente de Desenvolvimento/Visualização:** Jupyter Notebooks, Google Colaboratory (opcional para execução interativa)
*   **Manipulação de Dados:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Statsmodels, Prophet, TensorFlow/Keras
*   **Otimização:** Google OR-Tools, SciPy.optimize, (Potencialmente Gurobi, CPLEX)
*   **Simulação:** SimPy
*   **Visualização:** Matplotlib, Seaborn, Plotly, GeoPandas, (Potencialmente Dash, Streamlit, Tableau, Power BI)
*   **NLP (Básico):** NLTK, spaCy
*   **Versionamento:** Git, GitHub
*   **Perspectiva de Deploy (Nuvem):** AWS (SageMaker, Lambda, etc.), GCP (AI Platform, Cloud Functions, etc.), Azure (Machine Learning, Functions, etc.)

---
