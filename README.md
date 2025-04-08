# Detecção Inteligente de Anomalias em Logs de TI para o Setor Financeiro

## Descrição
Projeto que implementa um pipeline para simulação e análise de logs de TI no setor financeiro. O sistema gera logs fictícios, realiza a ingestão e pré-processamento dos dados, utiliza algoritmos de Inteligência Artificial (Isolation Forest) para detectar e classificar anomalias, gera alertas automatizados e visualiza os eventos processados.

## Principais Funcionalidades
- **Simulação de Logs:** Geração de logs contendo informações como timestamp, IP, nível de acesso e mensagem.
- **Ingestão e Pré-processamento:** Leitura e formatação dos logs utilizando o pandas, com extração de features para análise.
- **Detecção de Anomalias:** Utilização de um modelo de Isolation Forest para identificar padrões anômalos.
- **Classificação dos Eventos:** Categoriza os eventos em Normais, Suspeitos e Críticos com base no resultado do modelo e na presença de indicadores de ataque.
- **Geração de Alertas:** Emite alertas para os eventos críticos, simulando integração com sistemas de notificação.
- **Visualização dos Eventos:** Criação de gráficos que detalham a distribuição dos eventos e sua evolução temporal.

## Instruções de Instalação e Configuração
1. Clone o repositório e navegue para a pasta do projeto; crie um ambiente virtual, se desejar, e instale as dependências:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   python -m venv venv && source venv/bin/activate  # Linux/MacOS (ou use "venv\Scripts\activate" no Windows)
   pip install -r requirements.txt
2. Execute o Jupyter Notebook ou o script Python conforme sua necessidade:
   ```bash
    jupyter notebook
    # ou
    python <nome_do_script>.py
3. Formato esperado do arquivo logs_reais.txt (caso use o arquivo ..._ler_arquivo.py)
   ```yaml
   2025-04-07 14:23:01 | 192.168.1.1 | INFO | Operação realizada com sucesso
   2025-04-07 14:25:12 | 192.168.1.10 | WARN | Tentativa de acesso indevido detectada
Ou seja:
**data hora | IP | nível | mensagem**

## Autores
* Carlos Matos | RGM: 29622182
* Gustavo Taglianetti | RGM: 29649111
