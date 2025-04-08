# =============================================================================
# Projeto: Detecção Inteligente de Anomalias em Logs de TI para o Setor Financeiro
# Objetivo: Desenvolver um pipeline que realiza a ingestão de logs reais,
#          aplica pré-processamento sem manipulação dos dados originais, 
#          e utiliza algoritmos de IA considerando parâmetros do contexto real.
# =============================================================================

# =============================================================================
# Etapa 1: Importação das bibliotecas necessárias
# =============================================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
import seaborn as sns
import json
import os

# Configurações para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# =============================================================================
# Etapa 2: Função para carregar parâmetros reais do contexto
# - Esses parâmetros podem influenciar a análise (ex.: fuso horário, taxa esperada de anomalias,
#   parâmetros de configuração do modelo, etc.).
# - É esperado um arquivo de configuração no formato JSON (ex: config_contexto.json).
# =============================================================================
def ler_config_contexto(config_path="config_contexto.json"):
    """Lê arquivo de configuração com parâmetros reais do contexto de coleta dos dados.
       Retorna um dicionário com os parâmetros ou um dicionário vazio se o arquivo não existir."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                print("Configuração de contexto carregada com sucesso.")
                return config
            except json.JSONDecodeError:
                print("Erro ao ler o arquivo de configuração. Verifique o formato do JSON.")
                return {}
    else:
        print("Arquivo de configuração não encontrado. Usando parâmetros padrão.")
        return {}

# =============================================================================
# Etapa 3: Ingestão dos Dados a partir do arquivo de log real
# - O arquivo de log deve ser uma representação fiel do ambiente real.
# =============================================================================
def ingestao_logs(file_path):
    """Lê o arquivo de logs real e retorna um DataFrame estruturado sem manipulação dos dados."""
    data = []
    padrao = r"^(.*?) \| (.*?) \| (.*?) \| (.*)$"  # Expressão regular para extrair os campos

    with open(file_path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            match = re.match(padrao, linha)
            if match:
                ts, ip, nivel, mensagem = match.groups()
                data.append({
                    "timestamp": ts,
                    "ip": ip,
                    "nivel": nivel,
                    "mensagem": mensagem
                })
    df = pd.DataFrame(data)
    # Conversão da coluna de timestamp para datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    return df

# =============================================================================
# Etapa 4: Pré-processamento dos Dados
# - Extração de features relevantes sem manipular os dados originais.
# =============================================================================
def extrai_features(df):
    """Adiciona colunas com features extraídas dos logs sem alterar os dados originais."""
    # Flag para indicar se a mensagem sugere comportamento anômalo (buscando pela palavra 'ataque')
    df['possivel_ataque'] = df['mensagem'].apply(lambda x: 1 if "ataque" in x.lower() else 0)
    # Extração do horário para análises temporais
    df['hora'] = df['timestamp'].dt.hour
    return df

# =============================================================================
# Etapa 5: Análise com IA – Detecção de Anomalias
# - Utiliza o IsolationForest ajustado com parâmetros do contexto real, se fornecidos.
# =============================================================================
def detectar_anomalias(df, config):
    """
    Treina o modelo IsolationForest considerando as features extraídas e os parâmetros do contexto.
    - Utiliza a taxa de contaminação definida em config (se disponível) para ajustar o modelo.
    """
    # Seleção das features para o modelo
    features = df[['hora', 'possivel_ataque']]
    
    # Obtém o parâmetro de contaminação real do contexto; se não estiver definido, usa 0.1
    contamination = config.get("contamination", 0.1)
    print(f"Utilizando taxa de contaminação = {contamination}")
    
    # Criação e treinamento do modelo de detecção de anomalias
    modelo_if = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    modelo_if.fit(features)

    # Predição de anomalias: -1 para anomalia, 1 para normal
    df['anomaly_score'] = modelo_if.decision_function(features)
    df['anomaly'] = modelo_if.predict(features)
    
    print("Detecção de anomalias concluída. Contagem de rótulos (-1: anomalia, 1: normal):")
    print(df['anomaly'].value_counts())
    return df

# =============================================================================
# Etapa 6: Classificação dos Eventos
# - Classifica eventos em Normais, Suspeitos ou Críticos, considerando a previsão do modelo
#   e a presença de indicadores no log.
# =============================================================================
def classifica_eventos(row):
    """Classifica cada log com base na predição do modelo e na presença de indicadores de ataque."""
    if row['anomaly'] == -1 and row['possivel_ataque'] == 1:
        return "Crítico"
    elif row['anomaly'] == 1 and row['possivel_ataque'] == 1:
        return "Suspeito"
    else:
        return "Normal"

# =============================================================================
# Etapa 7: Geração de Alertas Automatizados
# - Para cada log classificado como 'Crítico', gera um alerta.
# =============================================================================
def gerar_alertas(df):
    """Gera alertas para eventos críticos encontrados nos logs."""
    eventos_criticos = df[df['classificacao'] == "Crítico"]
    for idx, evento in eventos_criticos.iterrows():
        alerta = (f"ALERTA CRÍTICO: {evento['timestamp']} | IP: {evento['ip']} | "
                  f"Nível: {evento['nivel']} | Mensagem: {evento['mensagem']}")
        print(alerta)

# =============================================================================
# Etapa 8: Visualização dos Eventos Processados
# - Exibe gráficos que ilustram a distribuição dos eventos por classificação ao longo do tempo.
# =============================================================================
def visualizar_eventos(df):
    # Exibe a contagem dos eventos por classificação
    contagem_classificacao = df['classificacao'].value_counts()
    print("Contagem de eventos por classificação:")
    print(contagem_classificacao)

    # Gráfico de barras para a classificação dos eventos
    plt.figure()
    sns.countplot(x='classificacao', data=df, order=["Normal", "Suspeito", "Crítico"])
    plt.title("Distribuição de Eventos por Classificação")
    plt.xlabel("Classificação")
    plt.ylabel("Contagem")
    plt.show()

    # Gráfico de linhas para visualização temporal dos eventos críticos (caso existam)
    df_criticos = df[df['classificacao'] == "Crítico"]
    if not df_criticos.empty:
        df_criticos_sorted = df_criticos.sort_values('timestamp')
        plt.figure()
        plt.plot(df_criticos_sorted['timestamp'], np.arange(len(df_criticos_sorted)), marker='o', linestyle='-')
        plt.title("Ocorrências de Eventos Críticos ao Longo do Tempo")
        plt.xlabel("Timestamp")
        plt.ylabel("Número de Eventos Críticos")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum evento crítico para visualizar.")

# =============================================================================
# Execução do Pipeline
# =============================================================================
if __name__ == "__main__":
    # Carrega os parâmetros de contexto do ambiente real (se houver arquivo config_contexto.json)
    config = ler_config_contexto()

    # Caminho para o arquivo de log real
    log_file_path = "logs_reais.txt"

    try:
        df_logs = ingestao_logs(log_file_path)
        print("Ingestão concluída. Exibindo as primeiras linhas do DataFrame:")
        print(df_logs.head())

        df_logs = extrai_features(df_logs)
        print("Pré-processamento e extração de features concluídos. Exemplo:")
        print(df_logs.head())

        # Detecta anomalias utilizando parâmetros do contexto real
        df_logs = detectar_anomalias(df_logs, config)
        
        # Classifica os eventos conforme os critérios definidos
        df_logs['classificacao'] = df_logs.apply(classifica_eventos, axis=1)
        print("Classificação de eventos realizada. Distribuição:")
        print(df_logs['classificacao'].value_counts())

        # Geração de alertas para eventos críticos
        print("Gerando alertas para eventos críticos:")
        gerar_alertas(df_logs)

        # Visualiza os eventos processados
        visualizar_eventos(df_logs)
        
    except FileNotFoundError:
        print(f"Arquivo de log não encontrado: {log_file_path}. Certifique-se de que o caminho está correto.")

