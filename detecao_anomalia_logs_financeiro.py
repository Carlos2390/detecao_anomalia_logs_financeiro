# =============================================================================
# Projeto: Detecção Inteligente de Anomalias em Logs de TI para o Setor Financeiro
# Objetivo: Desenvolver um pipeline que simula logs, realiza ingestão, pré-processamento,
#          utiliza algoritmos de IA para detecção e classificação de anomalias,
#          gera alertas automatizados e visualiza os eventos processados.
# =============================================================================

# =============================================================================
# Etapa 1: Importação das bibliotecas necessárias
# =============================================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Configurações para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# =============================================================================
# Etapa 2: Simulação e Coleta de Logs
# - Criação de um arquivo de log fictício com informações simuladas.
# - Cada linha do log contém data, hora, IP, nível de acesso e uma mensagem.
# =============================================================================

# Função para simular dados de log
def simula_logs(n=500):
    """Gera uma lista de strings simulando linhas de log."""
    logs = []
    base_time = datetime.now() - timedelta(hours=1)  # logs dos últimos 60 minutos
    niveis = ['INFO', 'WARN', 'ERROR']
    mensagens = ['Acesso permitido', 'Acesso negado', 'Transação realizada',
                 'Falha de autenticação', 'Tentativa de invasão']

    for i in range(n):
        # Incremento de tempo aleatório
        timestamp = base_time + timedelta(seconds=np.random.randint(0, 3600))
        # Formatação da data/hora
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        # Geração aleatória de IP
        ip = ".".join(str(np.random.randint(1, 255)) for _ in range(4))
        # Seleção aleatória do nível e mensagem
        nivel = np.random.choice(niveis, p=[0.7, 0.2, 0.1])
        mensagem = np.random.choice(mensagens)

        # Em casos de ERROR, simula uma mensagem crítica (anomalia potencial)
        if nivel == 'ERROR':
            mensagem += " - Possível ataque detectado"

        log_line = f"{ts_str} | {ip} | {nivel} | {mensagem}"
        logs.append(log_line)
    return logs

# Gerar e salvar logs simulados em um arquivo (simulação de coleta de logs)
logs_simulados = simula_logs(n=500)
with open("logs_simulados.txt", "w", encoding="utf-8") as f:
    for linha in logs_simulados:
        f.write(linha + "\n")

print("Arquivo de logs simulados gerado com sucesso!")

# =============================================================================
# Etapa 3: Ingestão dos Dados
# - Leitura do arquivo de log fictício.
# - Estruturação dos dados utilizando pandas.
# =============================================================================

# Função para ler e estruturar os logs
def ingestao_logs(file_path):
    """Lê o arquivo de logs e retorna um DataFrame estruturado."""
    # Lista para armazenar dicionários com os campos extraídos
    data = []
    # Expressão regular para extrair os campos do log
    padrao = r"^(.*?) \| (.*?) \| (.*?) \| (.*)$"

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
    # Conversão para DataFrame
    df = pd.DataFrame(data)
    # Converter a coluna de timestamp para datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    return df

# Ingestão dos dados
df_logs = ingestao_logs("logs_simulados.txt")
print("Ingestão concluída. Exibindo as primeiras linhas do DataFrame:")
print(df_logs.head())

# =============================================================================
# Etapa 4: Pré-processamento dos Dados
# - Normalização dos campos (por exemplo, datas para um mesmo fuso horário).
# - Extração de informações relevantes: exemplo, extração de IP, identificação de padrões na mensagem.
# =============================================================================

# Normalização já realizada na conversão do timestamp.
# Exemplo de extração: verificar se a mensagem contém indicação de ataque

def extrai_features(df):
    """Adiciona colunas de features extraídas dos logs."""
    # Flag para indicar se a mensagem sugere comportamento anômalo
    df['possivel_ataque'] = df['mensagem'].apply(lambda x: 1 if "ataque" in x.lower() else 0)
    # Extração de hora para análises temporais
    df['hora'] = df['timestamp'].dt.hour
    return df

df_logs = extrai_features(df_logs)
print("Pré-processamento e extração de features concluídos. Exemplo:")
print(df_logs.head())

# =============================================================================
# Etapa 5: Análise com IA – Detecção de Anomalias
# - Utilização de um algoritmo de machine learning (Isolation Forest) para detectar acessos anômalos.
# - O modelo é treinado considerando as features extraídas (ex.: hora, possivel_ataque).
# =============================================================================

# Seleção das features para o modelo
features = df_logs[['hora', 'possivel_ataque']]

# Criação e treinamento do modelo de detecção de anomalias
modelo_if = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
modelo_if.fit(features)

# Predição de anomalias: -1 para anomalia, 1 para normal
df_logs['anomaly_score'] = modelo_if.decision_function(features)
df_logs['anomaly'] = modelo_if.predict(features)

print("Detecção de anomalias concluída. Contagem de rótulos (-1: anomalia, 1: normal):")
print(df_logs['anomaly'].value_counts())

# =============================================================================
# Etapa 6: Classificação dos Eventos
# - Classifica os eventos em Normais, Suspeitos ou Críticos.
# - Critérios:
#   * Normais: logs marcados como 1 pelo modelo e sem indicação de ataque.
#   * Suspeitos: logs normais, mas com indicação de possível ataque.
#   * Críticos: logs marcados como anomalia (-1) e com indicação de ataque.
# =============================================================================

def classifica_eventos(row):
    """Classifica cada log com base na predição do modelo e na presença de indicadores de ataque."""
    if row['anomaly'] == -1 and row['possivel_ataque'] == 1:
        return "Crítico"
    elif row['anomaly'] == 1 and row['possivel_ataque'] == 1:
        return "Suspeito"
    else:
        return "Normal"

df_logs['classificacao'] = df_logs.apply(classifica_eventos, axis=1)
print("Classificação de eventos realizada. Distribuição:")
print(df_logs['classificacao'].value_counts())

# =============================================================================
# Etapa 7: Resposta Automatizada – Geração de Alertas
# - Para cada log classificado como Crítico, gera um alerta (neste exemplo, um print).
# - Em um ambiente real, este módulo integraria com APIs de notificação (e.g., Slack, Webhook).
# =============================================================================

def gerar_alertas(df):
    """Gera alertas para eventos críticos."""
    eventos_criticos = df[df['classificacao'] == "Crítico"]
    for idx, evento in eventos_criticos.iterrows():
        alerta = (f"ALERTA CRÍTICO: {evento['timestamp']} | IP: {evento['ip']} | "
                  f"Nível: {evento['nivel']} | Mensagem: {evento['mensagem']}")
        # Neste exemplo, o alerta é impresso; em produção, poderia ser enviado por email ou outro meio.
        print(alerta)

print("Gerando alertas para eventos críticos:")
gerar_alertas(df_logs)

# =============================================================================
# Etapa 8: Visualização dos Eventos Processados
# - Exibe gráficos que ilustram a distribuição dos eventos por classificação ao longo do tempo.
# =============================================================================

# Contagem de eventos por classificação
contagem_classificacao = df_logs['classificacao'].value_counts()
print("Contagem de eventos por classificação:")
print(contagem_classificacao)

# Gráfico de barras para a classificação dos eventos
plt.figure()
sns.countplot(x='classificacao', data=df_logs, order=["Normal", "Suspeito", "Crítico"])
plt.title("Distribuição de Eventos por Classificação")
plt.xlabel("Classificação")
plt.ylabel("Contagem")
plt.show()

# Gráfico de linhas para visualização temporal dos eventos críticos (se houver)
df_criticos = df_logs[df_logs['classificacao'] == "Crítico"]
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