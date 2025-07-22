# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

# Configurações Iniciais
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# 1) Carregar e concatenar os dados dos processos judiciais da pasta uploads
# Verificar se a pasta 'uploads' existe
if not os.path.exists('uploads'):
    print("A pasta 'uploads' não existe. Certifique-se de que os arquivos estão na pasta correta.")
    exit()  

# Listar os arquivos CSV na pasta 'uploads'
arquivos_csv = glob.glob('uploads/processos_*.csv')

# Verificar se foram encontrados arquivos
if not arquivos_csv:
    print("Nenhum arquivo processos_*.csv encontrado na pasta 'uploads'.")
    exit()

# Carregar os arquivos CSV e concatenar em um único DataFrame
dfs = []
for arquivo in arquivos_csv:
    # Extrair o ano do nome do arquivo
    ano = int(arquivo.split('_')[-1].split('.')[0])
    df_ano = pd.read_csv(arquivo, sep=',', encoding='utf-8')
    df_ano['ano_arquivo'] = ano  # Adicionar coluna com o ano do arquivo
    dfs.append(df_ano)

df = pd.concat(dfs, ignore_index=True)

# 2) Tratamento dos Dados
# Verificar se as colunas necessárias existem
colunas_necessarias = ['processo', 'data_distribuicao', 'data_baixa', 'is_segredo_justica']
for coluna in colunas_necessarias:
    if coluna not in df.columns:
        print(f"Coluna '{coluna}' não encontrada nos dados. Verifique os arquivos CSV.")
        exit()

# Converter colunas de data
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['data_baixa'] = pd.to_datetime(df['data_baixa'], errors='coerce')
df['ano_distribuicao'] = df['data_distribuicao'].dt.year # Criar coluna de ano de distribuição