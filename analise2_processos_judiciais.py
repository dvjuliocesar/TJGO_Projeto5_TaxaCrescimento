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
# Listar os arquivos CSV na pasta 'uploads'
arquivos_csv = glob.glob('uploads/processos_*.csv')

# Carregar os arquivos CSV e concatenar em um único DataFrame
dfs = []
for arquivo in arquivos_csv:
    # Extrair o ano do nome do arquivo
    ano = int(arquivo.split('_')[-1].split('.')[0])
    df_ano = pd.read_csv(arquivo, sep=',', encoding='utf-8')
    df_ano['ano_arquivo'] = ano  # Adicionar coluna com o ano do arquivo
    dfs.append(df_ano)

df = pd.concat(dfs, ignore_index=True)

# 2) Tratamento dos Dados para Advogados (Processos Sigilosos)
# Converter colunas de data
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['data_baixa'] = pd.to_datetime(df['data_baixa'], errors='coerce')
df['ano_distribuicao'] = df['data_distribuicao'].dt.year # Criar coluna de ano de distribuição

# Expandir a coluna 'oab' que pode conter múltiplos advogados por processo
df_advogados = df.assign(oab=df['oab'].str.split(';').explode('oab'))
df_advogados['oab'] = df_advogados['oab'].str.strip()  # Remover espaços em branco

# Filtrar apenas processos sigilosos
df_sigiliosos = df_advogados[df_advogados['is_segredo_justica'] == 'True']

# 3) Análise por advogado
# Contar processos sigilosos por advogado por ano
crescimento_advogados = df_sigiliosos.groupby(['ano_distribuicao', 'oab'])['processo'].nunique().unstack().fillna(0)

# Calcular crescimento percentual para os top 10 advogados
top_advogados = df_sigiliosos['oab'].value_counts().nlargest(10).index
crescimento_top = crescimento_advogados[top_advogados].T

# Calcular a taxa de crescimento anual
crescimento_top_pct = crescimento_top.pct_change(axis=1) * 100

