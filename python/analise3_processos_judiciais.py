# --- BIBLIOTECAS NECESSÁRIAS ---
# Manipulação de dados e cálculos numéricos
import pandas as pd 
import numpy as np  
# Visualização estática (gráficos tradicionais)
import matplotlib.pyplot as plt 
import seaborn as sns            
# Manipulação de datas
from datetime import datetime  
# Visualização interativa e dinâmica
import plotly.express as px     
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
# Manipulação de arquivos e sistemas
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
# Tratamento das colunas
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['ano_distribuicao'] = df['data_distribuicao'].dt.year # Criar coluna de ano de distribuição
df['is_segredo_justica'] = df['is_segredo_justica'].astype(bool)

# 3) Analise de dados para Serventias
# Processar dados por ano

# Dataframe serventia 



def processar_dados(df, ano):
        df_ano = df[df['ano_distribuicao'] ==ano]
        sigilosos = df_ano[df_ano['is_segredo_justica']].groupby('serventia')['processo'].nunique()
        nao_sigilosos = df_ano[~df_ano['is_segredo_justica']].groupby('serventia')['processo'].nunique()

        total = pd.DataFrame({
            f'sigilosos_{ano}': sigilosos,
            f'nao_sigilosos_{ano}': nao_sigilosos
        }).fillna(0)

        total[f'total_{ano}'] = total[f'sigilosos_{ano}'] + total[f'nao_sigilosos_{ano}']

        total[f'proporcao_sigilosos_{ano}'] = (total[f'sigilosos_{ano}'] / total[f'total_{ano}'].round(4) * 100)

        total[f'proporcao_nao_sigilosos_{ano}'] = (total[f'nao_sigilosos_{ano}'] / total[f'total_{ano}'].round(4) * 100)

        return total

