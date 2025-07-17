# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from datetime import datetime
import plotly.express as px

# Configurações Iniciais
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# Carregar dados
df = pd.read_csv('', 
                 parse_dates=['data_distribuicao', 'data_baixa']
                 )

# Verificar dados
'''print(df.head())
print(df.info())'''

# Tratamento das Colunas de Data
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['data_baixa'] = pd.to_datetime(df['data_baixa'], errors='coerce')


# 1) Análise de Crescimento Geral de Processos Sigilosos
# Criar coluna de ano por análise
df['ano_distribuicao'] = df['data_distribuicao'].dt.year

# Contagem de processos por ano
crescimento_geral = df.groupby('ano_distribuicao')['processo'].nunique().reset_index()
crescimento_geral.columns = ['Ano', 'Total de Processos Sigilosos']

# Converter ano para inteiro
crescimento_geral['Ano'] = crescimento_geral['Ano'].astype(int)

# Plotar gráfico em barras de crescimento
fig = px.bar(
    crescimento_geral, 
    x='Ano', 
    y='Total de Processos Sigilosos',
    title='<b>Crescimento Geral de Processos Sigilosos por Ano</b>',
    color='Total de Processos Sigilosos', 
    color_continuous_scale='Blues'
    )

# Formatando o Gráfico
fig.update_traces(
    hovertemplate='<b>Total:</b> %{y: ,d}<extra></extra>'
    )
fig.update_layout(separators=',.') # Formatar separador de milhar brasileiro

fig.update_xaxes(
    tickmode='array',
    tickvals=crescimento_geral['Ano'].unique(),
    ticktext=crescimento_geral['Ano'].astype(str)
    ) # Valor do ano, em inteiro, no eixo x

# Exibir gráfico interativo
fig.show()









