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
    try:
        # Extrair o ano do nome do arquivo
        ano = int(arquivo.split('_')[-1].split('.')[0])
        df_ano = pd.read_csv(arquivo, sep=',', encoding='utf-8')
        df_ano['ano_arquivo'] = ano  # Adicionar coluna com o ano do arquivo
        dfs.append(df_ano)
    except Exception as e:
        print(f"Erro ao carregar o arquivo {arquivo}: {e}")

if not dfs:
    print("Nenhum arquivo foi carregado com sucesso.")
    exit()

df = pd.concat(dfs, ignore_index=True)


# 2) Tratamento dos Dados
# Converter colunas de data
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['data_baixa'] = pd.to_datetime(df['data_baixa'], errors='coerce')

# Criar coluna de ano de distribuição
df['ano_distribuicao'] = df['data_distribuicao'].dt.year #


# 3) Análise Comparativa entre de Processos Sigilosos e Não Sigilosos
# Agrupar por ano e status de sigilo
analise_sigilo = df.groupby(
    ['ano_distribuicao', 'is_segredo_justica']).nunique().unstack().reset_index()
analise_sigilo.columns = ['Ano', 'Nao_Sigilosos', 'Sigilosos']

# Calcular totais e proporção de processos sigilosos
analise_sigilo['Total_Processos'] = analise_sigilo['Nao_Sigilosos'] + analise_sigilo['Sigilosos']
analise_sigilo['Proporcao_Sigilosos'] = analise_sigilo['Sigilosos'] / analise_sigilo['Total_Processos']*100

# converter ano para inteiro
analise_sigilo['Ano'] = analise_sigilo['Ano'].astype(int)

# 4) Tabela Resumo
print("\nTabela Resumo Anual:")
for index, row in analise_sigilo.iterrows():
    print(f"\n{row['Ano']}:")
    print(f"Casos novos: {row['Total_Processos']: ,d}")
    print(f"Casos novos sigilosos: {row['Sigilosos']: ,d}")
    print(f"Proporção sigilosos: {row['Proporcao_Sigilosos']:.2f}%")

'''
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
'''








