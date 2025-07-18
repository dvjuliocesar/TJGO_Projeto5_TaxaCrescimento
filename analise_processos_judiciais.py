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


# 3) Análise Comparativa entre de Processos Sigilosos e Não Sigilosos
# Garantir que temos apenas True/False
df['is_segredo_justica'] = df['is_segredo_justica'].astype(str)
df = df[df['is_segredo_justica'].isin(['True', 'False'])]

# Agrupar contando os processos únicos
contagem_sigilo = df.groupby(['ano_distribuicao', 'is_segredo_justica'])['processo'].nunique().reset_index()

# Pivotar a tabela
analise_sigilo = contagem_sigilo.pivot(
    index='ano_distribuicao', 
    columns='is_segredo_justica', 
    values='processo'
).reset_index()

# Renomear colunas
analise_sigilo.columns = ['Ano', 'Nao_Sigilosos', 'Sigilosos']

# Preencher com possíveis valores nulos
analise_sigilo = analise_sigilo.fillna(0)

# Calcular totais e proporção de processos sigilosos
analise_sigilo['Total_Processos'] = analise_sigilo['Nao_Sigilosos'] + analise_sigilo['Sigilosos']
analise_sigilo['Proporcao_Sigilosos'] = analise_sigilo['Sigilosos'] / analise_sigilo['Total_Processos'] * 100

# converter ano para inteiro
analise_sigilo['Ano'] = analise_sigilo['Ano'].astype(int)

# 4) Tabela Resumo
print("\nTabela Resumo Anual:")
for index, row in analise_sigilo.iterrows():
    print(f"\n{int(row['Ano'])}:") # Convertendo para inteiro antes de formatar
    print(f"Casos novos: {row['Nao_Sigilosos']:,.0f}".replace(",", "."))  # Separador de milhar como ponto
    print(f"Casos sigilosos: {row['Sigilosos']:,.0f}".replace(",", "."))
    print(f"Proporção sigilosos: {row['Proporcao_Sigilosos']:.2f}".replace(".", ",") + "%")

# 5) Gráfico em Barras Distintas da Análise Comparativa
# Plotar gráfico em barras de comparação entre processos sigilosos e não sigilosos
fig = px.bar(
    analise_sigilo,
    x='Ano',
    y=['Nao_Sigilosos', 'Sigilosos'],  
    title='<b>Comparativo Anual do Número de Processos Sigilosos e Não Sigilosos</b>',
    labels={'value': 'Total de Processos', 'variable': 'Tipo de Processo'},
    color_discrete_sequence=["#4375D3", '#203864'],  # Cores para os tipos de processo
    barmode='group'
    )

# Formatando o Gráfico
#
fig.for_each_trace(
    lambda trace: trace.update(hovertemplate='<b>Total de Casos Novos:</b> %{y:,.0f}<extra></extra>') 
    if trace.name == 'Nao_Sigilosos' 
    else trace.update(hovertemplate='<b>Total de Casos Sigilosos:</b> %{y:,.0f}<extra></extra>')
)

fig.update_layout(
    separators=',.', # Formatar separador de milhar brasileiro
    title_x=0.5, 
    legend_title_text='Tipo de Processo',
    xaxis_title='Ano',
    yaxis_title='Total de Processos'
    ) 

fig.update_xaxes(
    tickmode='array',
    tickvals=analise_sigilo['Ano'].unique(),
    ticktext=analise_sigilo['Ano'].astype(str)
    ) # Valor do ano, em inteiro, no eixo x

# Gráfico de proporção de processos sigilosos


# Configurações adicionais
fig2 = px.bar(
    analise_sigilo,
    x='Ano',
    y='Proporcao_Sigilosos',
    title='<b>Proporção de Processos Sigilosos</b>',
    text='Proporcao_Sigilosos',
    labels={'Proporcao_Sigilosos': 'Proporção de Sigilosos (%)', 'Ano': 'Ano'},
)

# Definir cores personalizadas para cada barra
cores_personalizadas = {
    2022: "#709EE3",
    2023: "#494D94", 
    2024: "#203864"
}

# Aplicar as cores manualmente
fig2.update_traces(
    marker_color=[cores_personalizadas[ano] for ano in analise_sigilo['Ano']],
    texttemplate='%{text:.2f}%',
    textposition='outside',
    hovertemplate='<b>Proporção de Processos Sigilosos:</b> %{y:.2f}%<extra></extra>'
)

fig2.update_layout(
    legend_title_text='Ano',
    xaxis_title='Ano',
    yaxis_title='Proporção de Processos Sigilosos (%)',
    title_x=0.5
    )

fig2.update_xaxes(
    tickmode='array',
    tickvals=analise_sigilo['Ano'].unique(),
    ticktext=analise_sigilo['Ano'].astype(str)
    ) # Valor do ano, em inteiro, no eixo x

# Exibir gráfico interativo
fig.show()
fig2.show()










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








