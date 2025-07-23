'''
Análise de Processos Judiciais Sigilosos e Não Sigilosos:
- Este script analisa dados de processos judiciais, comparando a quantidade de processos sigilosos
e não sigilosos ao longo dos anos. Ele gera tabelas e gráficos para visualização dos dados.
'''

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

# 2) Tratamento dos Dados
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
'''
print("\nTabela Resumo Anual:")
for index, row in analise_sigilo.iterrows():
    print(f"\n{int(row['Ano'])}:") # Convertendo para inteiro antes de formatar
    print(f"Casos novos: {row['Nao_Sigilosos']:,.0f}".replace(",", "."))  # Separador de milhar como ponto
    print(f"Casos sigilosos: {row['Sigilosos']:,.0f}".replace(",", "."))
    print(f"Proporção sigilosos: {row['Proporcao_Sigilosos']:.2f}".replace(".", ",") + "%")
'''

# Tabela como plotly table
tabela_resumo = analise_sigilo[['Ano', 'Total_Processos', 'Sigilosos', 'Proporcao_Sigilosos']].copy()
tabela_resumo.columns = ['Ano', 'Total', 'Sigilosos', '% Sigilosos']
tabela_resumo['Ano'] = tabela_resumo['Ano'].astype(str)  # Converter ano para string para exibição
tabela_resumo['Total'] = tabela_resumo['Total'].apply(lambda x: f"{x:,.0f}".replace(",", ".")) # Formatar decimal brasileiro
tabela_resumo['Sigilosos'] = tabela_resumo['Sigilosos'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
tabela_resumo['% Sigilosos'] = tabela_resumo['% Sigilosos'].apply(lambda x: f"{x:.2f}".replace(".", ",")) + "%"  # Formatar porcentagem


fig_table = go.Figure(data=[go.Table(
    header=dict(values=tabela_resumo.columns,  
                fill_color='paleturquoise',
                align='left'),  
    cells=dict(values=[tabela_resumo[col] for col in tabela_resumo.columns],
               fill_color='lavender',  
                align='left'))])  

fig_table.update_layout(
    title_text='<b>Tabela de Resumo Anual dos Processos Sigilosos e Não Sigilosos</b>', 
    title_x=0.5
    )    

# Cor do cabeçalho hexadecimal
fig_table.update_traces(
    header=dict(
        fill_color='#203864',  # Cor do cabeçalho em hexadecimal
        font=dict(color='white', size=16)  # Cor e tamanho da fonte do cabeçalho
    ),
    cells=dict(
        fill_color='lavender',  # Cor das células em hexadecimal
        font=dict(color='black', size=14)  # Cor e tamanho da fonte das células
    )
)

# Configurar o layout da tabela 
fig_table.update_layout(
    height=400,  # Altura da tabela
    width=800,   # Largura da tabela
    margin=dict(l=20, r=20, t=30, b=20),
    title_font=dict(size=20, color='black'),  # Tamanho e cor do título
    font=dict(size=14, color='black')  # Tamanho e cor da fonte
)

# Exibir tabela
fig_table.show()  

# 5) Gráficos
# 5.1) Plotar gráfico em barras de comparação entre processos sigilosos e não sigilosos
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

# 5.2) Gráfico de proporção de processos sigilosos
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
    2022: "#4375D3",
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

# Exibir gráficos
fig.show()
fig2.show()









