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
# Manipulação de arquivos e sistemas
import glob, os, re

arquivos_csv = glob.glob(os.path.join('uploads', 'processos_*.csv'))
if not arquivos_csv:
    raise FileNotFoundError("Nenhum CSV encontrado no padrão 'uploads/processos_*.csv'.")

dfs = []
for arquivo in arquivos_csv:
    base = os.path.basename(arquivo)
    m = re.search(r'processos_(\d{4})\.csv$', base)
    ano = int(m.group(1)) if m else None

    df_ano = pd.read_csv(
        arquivo,
        sep=',',
        encoding='utf-8',
    )
    if ano is not None:
        df_ano['ano_arquivo'] = ano
    dfs.append(df_ano)

df = pd.concat(dfs, ignore_index=True)

# 2) Tratamento dos Dados para Serventias (Processos Sigilosos)
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['ano_distribuicao'] = df['data_distribuicao'].dt.year

# Converta para string, padronize e MAPEIE para booleano
tmp = df['is_segredo_justica'].astype(str).str.strip().str.lower()
df['is_segredo_justica'] = tmp.map({
    'true': True, 'false': False,
    '1': True, '0': False,
    'sim': True, 'não': False, 'nao': False
})

df['is_segredo_justica'] = df['is_segredo_justica'].fillna(False).astype(bool)

# Limpeza da serventia
df['serventia'] = df['serventia'].astype(str).str.strip()

# DataFrame Serventia
df_serventia = (
    df.loc[df['serventia'].ne(''), ['serventia', 'processo', 'ano_distribuicao', 'is_segredo_justica']]
      .dropna(subset=['ano_distribuicao'])
      .copy()
)
# Agrupar contando os processos únicos
contagem = df.groupby(['ano_distribuicao', 'serventia','is_segredo_justica'])['processo'].nunique().reset_index()
df['serventia'] = df['serventia'].astype(str).str.strip()

# Processar dados por ano
def processar_dados(df_base, ano: int) -> pd.DataFrame:
    df_ano = df_base[df_base['ano_distribuicao'] == ano]

    sigilosos = (df_ano[df_ano['is_segredo_justica']]
                 .groupby('serventia')['processo'].nunique())
    nao_sigilosos = (df_ano[~df_ano['is_segredo_justica']]
                     .groupby('serventia')['processo'].nunique())

    total = pd.DataFrame({
        f'sigilosos_{ano}': sigilosos,
        f'nao_sigilosos_{ano}': nao_sigilosos
    }).fillna(0)

    total[f'total_{ano}'] = total[f'sigilosos_{ano}'] + total[f'nao_sigilosos_{ano}']

    denom = total[f'total_{ano}']
    total[f'proporcao_sigilosos_{ano}'] = np.where(
        denom > 0, (total[f'sigilosos_{ano}'] / denom) * 100, 0.0
    ).round(4)

    total[f'proporcao_nao_sigilosos_{ano}'] = np.where(
        denom > 0, (total[f'nao_sigilosos_{ano}'] / denom) * 100, 0.0
    ).round(4)

    return total
# Processar dados para cada ano
dados_2022 = processar_dados(df_serventia, 2022)
dados_2023 = processar_dados(df_serventia, 2023)
dados_2024 = processar_dados(df_serventia, 2024)

# Concatenar os dados
tabela_final = pd.concat([dados_2022, dados_2023, dados_2024], axis=1).fillna(0)
tabela_final = tabela_final.reset_index()
# Formatar valores para exibição
for ano in [2022, 2023, 2024]:
    for col in [f'sigilosos_{ano}', f'nao_sigilosos_{ano}', f'total_{ano}']:
        tabela_final[col] = tabela_final[col].astype(int)

# --- TABELA DE PROPORÇÕES COM VARIAÇÃO TOTAL E MÉDIA ---
tabela_proporcoes = tabela_final[['serventia'] + 
                        [f'sigilosos_{ano}' for ano in [2022, 2023, 2024]].copy() +
                        [f'nao_sigilosos_{ano}' for ano in [2022, 2023, 2024]].copy() +
                        [f'total_{ano}' for ano in [2022, 2023, 2024]].copy() +
                        [f'proporcao_sigilosos_{ano}' for ano in [2022, 2023, 2024]].copy() +
                        [f'proporcao_nao_sigilosos_{ano}' for ano in [2022, 2023, 2024]]].copy()
                
# Converter proporções para float (remover % temporariamente)
for ano in [2022, 2023, 2024]:
    tabela_proporcoes[f'proporcao_sigilosos_{ano}'] = tabela_proporcoes[f'proporcao_sigilosos_{ano}'].astype(float)
    tabela_proporcoes[f'proporcao_nao_sigilosos_{ano}'] = tabela_proporcoes[f'proporcao_nao_sigilosos_{ano}'].astype(float)

# Calcular Variação de Sigilosos Total (2022 → 2024)
tabela_proporcoes['variacao_total_sigilosos'] = (
    (tabela_proporcoes['proporcao_sigilosos_2023'] - tabela_proporcoes['proporcao_sigilosos_2022']) +
    (tabela_proporcoes['proporcao_sigilosos_2024'] - tabela_proporcoes['proporcao_sigilosos_2023'])
)

# Calcular Variação de Não Sigilosos Total (2022 → 2024)
tabela_proporcoes['variacao_total_nao_sigilosos'] = (
    (tabela_proporcoes['proporcao_nao_sigilosos_2023'] - tabela_proporcoes['proporcao_nao_sigilosos_2022']) +
    (tabela_proporcoes['proporcao_nao_sigilosos_2024'] - tabela_proporcoes['proporcao_nao_sigilosos_2023'])
)

# Calcular Proporção Média de Sigilosos (2022-2024)
tabela_proporcoes['proporcao_media_sigilosos'] = tabela_proporcoes[
    ['proporcao_sigilosos_2022', 'proporcao_sigilosos_2023', 'proporcao_sigilosos_2024']
].mean(axis=1)

# Calcular a proporção Média de Não Sigilosos (2022-2024)
tabela_proporcoes['proporcao_media_nao_sigilosos'] = tabela_proporcoes[
    ['proporcao_nao_sigilosos_2022', 'proporcao_nao_sigilosos_2023', 'proporcao_nao_sigilosos_2024']
].mean(axis=1)
    
# Calcular total de Sigilosos
tabela_proporcoes['total_sigilosos'] = (
    tabela_proporcoes['sigilosos_2022'] +
    tabela_proporcoes['sigilosos_2023'] +
    tabela_proporcoes['sigilosos_2024']
)

# Calcular total de Não Sigilosos
tabela_proporcoes['total_nao_sigilosos'] = (
    tabela_proporcoes['nao_sigilosos_2022'] +
    tabela_proporcoes['nao_sigilosos_2023'] +
    tabela_proporcoes['nao_sigilosos_2024']
)
    
# Calcular total geral de processos
tabela_proporcoes['total_processos'] = (
    tabela_proporcoes['total_sigilosos'] +
    tabela_proporcoes['total_nao_sigilosos']
)

# Formatar para exibição (padrão brasileiro)
tabela_proporcoes_formatada = tabela_proporcoes.fillna(0).copy()
for ano in [2022, 2023, 2024]:
    tabela_proporcoes_formatada[f'proporcao_sigilosos_{ano}'] = tabela_proporcoes_formatada[
        f'proporcao_sigilosos_{ano}'
    ].apply(lambda x: f"{x:.2f}%".replace('.', ','))
    
for ano in [2022, 2023, 2024]:
    tabela_proporcoes_formatada[f'proporcao_nao_sigilosos_{ano}'] = tabela_proporcoes_formatada[
        f'proporcao_nao_sigilosos_{ano}'
    ].apply(lambda x: f"{x:.2f}%".replace('.', ','))

tabela_proporcoes_formatada['variacao_total_sigilosos'] = tabela_proporcoes_formatada['variacao_total_sigilosos'].apply(
    lambda x: f"{x:+.2f}%".replace('.', ',') 
)

tabela_proporcoes_formatada['variacao_total_nao_sigilosos'] = tabela_proporcoes_formatada['variacao_total_nao_sigilosos'].apply(
    lambda x: f"{x:+.2f}%".replace('.', ',')
) 

tabela_proporcoes_formatada['proporcao_media_sigilosos'] = tabela_proporcoes_formatada['proporcao_media_sigilosos'].apply(
    lambda x: f"{x:.2f}%".replace('.', ',')
)

tabela_proporcoes_formatada['proporcao_media_nao_sigilosos'] = tabela_proporcoes_formatada['proporcao_media_nao_sigilosos'].apply(
    lambda x: f"{x:.2f}%".replace('.', ',')
)

# Função para cores alternadas (zebrado)
def get_row_colors(n):
    return ['lavender' if i % 2 == 0 else 'white' for i in range(n)]

num_rows = len(tabela_proporcoes_formatada)
num_cols = 20  # ajuste se mudar as colunas
row_colors = get_row_colors(num_rows)
fill_color = [row_colors] * num_cols 

# Tabela Plotly das Proporções com Variação e Média
fig_proporcoes = go.Figure(data=[go.Table(
    header=dict(
        values=[
            'Serventia',
            'Sigilosos 2022',
            'Sigilosos 2023',
            'Sigilosos 2024',
            'Total Sigilosos',
            'Proporção de Sigilosos 2022', 
            'Proporção de Sigilosos 2023', 
            'Proporção de Sigilosos 2024', 
            'Variação Total de Sigilosos', 
            'Proporção Média de Sigilosos',
            'Não Sigilosos 2022',
            'Não Sigilosos 2023',
            'Não Sigilosos 2024',
            'Total Não Sigilosos',
            'Proporção de Não Sigilosos 2022',
            'Proporção de Não Sigilosos 2023',
            'Proporção de Não Sigilosos 2024',
            'Variação Total de Não Sigilosos',
            'Proporção Média de Não Sigilosos',
            'Total de Processos'  
        ],
        fill_color='#203864',
        font=dict(color='white', size=12),
        align='left',
        line_color='darkslategray'
    ),
    cells=dict(
        values=[
            tabela_proporcoes_formatada['serventia'],
            tabela_proporcoes_formatada['sigilosos_2022'],
            tabela_proporcoes_formatada['sigilosos_2023'],
            tabela_proporcoes_formatada['sigilosos_2024'],
            tabela_proporcoes_formatada['total_sigilosos'],
            tabela_proporcoes_formatada['proporcao_sigilosos_2022'],
            tabela_proporcoes_formatada['proporcao_sigilosos_2023'],
            tabela_proporcoes_formatada['proporcao_sigilosos_2024'],
            tabela_proporcoes_formatada['variacao_total_sigilosos'],
            tabela_proporcoes_formatada['proporcao_media_sigilosos'],
            tabela_proporcoes_formatada['nao_sigilosos_2022'],
            tabela_proporcoes_formatada['nao_sigilosos_2023'],
            tabela_proporcoes_formatada['nao_sigilosos_2024'],
            tabela_proporcoes_formatada['total_nao_sigilosos'],
            tabela_proporcoes_formatada['proporcao_nao_sigilosos_2022'],
            tabela_proporcoes_formatada['proporcao_nao_sigilosos_2023'],
            tabela_proporcoes_formatada['proporcao_nao_sigilosos_2024'],
            tabela_proporcoes_formatada['variacao_total_nao_sigilosos'],
            tabela_proporcoes_formatada['proporcao_media_nao_sigilosos'],
            tabela_proporcoes_formatada['total_processos']
        ],
        fill_color=fill_color,
        align='left',
        font=dict(color='black', size=11),
        line_color='darkslategray'
    )
)])

fig_proporcoes.update_layout(
    title='<b>Proporção de Casos Sigilosos por Serventia (2022-2024)</b><br>'
            '<i>Ordenado por Variação Total</i>',
    title_x=0.5,
    margin=dict(l=20, r=20, t=100, b=20),
    height=900,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# --- GRÁFICO DE DISPERSÃO ESTRATÉGICO (SIGILOSOS vs. NÃO SIGILOSOS) ---

# 1. Gráfico de Dispersão
# Dataframe para o gráfico de disperção
tabela_dispersao = pd.merge(
    tabela_proporcoes,
    tabela_final[['serventia', 'sigilosos_2022', 'sigilosos_2023', 'sigilosos_2024',
                    'nao_sigilosos_2022', 'nao_sigilosos_2023', 'nao_sigilosos_2024']],
    on='serventia',
    how='left'
)
# Criar gráfico
colunas_hover = [
    'variacao_total_sigilosos',  # -> customdata[0]
    'sigilosos_2022_x',          # -> customdata[1]
    'sigilosos_2023_x',          # -> customdata[2]
    'sigilosos_2024_x',          # -> customdata[3]
    'nao_sigilosos_2022_x',      # -> customdata[4]
    'nao_sigilosos_2023_x',      # -> customdata[5]
    'nao_sigilosos_2024_x'       # -> customdata[6]
]
fig_dispersao = px.scatter(
    tabela_dispersao,
    x='proporcao_media_sigilosos',
    y='variacao_total_sigilosos',
    title='<b>Análise Estratégica Comparativa: Casos Sigilosos</b>',
    labels={
        'proporcao_media_sigilosos': 'Proporção Média de Casos Sigilosos (%)',
        'variacao_total_sigilosos': 'Variação da Proporção de Casos Sigilosos (2024 - 2022)'
    },
    hover_name='serventia',
    custom_data=colunas_hover
)

# 3. Adicionar as linhas de referência para os quadrantes de casos sigilosos
media_proporcao_sigilosos = tabela_proporcoes['proporcao_media_sigilosos'].mean()
fig_dispersao.add_hline(y=0, line_dash="dash", line_color="grey")
fig_dispersao.add_vline(x=media_proporcao_sigilosos, line_dash="dash", line_color="grey")

# 4. Adicionar anotações para explicar os quadrantes
fig_dispersao.add_annotation(x=95, y=tabela_proporcoes['variacao_total_sigilosos'].max()*0.9, text="<b>Especialistas em Expansão</b>", showarrow=False, bgcolor="#e3f2fd", xanchor='right')
fig_dispersao.add_annotation(x=5, y=tabela_proporcoes['variacao_total_sigilosos'].max()*0.9, text="<b>Novos Focos de Atuação</b>", showarrow=False, bgcolor="#e8f5e9", xanchor='left')
fig_dispersao.add_annotation(x=5, y=tabela_proporcoes['variacao_total_sigilosos'].min()*0.9, text="<b>Fora do Foco</b>", showarrow=False, bgcolor="#ffebee", xanchor='left')
fig_dispersao.add_annotation(x=95, y=tabela_proporcoes['variacao_total_sigilosos'].min()*0.9, text="<b>Especialistas em Transição</b>", showarrow=False, bgcolor="#fff3e0", xanchor='right')

# --- ATUALIZAÇÕES GERAIS DE LAYOUT ---

# 8. Atualizar os títulos dos eixos e o título principal
fig_dispersao.update_xaxes(title_text="Proporção Média de Casos Sigilosos (%)")
fig_dispersao.update_yaxes(title_text="Variação da Proporção (2024 - 2022)")

fig_dispersao.update_traces(
    marker=dict(size=10, color='#203864'),
    hovertemplate="<br>".join([
        "<b>Serventia:</b> %{hovertext}",
        "<b>Variação Total:</b> %{customdata[0]:.2f}%",
        "<b>--- <b>Contagem de Casos</b> ---",
        "<b>Sigilosos 2022:</b> %{customdata[1]}",
        "<b>Sigilosos 2023:</b> %{customdata[2]}",
        "<b>Sigilosos 2024:</b> %{customdata[3]}",
        "<b>Não Sigilosos 2022:</b> %{customdata[4]}",
        "<b>Não Sigilosos 2023:</b> %{customdata[5]}",
        "<b>Não Sigilosos 2024:</b> %{customdata[6]}",
        "<extra></extra>"
    ])
)

fig_dispersao.update_layout(
    title_text='<b>Análise Estratégica Comparativa: Foco em Casos Sigilosos VS Não Sigilosos</b>',
    title_x=0.5,
    height=900,
    showlegend=False # A legenda não é necessária, pois os títulos dos subplots explicam
)
fig_dispersao.update_xaxes(ticksuffix="%")
fig_dispersao.update_yaxes(ticksuffix="%")

# Exibição dos resultados
fig_proporcoes.show()
fig_dispersao.show()

