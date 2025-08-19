'''Análise de Processos Judiciais Sigilosos - Área de Ação Geral:
- Este script analisa dados de processos judiciais, focando no número de processos e
na proporção de processos sigilosos por Área de Ação. Ele gera tabelas e gráficos para visualização dos dados.'''

# --- BIBLIOTECAS NECESSÁRIAS ---
import pandas as pd 
import numpy as np  
import plotly.express as px     
import plotly.graph_objects as go  
import glob, os, re

arquivos_csv = glob.glob(os.path.join('uploads', 'processo_*.csv'))
if not arquivos_csv:
    raise FileNotFoundError("Nenhum CSV encontrado no padrão 'uploads/processo_*.csv'.")

dfs = []
for arquivo in arquivos_csv:
    base = os.path.basename(arquivo)
    m = re.search(r'processo_(\d{4})\.csv$', base)
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

# 2) Tratamento dos Dados para Área de Ação (Processos Sigilosos)
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

# Limpeza da área de ação
df['nome_area_acao'] = df['nome_area_acao'].astype(str).str.strip()
df = df.reset_index(drop=True)

# DataFrame Área de Ação (sem comarca)
df_area_acao = (
    df.loc[df['nome_area_acao'].ne(''),
           ['nome_area_acao', 'processo', 'ano_distribuicao', 'is_segredo_justica']]
      .dropna(subset=['ano_distribuicao'])
      .copy()
)

# (Opcional, mantido por estrutura) Contagem única por ano/área/tipo
contagem = (
    df.groupby(['ano_distribuicao', 'nome_area_acao', 'is_segredo_justica'])['processo']
      .nunique()
      .reset_index()
)

# Processar dados por ano (sem comarca)
def processar_dados(df_base, ano: int) -> pd.DataFrame:
    cols_need = ['nome_area_acao', 'processo', 'is_segredo_justica', 'ano_distribuicao']
    missing = [c for c in cols_need if c not in df_base.columns]
    if missing:
        raise KeyError(f"Colunas ausentes em df_base: {missing}")

    df_ano = df_base.loc[df_base['ano_distribuicao'] == ano,
                         ['nome_area_acao', 'processo', 'is_segredo_justica']].copy()

    if df_ano.empty:
        idx = pd.Index([], name='nome_area_acao')
        out = pd.DataFrame(index=idx)
        out[f'sigilosos_{ano}'] = []
        out[f'nao_sigilosos_{ano}'] = []
        out[f'total_{ano}'] = []
        out[f'proporcao_sigilosos_{ano}'] = []
        out[f'proporcao_nao_sigilosos_{ano}'] = []
        return out

    df_ano['nome_area_acao'] = df_ano['nome_area_acao'].astype(str).str.strip()

    df_ano['tipo'] = np.where(df_ano['is_segredo_justica'], 'sigilosos', 'nao_sigilosos')
    df_ano = df_ano.drop_duplicates(subset=['nome_area_acao', 'processo', 'tipo'])

    grp = (
        df_ano.groupby(['nome_area_acao', 'tipo'], as_index=False)['processo']
              .nunique()
    )

    pv = grp.pivot_table(index=['nome_area_acao'],
                         columns='tipo',
                         values='processo',
                         aggfunc='sum',
                         fill_value=0)

    out = pd.DataFrame(index=pv.index)
    out[f'sigilosos_{ano}'] = pv['sigilosos'] if 'sigilosos' in pv.columns else pd.Series(0, index=pv.index)
    out[f'nao_sigilosos_{ano}'] = pv['nao_sigilosos'] if 'nao_sigilosos' in pv.columns else pd.Series(0, index=pv.index)
    out[f'total_{ano}'] = out[f'sigilosos_{ano}'] + out[f'nao_sigilosos_{ano}']

    denom = out[f'total_{ano}'].replace(0, np.nan)
    out[f'proporcao_sigilosos_{ano}'] = ((out[f'sigilosos_{ano}'] / denom) * 100).fillna(0.0).round(4)
    out[f'proporcao_nao_sigilosos_{ano}'] = ((out[f'nao_sigilosos_{ano}'] / denom) * 100).fillna(0.0).round(4)

    return out 

# Processar dados para cada ano
dados_2022 = processar_dados(df_area_acao, 2022)
dados_2023 = processar_dados(df_area_acao, 2023)
dados_2024 = processar_dados(df_area_acao, 2024)

# Concatenar os dados
tabela_final = pd.concat([dados_2022, dados_2023, dados_2024], axis=1).fillna(0)
tabela_final = tabela_final.reset_index()  # terá apenas 'nome_area_acao'

# Formatar valores inteiros para exibição
for ano in [2022, 2023, 2024]:
    for col in [f'sigilosos_{ano}', f'nao_sigilosos_{ano}', f'total_{ano}']:
        tabela_final[col] = tabela_final[col].astype(int)

# --- TABELA DE PROPORÇÕES COM VARIAÇÃO TOTAL E MÉDIA ---
tabela_proporcoes = tabela_final[['nome_area_acao'] + 
                        [f'sigilosos_{ano}' for ano in [2022, 2023, 2024]] +
                        [f'nao_sigilosos_{ano}' for ano in [2022, 2023, 2024]] +
                        [f'total_{ano}' for ano in [2022, 2023, 2024]] +
                        [f'proporcao_sigilosos_{ano}' for ano in [2022, 2023, 2024]] +
                        [f'proporcao_nao_sigilosos_{ano}' for ano in [2022, 2023, 2024]]].copy()
                
# Converter proporções para float
for ano in [2022, 2023, 2024]:
    tabela_proporcoes[f'proporcao_sigilosos_{ano}'] = tabela_proporcoes[f'proporcao_sigilosos_{ano}'].astype(float)
    tabela_proporcoes[f'proporcao_nao_sigilosos_{ano}'] = tabela_proporcoes[f'proporcao_nao_sigilosos_{ano}'].astype(float)

# Variações e médias
tabela_proporcoes['variacao_total_sigilosos'] = (
    (tabela_proporcoes['proporcao_sigilosos_2023'] - tabela_proporcoes['proporcao_sigilosos_2022']) +
    (tabela_proporcoes['proporcao_sigilosos_2024'] - tabela_proporcoes['proporcao_sigilosos_2023'])
)

tabela_proporcoes['variacao_total_nao_sigilosos'] = (
    (tabela_proporcoes['proporcao_nao_sigilosos_2023'] - tabela_proporcoes['proporcao_nao_sigilosos_2022']) +
    (tabela_proporcoes['proporcao_nao_sigilosos_2024'] - tabela_proporcoes['proporcao_nao_sigilosos_2023'])
)

tabela_proporcoes['proporcao_media_sigilosos'] = tabela_proporcoes[
    ['proporcao_sigilosos_2022', 'proporcao_sigilosos_2023', 'proporcao_sigilosos_2024']
].mean(axis=1)

tabela_proporcoes['proporcao_media_nao_sigilosos'] = tabela_proporcoes[
    ['proporcao_nao_sigilosos_2022', 'proporcao_nao_sigilosos_2023', 'proporcao_nao_sigilosos_2024']
].mean(axis=1)
    
tabela_proporcoes['total_sigilosos'] = (
    tabela_proporcoes['sigilosos_2022'] +
    tabela_proporcoes['sigilosos_2023'] +
    tabela_proporcoes['sigilosos_2024']
)

tabela_proporcoes['total_nao_sigilosos'] = (
    tabela_proporcoes['nao_sigilosos_2022'] +
    tabela_proporcoes['nao_sigilosos_2023'] +
    tabela_proporcoes['nao_sigilosos_2024']
)
    
tabela_proporcoes['total_processos'] = (
    tabela_proporcoes['total_sigilosos'] +
    tabela_proporcoes['total_nao_sigilosos']
)

# Formatação BR
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

header_values = [
    'Área de Ação',
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
]
num_cols = len(header_values)

row_colors = get_row_colors(num_rows)
fill_color = [row_colors] * num_cols 

cells_values = [
    tabela_proporcoes_formatada['nome_area_acao'],
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
]

# Tabela Plotly das Proporções (sem coluna de Comarca)
fig_proporcoes = go.Figure(data=[go.Table(
    header=dict(
        values=header_values,
        fill_color='#203864',
        font=dict(color='white', size=12),
        align='left',
        line_color='darkslategray'
    ),
    cells=dict(
        values=cells_values,
        fill_color=fill_color,
        align='left',
        font=dict(color='black', size=11),
        line_color='darkslategray'
    )
)])

fig_proporcoes.update_layout(
    title='<b>Proporção de Casos Sigilosos por Área de Ação (2022-2024)</b><br>'
          '<i>Ordenado por Variação Total</i>',
    title_x=0.5,
    margin=dict(l=20, r=20, t=100, b=20),
    height=900,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# --- GRÁFICO DE DISPERSÃO ESTRATÉGICO (SIGILOSOS vs. NÃO SIGILOSOS) ---

# Dataframe para o gráfico de dispersão
tabela_dispersao = tabela_proporcoes.copy()

# Rótulo único por ponto (somente área de ação)
tabela_dispersao['rotulo'] = tabela_dispersao['nome_area_acao'].astype(str).str.strip()

colunas_hover = [
    'variacao_total_sigilosos',
    'sigilosos_2022',
    'sigilosos_2023',
    'sigilosos_2024',
    'nao_sigilosos_2022',
    'nao_sigilosos_2023',
    'nao_sigilosos_2024'
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
    hover_name='rotulo',
    custom_data=colunas_hover
)

media_proporcao_sigilosos = tabela_proporcoes['proporcao_media_sigilosos'].mean()
fig_dispersao.add_hline(y=0, line_dash="dash", line_color="grey")
fig_dispersao.add_vline(x=media_proporcao_sigilosos, line_dash="dash", line_color="grey")

fig_dispersao.add_annotation(x=95, y=tabela_proporcoes['variacao_total_sigilosos'].max()*0.9, text="<b>Especialistas em Expansão</b>", showarrow=False, bgcolor="#e3f2fd", xanchor='right')
fig_dispersao.add_annotation(x=5, y=tabela_proporcoes['variacao_total_sigilosos'].max()*0.9, text="<b>Novos Focos de Atuação</b>", showarrow=False, bgcolor="#e8f5e9", xanchor='left')
fig_dispersao.add_annotation(x=5, y=tabela_proporcoes['variacao_total_sigilosos'].min()*0.9, text="<b>Fora do Foco</b>", showarrow=False, bgcolor="#ffebee", xanchor='left')
fig_dispersao.add_annotation(x=95, y=tabela_proporcoes['variacao_total_sigilosos'].min()*0.9, text="<b>Especialistas em Transição</b>", showarrow=False, bgcolor="#fff3e0", xanchor='right')

fig_dispersao.update_xaxes(title_text="Proporção Média de Casos Sigilosos (%)", ticksuffix="%")
fig_dispersao.update_yaxes(title_text="Variação da Proporção (2024 - 2022)", ticksuffix="%")

fig_dispersao.update_traces(
    marker=dict(size=10, color='#203864'),
    hovertemplate="<br>".join([
        "<b>%{hovertext}</b>",
        "<b>Variação Total de Sigilosos:</b> %{customdata[0]:.2f}%",
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

fig_dispersao.add_hline(
    y=0, line_dash="dash", line_color="grey",
    annotation_text="Variação = 0", annotation_position="right"
)

fig_dispersao.add_vline(
    x=media_proporcao_sigilosos,
    line_dash="dash",
    line_color="grey",
    # rótulo com o valor da média
    annotation_text=f"Média: {media_proporcao_sigilosos:.2f}%".replace('.', ','),
    annotation_position="top"
)

fig_dispersao.update_layout(
    title_text='<b>Análise Estratégica Comparativa: Foco em Casos Sigilosos VS Não Sigilosos</b>',
    title_x=0.5,
    height=900,
    showlegend=False 
)

# Exibição dos resultados
fig_proporcoes.show()
fig_dispersao.show()

