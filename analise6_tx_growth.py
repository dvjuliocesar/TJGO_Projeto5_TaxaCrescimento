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

# Calcula Crescimento Total (%) e CAGR (%/ano) usando o primeiro ano com base > 0
def _calc_growth(row):
    totals = {2022: row['total_2022'], 2023: row['total_2023'], 2024: row['total_2024']}
    # escolhe a primeira base válida entre 2022 e 2023
    base_year = 2022 if totals[2022] > 0 else (2023 if totals[2023] > 0 else None)
    end_year = 2024
    end_val = totals[end_year]

    if base_year is None:
        # sem base > 0 em 2022 e 2023 → sem como medir crescimento
        return pd.Series({'crescimento_percentual_volume': 0.0, 'cagr_volume': 0.0, 'ano_base': np.nan})

    base_val = totals[base_year]
    n_periods = end_year - base_year  # 2 se base=2022, 1 se base=2023

    # Crescimento total (%)
    crescimento = ((end_val / base_val) - 1.0) * 100.0 if base_val > 0 else np.nan

    # CAGR (%/ano) — se zerou, considera queda total
    if base_val > 0 and end_val > 0:
        cagr = ((end_val / base_val) ** (1.0 / n_periods) - 1.0) * 100.0
    elif base_val > 0 and end_val == 0:
        cagr = -100.0
    else:
        cagr = np.nan

    return pd.Series({'crescimento_percentual_volume': crescimento, 'cagr_volume': cagr, 'ano_base': base_year})

tabela_proporcoes[['crescimento_percentual_volume', 'cagr_volume', 'ano_base']] = (
    tabela_proporcoes.apply(_calc_growth, axis=1)
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
tabela_dispersao['rotulo'] = tabela_dispersao['nome_area_acao'].astype(str).str.strip()

# Opcional: usar tamanho da bolha ~ volume total (2022–2024)
tabela_dispersao['volume_total'] = tabela_dispersao['total_processos'].astype(float)

# Informações no hover
colunas_hover = [
    'cagr_volume',                     # customdata[0]
    'crescimento_percentual_volume',   # customdata[1]
    'total_2022',                      # customdata[2]
    'total_2023',                      # customdata[3]
    'total_2024',                      # customdata[4]
    'proporcao_media_sigilosos'        # customdata[5]
]

# Escolha do eixo Y:
y_metric = 'cagr_volume'  # recomendado (%/ano). Alternativa: 'crescimento_percentual_volume'

fig_dispersao = px.scatter(
    tabela_dispersao,
    x='proporcao_media_sigilosos',
    y=y_metric,
    size='volume_total',        # opcional: remove se não quiser bolhas proporcionais
    size_max=28,
    title='<b>Análise Estratégica: Sigilo (média) vs Crescimento de Entradas</b>',
    labels={
        'proporcao_media_sigilosos': 'Proporção Média de Casos Sigilosos (%)',
        'cagr_volume': 'Crescimento Médio Anual de Entradas (CAGR, %/ano)',
        'crescimento_percentual_volume': 'Crescimento Total de Entradas (2024 vs 1º ano base, %)'
    },
    hover_name='rotulo',
    custom_data=colunas_hover
)

# Linhas de referência
media_proporcao_sigilosos = tabela_proporcoes['proporcao_media_sigilosos'].mean()
fig_dispersao.add_hline(y=0, line_dash="dash", line_color="grey",
                        annotation_text="Crescimento = 0", annotation_position="right")
fig_dispersao.add_vline(x=media_proporcao_sigilosos, line_dash="dash", line_color="grey",
                        annotation_text=f"Média Sigilo: {media_proporcao_sigilosos:.2f}%".replace('.', ','),
                        annotation_position="top")

# Anotações dos quadrantes (mantidas)
fig_dispersao.add_annotation(x=95, y=max(0, tabela_dispersao[y_metric].max())*0.9,
                             text="<b>Alta Demanda & Alto Sigilo</b>",
                             showarrow=False, bgcolor="#e3f2fd", xanchor='right')
fig_dispersao.add_annotation(x=5, y=max(0, tabela_dispersao[y_metric].max())*0.9,
                             text="<b>Crescimento com Baixo Sigilo</b>",
                             showarrow=False, bgcolor="#e8f5e9", xanchor='left')
fig_dispersao.add_annotation(x=5, y=min(0, tabela_dispersao[y_metric].min())*0.9,
                             text="<b>Baixa Demanda & Baixo Sigilo</b>",
                             showarrow=False, bgcolor="#ffebee", xanchor='left')
fig_dispersao.add_annotation(x=95, y=min(0, tabela_dispersao[y_metric].min())*0.9,
                             text="<b>Alto Sigilo em Queda</b>",
                             showarrow=False, bgcolor="#fff3e0", xanchor='right')

# Eixos
fig_dispersao.update_xaxes(title_text="Proporção Média de Casos Sigilosos (%)", ticksuffix="%")
fig_dispersao.update_yaxes(
    title_text="Crescimento de Entradas (%/ano)" if y_metric == 'cagr_volume' else "Crescimento Total de Entradas (%)",
    ticksuffix="%"
)

# Hover customizado
fig_dispersao.update_traces(
    marker=dict(size=10, color='#203864'),
    hovertemplate="<br>".join([
        "<b>Área de Ação:</b> %{hovertext}",
        "<b>CAGR de Entradas:</b> %{customdata[0]:.2f}%", 
        "<b>Crescimento Total:</b> %{customdata[1]:.2f}%",
        "<b>--- <b>Totais</b> ---",
        "<b>Total 2022:</b> %{customdata[2]}",
        "<b>Total 2023:</b> %{customdata[3]}",
        "<b>Total 2024:</b> %{customdata[4]}",
        "<b>Sigilo (média 22–24):</b> %{customdata[5]:.2f}%",
        "<extra></extra>"
    ])
)

fig_dispersao.update_layout(
    title_text='<b>Análise Estratégica: Foco em Crescimento de Entradas vs Nível de Sigilo</b>',
    title_x=0.5,
    height=900,
    showlegend=False 
)

# Exibição dos resultados
#fig_proporcoes.show()
fig_dispersao.show()

