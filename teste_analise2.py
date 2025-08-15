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
# Expressões regulares
import re 

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

# Tratamento dos números de OAB
def is_oab_valida(oab):
    """
    Verifica se um número de OAB é válido seguindo o formato:
    NÚMEROS + LETRA + ESPAÇO + UF. Ex: '2153421N GO'
    """
    if not isinstance(oab, str) or not oab.strip():
        return False
    oab_limpa = oab.upper().strip()
    ufs_validas = ['GO', 'DF', 'SP', 'RJ', 'MG', 'RS', 'SC', 'PR', 'BA', 'PE',
                  'CE', 'MA', 'ES', 'AL', 'SE', 'PB', 'RN', 'PI', 'MT', 'MS', 
                  'TO', 'PA', 'AP', 'AM', 'RR', 'AC', 'RO'
                  ]
    padrao_regex = re.compile(f"^[1-9]\\d*[A-Z]\\s({'|'.join(ufs_validas)})$")
    return bool(padrao_regex.match(oab_limpa))

# Aplicar a validação de OAB
df['oab_valida'] = df['oab'].apply(is_oab_valida)

# Contar e exibir a quantidade de OABs inválidas
registros_invalidos = df[df['oab_valida'] == False]
qtd_invalidos = len(registros_invalidos)

'''print("--- Validação de Registros de OAB ---")
print(f"Total de registros com OAB em formato inválido ou nulo: {qtd_invalidos}")

if qtd_invalidos > 0:
    exemplos_invalidos = registros_invalidos['oab'].unique()
    print(f"Exemplos de OABs inválidas: {exemplos_invalidos}")
print("\n" + "="*100 + "\n")'''

# Dataframe com apenas OABs válidas
df_validos = df[df['oab_valida'] == True].copy()

# --- 3) Análise de Dados ---
if not df_validos.empty:
    # Expandir múltiplos advogados por processo
    df_advogados = df_validos.assign(oab=df_validos['oab'].str.split(';')).explode('oab')
    df_advogados['oab'] = df_advogados['oab'].str.strip()
    
    # Processar dados por ano
    def processar_dados(df, ano):
        df_ano = df[df['ano_distribuicao'] ==ano]
        sigilosos = df_ano[df_ano['is_segredo_justica']].groupby('oab')['processo'].nunique()
        nao_sigilosos = df_ano[~df_ano['is_segredo_justica']].groupby('oab')['processo'].nunique()

        total = pd.DataFrame({
            f'sigilosos_{ano}': sigilosos,
            f'nao_sigilosos_{ano}': nao_sigilosos
        }).fillna(0)

        total[f'total_{ano}'] = total[f'sigilosos_{ano}'] + total[f'nao_sigilosos_{ano}']

        total[f'proporcao_sigilosos_{ano}'] = (total[f'sigilosos_{ano}'] / total[f'total_{ano}'].round(4) * 100)

        total[f'proporcao_nao_sigilosos_{ano}'] = (total[f'nao_sigilosos_{ano}'] / total[f'total_{ano}'].round(4) * 100)

        return total
    
    # Processar dados para cada ano
    dados_2022 = processar_dados(df_advogados, 2022)
    dados_2023 = processar_dados(df_advogados, 2023)
    dados_2024 = processar_dados(df_advogados, 2024)

    # Concatenar os dados
    tabela_final = pd.concat([dados_2022, dados_2023, dados_2024], axis=1).fillna(0)
    tabela_final = tabela_final.reset_index()

    # Formatar valores para exibição
    for ano in [2022, 2023, 2024]:
        for col in [f'sigilosos_{ano}', f'nao_sigilosos_{ano}', f'total_{ano}']:
            tabela_final[col] = tabela_final[col].astype(int)

    # --- TABELA DE PROPORÇÕES COM VARIAÇÃO TOTAL E MÉDIA ---
    tabela_proporcoes = tabela_final[['oab'] + 
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

    # Ordenar pela OAB crescente
    tabela_proporcoes = tabela_proporcoes.sort_values('oab')

    # Formatar para exibição (padrão brasileiro)
    tabela_proporcoes_formatada = tabela_proporcoes.copy()
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
    def get_row_colors(num_rows):
        return ['lavender' if i % 2 == 0 else 'white' for i in range(num_rows)]

    # Tabela Plotly das Proporções com Variação e Média
    fig_proporcoes = go.Figure(data=[go.Table(
        header=dict(
            values=[
                'OAB',
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
                tabela_proporcoes_formatada['oab'],
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
            fill_color=[get_row_colors(len(tabela_proporcoes_formatada))],
            align='left',
            font=dict(color='black', size=11),
            line_color='darkslategray'
        )
    )])

    fig_proporcoes.update_layout(
        title='<b>Proporção de Casos Sigilosos por Advogado (2022-2024)</b><br>'
              '<i>Ordenado por Variação Total</i>',
        title_x=0.5,
        margin=dict(l=20, r=20, t=100, b=20),
        height=900,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
   
    # --- GRÁFICO DE EVOLUÇÃO TEMPORAL POR ADVOGADO (TOP 10) --- 
    # Preparar dados para o gráfico
    top_advogados = tabela_proporcoes.nlargest(10, 'variacao_total_sigilosos')
    dados_grafico = top_advogados.melt(id_vars=['oab'], 
                                    value_vars=[
                                        'proporcao_sigilosos_2022', 
                                        'proporcao_sigilosos_2023', 
                                        'proporcao_sigilosos_2024'],
                                    var_name='ano',
                                    value_name='proporcao')

    # Converter anos para formato mais limpo
    dados_grafico['ano'] = dados_grafico['ano'].str.extract('(\d+)').astype(int)

    # Criar gráfico
    fig_top_10 = px.line(dados_grafico, 
                x='ano', 
                y='proporcao', 
                color='oab',
                markers=True,
                title='<b>Evolução da Proporção de Casos Sigilosos (Top 10 Advogados)</b>',
                labels={'proporcao': 'Proporção de Casos Sigilosos (%)', 'ano': 'Ano'})

    fig_top_10.update_layout(
        hovermode='x unified',
        yaxis=dict(tickformat=".2f%"),
        xaxis=dict(tickmode='linear', dtick=1),
        title_x=0.5,
        margin=dict(l=20, r=20, t=100, b=20),
        height=900,
        legend_title_text='OAB'
    )

    # --- GRÁFICO DE DISPERSÃO ESTRATÉGICO (SIGILOSOS vs. NÃO SIGILOSOS) ---

    # 1. Gráfico de Dispersão
    # Dataframe para o gráfico de disperção
    tabela_dispersao = pd.merge(
        tabela_proporcoes,
        tabela_final[['oab', 'sigilosos_2022', 'sigilosos_2023', 'sigilosos_2024',
                      'nao_sigilosos_2022', 'nao_sigilosos_2023', 'nao_sigilosos_2024']],
        on='oab',
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
        hover_name='oab',
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
            "<b>OAB:</b> %{hovertext}",
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
    #fig_top_10.show()


    

