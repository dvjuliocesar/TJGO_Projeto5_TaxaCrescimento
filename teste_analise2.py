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

# --- 3) Análise e Geração da Tabela ---
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
                            [f'proporcao_sigilosos_{ano}' for ano in [2022, 2023, 2024]]].copy()

    # Converter proporções para float (remover % temporariamente)
    for ano in [2022, 2023, 2024]:
        tabela_proporcoes[f'proporcao_sigilosos_{ano}'] = tabela_proporcoes[f'proporcao_sigilosos_{ano}'].astype(float)

    # Calcular Variação Total (2022 → 2024)
    tabela_proporcoes['variacao_total'] = (
        (tabela_proporcoes['proporcao_sigilosos_2023'] - tabela_proporcoes['proporcao_sigilosos_2022']) +
        (tabela_proporcoes['proporcao_sigilosos_2024'] - tabela_proporcoes['proporcao_sigilosos_2023'])
    )

    # Calcular Proporção Média (2022-2024)
    tabela_proporcoes['proporcao_media'] = tabela_proporcoes[
        ['proporcao_sigilosos_2022', 'proporcao_sigilosos_2023', 'proporcao_sigilosos_2024']
    ].mean(axis=1)

    # Ordenar pela Variação Total (maior → menor)
    tabela_proporcoes = tabela_proporcoes.sort_values('variacao_total', ascending=False)

    # Formatar para exibição (padrão brasileiro)
    tabela_proporcoes_formatada = tabela_proporcoes.copy()
    for ano in [2022, 2023, 2024]:
        tabela_proporcoes_formatada[f'proporcao_sigilosos_{ano}'] = tabela_proporcoes_formatada[
            f'proporcao_sigilosos_{ano}'
        ].apply(lambda x: f"{x:.2f}%".replace('.', ','))

    tabela_proporcoes_formatada['variacao_total'] = tabela_proporcoes_formatada['variacao_total'].apply(
        lambda x: f"{x:+.2f}%".replace('.', ',')  # Formato +X,YZ% ou -X,YZ%
    )

    tabela_proporcoes_formatada['proporcao_media'] = tabela_proporcoes_formatada['proporcao_media'].apply(
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
                'Proporção de Sigilosos em 2022', 
                'Proporção de Sigilosos em 2023', 
                'Proporção de Sigilosos em 2024', 
                'Variação Total', 
                'Proporção Média'    
            ],
            fill_color='#203864',
            font=dict(color='white', size=12),
            align='left',
            line_color='darkslategray'
        ),
        cells=dict(
            values=[
                tabela_proporcoes_formatada['oab'],
                tabela_proporcoes_formatada['proporcao_sigilosos_2022'],
                tabela_proporcoes_formatada['proporcao_sigilosos_2023'],
                tabela_proporcoes_formatada['proporcao_sigilosos_2024'],
                tabela_proporcoes_formatada['variacao_total'],
                tabela_proporcoes_formatada['proporcao_media']
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
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
   
    # Exibir tabela
    fig_proporcoes.show()
