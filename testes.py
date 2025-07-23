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

print("--- Validação de Registros de OAB ---")
print(f"Total de registros com OAB em formato inválido ou nulo: {qtd_invalidos}")

if qtd_invalidos > 0:
    exemplos_invalidos = registros_invalidos['oab'].unique()
    print(f"Exemplos de OABs inválidas: {exemplos_invalidos}")
print("\n" + "="*80 + "\n")

# Dataframe com apenas OABs válidas
df_validos = df[df['oab_valida'] == True].copy()

# 3) Análises

if not df_validos.empty:
    
    analise_advogados = df_validos.groupby(['ano_distribuicao', 'oab', 'is_segredo_justica'])['processo'].nunique().unstack(fill_value=0)
    analise_advogados.columns = ['Nao_Sigilosos', 'Sigilosos']
    analise_advogados['Total_Processos'] = analise_advogados['Nao_Sigilosos'] + analise_advogados['Sigilosos']
    analise_advogados['Proporcao_Sigilosos'] = (analise_advogados['Sigilosos'] / analise_advogados['Total_Processos'] * 100)
    total_sigilosos_adv = analise_advogados.groupby('oab')['Sigilosos'].sum()
    top_advogados = total_sigilosos_adv.nlargest(10).index
    analise_top_advogados = analise_advogados[analise_advogados.index.get_level_values('oab').isin(top_advogados)]

    # Análise 2: Gráfico de Dispersão dos Advogados Acima da Média Anual de Casos Sigilosos
    print("--- ANÁLISE 2: Gráfico de Dispersão dos Advogados Acima da Média Anual ---")

    # A lógica inicial para encontrar os advogados acima da média permanece a mesma
    media_anual_sigilosos = analise_advogados.groupby('ano_distribuicao')['Sigilosos'].mean()
    df_analise = analise_advogados.reset_index()
    df_analise['Media_Anual'] = df_analise['ano_distribuicao'].map(media_anual_sigilosos)
    adv_acima_media_anual = df_analise[df_analise['Sigilosos'] > df_analise['Media_Anual']]

    if not adv_acima_media_anual.empty:
        # --- INÍCIO DA ALTERAÇÃO ---
        # Criar o gráfico de dispersão em vez da tabela
        fig = px.scatter(
            adv_acima_media_anual,
            x='oab',
            y='Sigilosos',
            facet_col='ano_distribuicao', # Cria um gráfico separado para cada ano
            color='Sigilosos', # Colore os pontos pela quantidade, para destaque
            color_continuous_scale=px.colors.sequential.Blues,
            title='<b>Advogados Acima da Média Anual de Casos Sigilosos</b>',
            labels={'oab': 'Advogado', 'Sigilosos': 'Nº de Casos Sigilosos', 'ano_distribuicao': 'Ano'},
            hover_name='oab', # Mostra a OAB em negrito no topo do hover
            hover_data={
                'oab': False, # Oculta a OAB do corpo do hover, pois já está no título
                'Sigilosos': ':,', # Formata o número com separador de milhar
                'Media_Anual': ':.2f' # Mostra a média do ano no hover
            }
        )

        # Adicionar uma linha pontilhada para a média de cada ano
        for i, (ano, media) in enumerate(media_anual_sigilosos.items()):
            fig.add_hline(
                y=media,
                line_dash="dot",
                annotation_text=f"Média ({media:.2f})",
                annotation_position="bottom right",
                annotation_font_size=10,
                annotation_font_color="black",
                col=i + 1 # Aplica a linha ao subplot correto (1, 2, 3...)
            )

        # Limpar o layout para melhor visualização
        fig.update_layout(
            title_x=0.5,
            separators=',.',
            showlegend=False # Oculta a legenda de cores, pois não é necessária
        )
        # Ocultar os rótulos do eixo X para não poluir o gráfico, já que a OAB está no hover
        fig.update_xaxes(showticklabels=False, title_text="")
        
        # Exibir o gráfico
        fig.show()
        # --- FIM DA ALTERAÇÃO ---

    else:
        print("  Nenhum advogado foi encontrado acima da média em nenhum dos anos.")

    print("\n" + "="*80 + "\n")