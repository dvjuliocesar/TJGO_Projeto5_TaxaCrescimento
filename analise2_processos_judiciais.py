'''Análise de Processos Judiciais - Advogados e Sigilos:
- Este script analisa dados de processos judiciais, focando na validação de números de OAB
e na proporção de processos sigilosos por advogado. Ele gera tabelas e gráficos para visualização dos dados.'''

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
    # Análise 1: Proporção de processos sigilosos por advogado ao ano
    analise_advogados = df_validos.groupby(['ano_distribuicao', 'oab', 'is_segredo_justica'])['processo'].nunique().unstack(fill_value=0)
    analise_advogados.columns = ['Nao_Sigilosos', 'Sigilosos']
    analise_advogados['Total_Processos'] = analise_advogados['Nao_Sigilosos'] + analise_advogados['Sigilosos']
    analise_advogados['Proporcao_Sigilosos'] = (analise_advogados['Sigilosos'] / analise_advogados['Total_Processos'] * 100)
    total_sigilosos_adv = analise_advogados.groupby('oab')['Sigilosos'].sum()
    top_advogados = total_sigilosos_adv.nlargest(10).index
    analise_top_advogados = analise_advogados[analise_advogados.index.get_level_values('oab').isin(top_advogados)]
    # Tabela resumo
    tabela_resumo = analise_top_advogados.reset_index()
    tabela_resumo.rename(columns={'ano_distribuicao': 'Ano', 
                                'oab': 'OAB', 
                                'Nao_Sigilosos': 'Não Sigilosos',
                                'Total_Processos': 'Total de Processos',
                                'Proporcao_Sigilosos': 'Proporção de Sigilosos'}, 
                                inplace=True)
    # Ordenar primeiro por Ano (crescente) e depois pela Proporção (decrescente)
    tabela_resumo = tabela_resumo.sort_values(
        by=['Ano', 'Proporção de Sigilosos'], 
        ascending=[True, False]
    )
    # Formatar a coluna de proporção
    tabela_resumo['Proporção de Sigilosos'] = tabela_resumo['Proporção de Sigilosos'].map('{:,.2f}%'.format)  

    # Exibir a tabela
    print("--- Tabela de Análise: Top 10 Advogados com Mais Casos Sigilosos por Ano ---")
    # Ajustar a OAB para ser exibida como inteiro, sem casas decimais
    print(tabela_resumo.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # Análise 2: Advogados acima da média GERAL de processos sigilosos
    print('--- Análise: Advogados Acima da Média Geral de Processos Sigilosos ---')
    total_sigilosos_adv = analise_advogados.groupby('oab')['Sigilosos'].sum()
    media_geral_sigilosos = total_sigilosos_adv.mean()
    adv_acima_media_geral = total_sigilosos_adv[total_sigilosos_adv > media_geral_sigilosos].reset_index()
    adv_acima_media_geral.columns = ['oab', 'Sigilosos']
    # top 10 advogados acima da média
    adv_acima_media_geral = adv_acima_media_geral.nlargest(10, 'Sigilosos')
    adv_acima_media_geral['Sigilosos'] = adv_acima_media_geral['Sigilosos']
    adv_acima_media_geral['oab'] = adv_acima_media_geral['oab']
    adv_acima_media_geral = adv_acima_media_geral.reset_index(drop=True)
    adv_acima_media_geral = adv_acima_media_geral.rename(columns={'oab': 'OAB', 'Sigilosos': 'Total de Casos Sigilosos'})

    print(f"Média geral de casos sigilosos por advogado no período: {media_geral_sigilosos:.2f}")
    print("Advogados com atuação acima da média geral:")
    print(adv_acima_media_geral.sort_values(by='Total de Casos Sigilosos', ascending=False).to_string(index=False))
    print("\n" + "="*80 + "\n")

    # Análise 3: Top 10 Advogados acima da média ANUAL de processos sigilosos
    print("--- Análise: Advogados Acima da Média Anual de Casos Sigilosos ---")
    media_anual_sigilosos = analise_advogados.groupby('ano_distribuicao')['Sigilosos'].mean()

    # Mantém a lógica de encontrar TODOS os advogados acima da média
    df_analise = analise_advogados.reset_index()
    df_analise['Media_Anual'] = df_analise['ano_distribuicao'].map(media_anual_sigilosos)
    adv_acima_media_anual = df_analise[df_analise['Sigilosos'] > df_analise['Media_Anual']]

    # O loop agora irá filtrar e pegar apenas o Top 10 de cada ano
    for ano, media in media_anual_sigilosos.items():
        print(f"Ano: {int(ano)} (Média Anual de Casos Sigilosos: {media:.2f})")
        
        # Filtra a tabela para o ano corrente
        tabela_ano = adv_acima_media_anual[adv_acima_media_anual['ano_distribuicao'] == ano][['oab', 'Sigilosos']]
        
        if not tabela_ano.empty:
            # Ordena por quantidade de casos e seleciona os 10 primeiros
            top_10_ano = tabela_ano.sort_values(by='Sigilosos', ascending=False).head(10)
            print(top_10_ano.to_string(index=False))
        else:
            print("  Nenhum advogado acima da média neste ano.")
        print("-" * 50)
    print("\n" + "="*80 + "\n")
    







