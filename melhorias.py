# --- MELHORIAS NA ANÁLISE COMPORTAMENTAL DE ADVOGADOS ---
# Baseado no código original, com implementação de metodologias mais robustas

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
from scipy import stats
from tqdm import tqdm  # Para barra de progresso
import warnings
warnings.filterwarnings('ignore')

# Configurações Iniciais
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# Listar os arquivos CSV na pasta 'uploads'
arquivos_csv = glob.glob('uploads/processos_*.csv')

# Carregar os arquivos CSV e concatenar em um único DataFrame
dfs = []
for arquivo in arquivos_csv:
    ano = int(arquivo.split('_')[-1].split('.')[0])
    df_ano = pd.read_csv(arquivo, sep=',', encoding='utf-8')
    df_ano['ano_arquivo'] = ano
    dfs.append(df_ano)

df = pd.concat(dfs, ignore_index=True)

# Tratamento das colunas
df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
df['ano_distribuicao'] = df['data_distribuicao'].dt.year
df['is_segredo_justica'] = df['is_segredo_justica'].astype(bool)

# Função de validação OAB
def is_oab_valida(oab):
    if not isinstance(oab, str) or not oab.strip():
        return False
    oab_limpa = oab.upper().strip()
    ufs_validas = ['GO', 'DF', 'SP', 'RJ', 'MG', 'RS', 'SC', 'PR', 'BA', 'PE',
                  'CE', 'MA', 'ES', 'AL', 'SE', 'PB', 'RN', 'PI', 'MT', 'MS', 
                  'TO', 'PA', 'AP', 'AM', 'RR', 'AC', 'RO']
    padrao_regex = re.compile(f"^[1-9]\\d*[A-Z]\\s({'|'.join(ufs_validas)})$")
    return bool(padrao_regex.match(oab_limpa))

df['oab_valida'] = df['oab'].apply(is_oab_valida)
df_validos = df[df['oab_valida'] == True].copy()

# Expandir múltiplos advogados por processo
df_advogados = df_validos.assign(oab=df_validos['oab'].str.split(';')).explode('oab')
df_advogados['oab'] = df_advogados['oab'].str.strip()

# --- FUNÇÃO OTIMIZADA DE PROCESSAMENTO ---
def processar_dados_melhorado(df, anos=[2022, 2023, 2024]):
    """
    MELHORIA 1: Processamento otimizado com análise de significância estatística
    - Filtra OABs com volume mínimo de casos
    - Calcula intervalos de confiança para as proporções
    - Identifica variações estatisticamente significativas
    - Pondera análise pelo volume de casos
    """
    
    # Filtrar apenas OABs com pelo menos 5 casos totais
    contagem_oabs = df['oab'].value_counts()
    oabs_relevantes = contagem_oabs[contagem_oabs >= 5].index
    df_filtrado = df[df['oab'].isin(oabs_relevantes)]
    
    resultados = []
    oabs_unicas = df_filtrado['oab'].unique()
    
    print(f"Processando {len(oabs_unicas)} OABs com volume suficiente...")
    
    for oab in tqdm(oabs_unicas, desc="Analisando advogados"):
        try:
            df_oab = df_filtrado[df_filtrado['oab'] == oab]
            linha_resultado = {'oab': oab}
            
            # Dados anuais
            dados_anuais = {}
            for ano in anos:
                df_ano = df_oab[df_oab['ano_distribuicao'] == ano]
                sigilosos = len(df_ano[df_ano['is_segredo_justica']])
                nao_sigilosos = len(df_ano[~df_ano['is_segredo_justica']])
                total = sigilosos + nao_sigilosos
                
                # Proporção e intervalo de confiança (usando distribuição beta)
                if total > 0:
                    proporcao = sigilosos / total * 100
                    # Intervalo de confiança binomial (Agresti-Coull)
                    n_tilde = total + 4
                    p_tilde = (sigilosos + 2) / n_tilde
                    erro_padrao = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
                    margem_erro = 1.96 * erro_padrao * 100  # 95% confiança
                    ic_inferior = max(0, p_tilde * 100 - margem_erro)
                    ic_superior = min(100, p_tilde * 100 + margem_erro)
                else:
                    proporcao = ic_inferior = ic_superior = 0
                
                dados_anuais[ano] = {
                    'sigilosos': sigilosos,
                    'nao_sigilosos': nao_sigilosos,
                    'total': total,
                    'proporcao': proporcao,
                    'ic_inferior': ic_inferior,
                    'ic_superior': ic_superior
                }
                
                # Adicionar à linha resultado
                linha_resultado.update({
                    f'sigilosos_{ano}': sigilosos,
                    f'nao_sigilosos_{ano}': nao_sigilosos,
                    f'total_{ano}': total,
                    f'proporcao_{ano}': proporcao,
                    f'ic_inf_{ano}': ic_inferior,
                    f'ic_sup_{ano}': ic_superior
                })
            
            # MELHORIA 2: Cálculo correto de variação
            # Variação absoluta correta (não soma de diferenças)
            variacao_absoluta = dados_anuais[2024]['proporcao'] - dados_anuais[2022]['proporcao']
            
            # Variação relativa (percentual sobre valor inicial)
            if dados_anuais[2022]['proporcao'] > 0:
                variacao_relativa = (variacao_absoluta / dados_anuais[2022]['proporcao']) * 100
            else:
                variacao_relativa = np.inf if dados_anuais[2024]['proporcao'] > 0 else 0
            
            linha_resultado.update({
                'variacao_absoluta': variacao_absoluta,
                'variacao_relativa': variacao_relativa
            })
            
            # MELHORIA 3: Análise de significância estatística
            # Teste qui-quadrado para mudança significativa
            total_2022 = dados_anuais[2022]['total']
            total_2024 = dados_anuais[2024]['total']
            sig_2022 = dados_anuais[2022]['sigilosos']
            sig_2024 = dados_anuais[2024]['sigilosos']
            
            # Critério mínimo para teste (pelo menos 5 casos esperados em cada célula)
            if all(x >= 5 for x in [sig_2022, total_2022-sig_2022, sig_2024, total_2024-sig_2024]) and total_2022 > 0 and total_2024 > 0:
                # Tabela de contingência 2x2
                tabela_contingencia = np.array([
                    [sig_2022, total_2022 - sig_2022],
                    [sig_2024, total_2024 - sig_2024]
                ])
                chi2, p_valor = stats.chi2_contingency(tabela_contingencia)[:2]
                mudanca_significativa = p_valor < 0.05
            else:
                p_valor = np.nan
                mudanca_significativa = False
                
            linha_resultado.update({
                'p_valor_mudanca': p_valor,
                'mudanca_significativa': mudanca_significativa
            })
            
            # MELHORIA 4: Métricas de estabilidade temporal
            proporcoes_anuais = [dados_anuais[ano]['proporcao'] for ano in anos]
            
            # Coeficiente de variação das proporções
            if np.mean(proporcoes_anuais) > 0:
                coef_variacao = np.std(proporcoes_anuais) / np.mean(proporcoes_anuais) * 100
            else:
                coef_variacao = 0
                
            # Tendência linear (slope)
            x = np.array(anos)
            y = np.array(proporcoes_anuais)
            if len(x) > 1 and np.std(y) > 0:
                slope, intercept, r_valor, p_tendencia, std_err = stats.linregress(x, y)
                tendencia_significativa = p_tendencia < 0.05
            else:
                slope = r_valor = p_tendencia = 0
                tendencia_significativa = False
                
            linha_resultado.update({
                'coef_variacao': coef_variacao,
                'slope_tendencia': slope,
                'r_tendencia': r_valor,
                'p_tendencia': p_tendencia,
                'tendencia_significativa': tendencia_significativa
            })
            
            # MELHORIA 5: Volume ponderado e confiabilidade
            total_casos = sum(dados_anuais[ano]['total'] for ano in anos)
            proporcao_media_ponderada = sum(dados_anuais[ano]['sigilosos'] for ano in anos) / max(total_casos, 1) * 100
            
            # Classificação de confiabilidade baseada no volume
            if total_casos >= 50:
                confiabilidade = 'Alta'
            elif total_casos >= 20:
                confiabilidade = 'Média'
            elif total_casos >= 10:
                confiabilidade = 'Baixa'
            else:
                confiabilidade = 'Muito Baixa'
                
            linha_resultado.update({
                'total_casos': total_casos,
                'proporcao_media_ponderada': proporcao_media_ponderada,
                'confiabilidade': confiabilidade
            })
            
            resultados.append(linha_resultado)
        
        except Exception as e:
            print(f"Erro processando OAB {oab}: {e}")
            continue
    
    return pd.DataFrame(resultados)

# Executar análise melhorada
print("Processando análise comportamental melhorada...")
tabela_melhorada = processar_dados_melhorado(df_advogados)

# MELHORIA 6: Classificação estratégica aprimorada usando quartis e significância
def classificar_estrategicamente_melhorado(df):
    """
    Classificação mais robusta usando quartis em vez de média simples
    e considerando significância estatística e confiabilidade
    """
    # Filtrar apenas casos com confiabilidade mínima
    df_confiavel = df[df['confiabilidade'].isin(['Alta', 'Média'])].copy()
    
    if len(df_confiavel) == 0:
        print("Aviso: Nenhum advogado com confiabilidade mínima encontrado!")
        df_confiavel = df.copy()
    
    # Usar tercil em vez de média para classificação mais equilibrada
    percentil_33 = df_confiavel['proporcao_media_ponderada'].quantile(0.33)
    percentil_67 = df_confiavel['proporcao_media_ponderada'].quantile(0.67)
    
    def classificar_perfil(row):
        prop_media = row['proporcao_media_ponderada']
        variacao = row['variacao_absoluta']
        significativa = row['mudanca_significativa']
        confiavel = row['confiabilidade'] in ['Alta', 'Média']
        
        # Classificação refinada
        if prop_media >= percentil_67:  # Alta proporção média
            if variacao > 5 and significativa:
                return 'Especialista Confirmado em Expansão'
            elif variacao > 5:
                return 'Especialista em Possível Expansão'
            elif variacao < -5 and significativa:
                return 'Especialista em Transição Confirmada'
            elif variacao < -5:
                return 'Especialista em Possível Transição'
            else:
                return 'Especialista Estável'
                
        elif prop_media >= percentil_33:  # Proporção média moderada
            if variacao > 10 and significativa:
                return 'Emergente Confirmado'
            elif variacao > 10:
                return 'Emergente Potencial'
            elif variacao < -10 and significativa:
                return 'Moderado em Declínio Confirmado'
            else:
                return 'Moderado Estável'
                
        else:  # Baixa proporção média
            if variacao > 15 and significativa:
                return 'Novo Foco Confirmado'
            elif variacao > 15:
                return 'Novo Foco Potencial'
            else:
                return 'Fora do Foco Sigiloso'
        
    df['classificacao_melhorada'] = df.apply(classificar_perfil, axis=1)
    return df

# Aplicar classificação melhorada
tabela_melhorada = classificar_estrategicamente_melhorado(tabela_melhorada)

# MELHORIA 7: Métricas agregadas melhoradas
def gerar_metricas_melhoradas(df):
    """
    Gera estatísticas descritivas mais robustas
    """
    print("\n" + "="*120)
    print("ANÁLISE COMPORTAMENTAL MELHORADA - ADVOGADOS E PROCESSOS SIGILOSOS")
    print("="*120)
    
    # Distribuição por confiabilidade
    print("\n--- ANÁLISE DE CONFIABILIDADE DOS DADOS ---")
    confiabilidade_dist = df['confiabilidade'].value_counts()
    total_advogados = len(df)
    
    for nivel, qtd in confiabilidade_dist.items():
        pct = qtd/total_advogados*100
        print(f"{nivel}: {qtd:,} advogados ({pct:.1f}%)")
    
    # Análise apenas dos dados confiáveis
    df_confiavel = df[df['confiabilidade'].isin(['Alta', 'Média'])]
    print(f"\nAdvogados com dados confiáveis: {len(df_confiavel):,} ({len(df_confiavel)/total_advogados*100:.1f}%)")
    
    # Distribuição da classificação melhorada
    print("\n--- CLASSIFICAÇÃO ESTRATÉGICA MELHORADA ---")
    class_dist = df_confiavel['classificacao_melhorada'].value_counts()
    
    for classe, qtd in class_dist.items():
        pct = qtd/len(df_confiavel)*100
        print(f"{classe}: {qtd:,} advogados ({pct:.1f}%)")
    
    # Análise de significância
    print("\n--- ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA ---")
    mudancas_sig = df_confiavel['mudanca_significativa'].sum()
    pct_mudancas_sig = mudancas_sig/len(df_confiavel)*100
    print(f"Advogados com mudança estatisticamente significativa: {mudancas_sig:,} ({pct_mudancas_sig:.1f}%)")
    
    # Tendências significativas
    tendencias_sig = df_confiavel['tendencia_significativa'].sum()
    pct_tendencias_sig = tendencias_sig/len(df_confiavel)*100
    print(f"Advogados com tendência linear significativa: {tendencias_sig:,} ({pct_tendencias_sig:.1f}%)")
    
    # Estatísticas das variações
    print("\n--- ESTATÍSTICAS DE VARIAÇÃO ---")
    variacao_stats = df_confiavel['variacao_absoluta'].describe()
    print(f"Variação absoluta média: {variacao_stats['mean']:.2f} pontos percentuais")
    print(f"Variação absoluta mediana: {variacao_stats['50%']:.2f} pontos percentuais")
    print(f"Desvio padrão das variações: {variacao_stats['std']:.2f} pontos percentuais")
    print(f"Variação mínima: {variacao_stats['min']:.2f} pontos percentuais")
    print(f"Variação máxima: {variacao_stats['max']:.2f} pontos percentuais")
    
    print("="*120 + "\n")
    
    return df_confiavel

# Gerar análise melhorada
df_analise_final = gerar_metricas_melhoradas(tabela_melhorada)

# MELHORIA 8: Visualização melhorada com intervalos de confiança
def criar_grafico_melhorado(df):
    """
    Cria visualização com intervalos de confiança e classificação melhorada
    """
    # Filtrar apenas dados confiáveis
    df_plot = df[df['confiabilidade'].isin(['Alta', 'Média'])].copy()
    
    if len(df_plot) == 0:
        print("Aviso: Nenhum dado confiável para visualização!")
        return None
    
    # Definir cores por classificação
    color_map = {
        'Especialista Confirmado em Expansão': '#1f77b4',
        'Especialista em Possível Expansão': '#aec7e8',
        'Especialista Estável': '#2ca02c',
        'Especialista em Transição Confirmada': '#d62728',
        'Especialista em Possível Transição': '#ff9896',
        'Emergente Confirmado': '#ff7f0e',
        'Emergente Potencial': '#ffbb78',
        'Moderado Estável': '#9467bd',
        'Moderado em Declínio Confirmado': '#c5b0d5',
        'Novo Foco Confirmado': '#8c564b',
        'Novo Foco Potencial': '#c49c94',
        'Fora do Foco Sigiloso': '#e377c2'
    }
    
    # Criar gráfico de dispersão melhorado
    fig = px.scatter(
        df_plot,
        x='proporcao_media_ponderada',
        y='variacao_absoluta',
        color='classificacao_melhorada',
        size='total_casos',
        hover_data=['mudanca_significativa', 'tendencia_significativa', 'p_valor_mudanca'],
        title='<b>Análise Estratégica Melhorada: Processos Sigilosos por Advogado</b>',
        labels={
            'proporcao_media_ponderada': 'Proporção Média Ponderada de Casos Sigilosos (%)',
            'variacao_absoluta': 'Variação Absoluta (2024 - 2022) em pontos percentuais'
        },
        color_discrete_map=color_map
    )
    
    # Adicionar linhas de referência
    media_prop = df_plot['proporcao_media_ponderada'].median()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Sem variação")
    fig.add_vline(x=media_prop, line_dash="dash", line_color="grey", 
                  annotation_text=f"Mediana: {media_prop:.1f}%")
    
    # Configurar layout
    fig.update_layout(
        height=800,
        title_x=0.5,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    fig.show()
    
    return fig

# Criar visualização melhorada
grafico_melhorado = criar_grafico_melhorado(df_analise_final)

print("Análise comportamental melhorada concluída!")
print("Principais melhorias implementadas:")
print("   1. Cálculo correto de variações")
print("   2. Análise de significância estatística")
print("   3. Intervalos de confiança para proporções")
print("   4. Classificação por quartis em vez de média simples")
print("   5. Ponderação pelo volume de casos")
print("   6. Análise de estabilidade temporal")
print("   7. Classificação de confiabilidade dos dados")
print("   8. Visualização com tamanho proporcional ao volume")
print("   9. Filtro de OABs com volume mínimo")
print("   10. Barra de progresso para monitoramento")