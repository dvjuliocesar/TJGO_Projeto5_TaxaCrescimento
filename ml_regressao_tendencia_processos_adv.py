# -- BIBLIOTECAS --
# Manipulação de dados
import pandas as pd
import numpy as np
import glob
import re

# Visualização
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

# Machine Learning e Estatística
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Configurações Iniciais
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# 1) Carregar e tratar dados
def carregar_dados():
    """Carrega e concatena todos os arquivos CSV da pasta uploads"""
    arquivos_csv = glob.glob('uploads/processos_*.csv')
    dfs = []
    
    for arquivo in arquivos_csv:
            ano = int(arquivo.split('_')[-1].split('.')[0])
            df_ano = pd.read_csv(arquivo, sep=',', encoding='utf-8')
            df_ano['ano_arquivo'] = ano
            dfs.append(df_ano)
    return pd.concat(dfs, ignore_index=True)

df = carregar_dados()

# 2) Pré-processamento
def preprocessar(df):
    """Realiza tratamento inicial dos dados"""
    # Converter tipos
    df['data_distribuicao'] = pd.to_datetime(df['data_distribuicao'], errors='coerce')
    df['ano_distribuicao'] = df['data_distribuicao'].dt.year
    df['is_segredo_justica'] = df['is_segredo_justica'].astype(bool)
    
    # Validar OAB
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
    
    # Filtrar apenas registros válidos
    df_validos = df[df['oab_valida'] & df['is_segredo_justica']].copy()
    
    return df_validos 

df_sigilosos = preprocessar(df)

# 3) Preparar dados para análise temporal
def preparar_serie_temporal(df):
    """Prepara série temporal de processos sigilosos por advogado"""
    # Expandir múltiplos advogados por processo
    df_exp = df.assign(oab=df['oab'].str.split(';')).explode('oab')
    df_exp['oab'] = df_exp['oab'].str.strip()
    
    # Agregar por advogado e ano
    df_agg = df_exp.groupby(['oab', 'ano_distribuicao']).agg(
        processos_sigilosos=('processo', 'nunique')
    ).reset_index()
    
    # Verificar anos necessários
    anos_necessarios = {2022, 2023, 2024}
    anos_presentes = set(df_agg['ano_distribuicao'].unique())
    
    if not anos_necessarios.issubset(anos_presentes):
        faltantes = anos_necessarios - anos_presentes
        raise ValueError(f"Anos necessários não encontrados nos dados: {faltantes}")
    
    # Pivotar para formato wide
    df_pivot = df_agg.pivot(
        index='oab', 
        columns='ano_distribuicao', 
        values='processos_sigilosos'
    ).fillna(0)
    
    # Renomear colunas para facilitar acesso
    df_pivot.columns = [f"sigilosos_{col}" for col in df_pivot.columns]
    
    return df_pivot

df_temporal = preparar_serie_temporal(df_sigilosos)

# 4) Análise de Regressão
def analisar_tendencia(df):
    """Realiza análise de regressão linear para tendência temporal"""
    # Preparar variáveis
    X = df[['sigilosos_2022', 'sigilosos_2023']]
    y = df['sigilosos_2024']
    
    # Adicionar constante para intercepto
    X = sm.add_constant(X)
    
    # Ajustar modelo
    modelo = sm.OLS(y, X).fit()
    
    return modelo

modelo = analisar_tendencia(df_temporal)

# 5) Verificação de pressupostos
def verificar_pressupostos(modelo):
    """Verifica os pressupostos da regressão linear"""
    print("\n=== VERIFICAÇÃO DOS PRESSUPOSTOS ===")
    
    # 1. Linearidade
    print("\n1. LINEARIDADE:")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=modelo.model.exog[:,1], y=modelo.model.endog, ax=ax[0])
    ax[0].set_title('2022 vs 2023')
    sns.scatterplot(x=modelo.model.exog[:,2], y=modelo.model.endog, ax=ax[1])
    ax[1].set_title('2023 vs 2024')
    plt.tight_layout()
    plt.show()
    
    # 2. Independência dos erros
    print("\n2. INDEPENDÊNCIA DOS ERROS (Durbin-Watson):")
    dw = sm.stats.durbin_watson(modelo.resid)
    print(f"Valor: {dw:.2f} (próximo de 2 indica independência)")
    
    # 3. Homocedasticidade
    print("\n3. HOMOCEDASTICIDADE (Breusch-Pagan):")
    _, pval, _, _ = het_breuschpagan(modelo.resid, modelo.model.exog)
    print(f"p-valor: {pval:.4f} (p > 0.05 indica homocedasticidade)")
    
    # Gráfico de resíduos vs fitted
    plt.figure(figsize=(8, 6))
    sns.residplot(x=modelo.fittedvalues, y=modelo.resid, lowess=True)
    plt.title('Resíduos vs Valores Ajustados')
    plt.xlabel('Valores Ajustados')
    plt.ylabel('Resíduos')
    plt.show()
    
    # 4. Normalidade dos resíduos
    print("\n4. NORMALIDADE DOS RESÍDUOS (Shapiro-Wilk):")
    shapiro_test = stats.shapiro(modelo.resid)
    print(f"p-valor: {shapiro_test[1]:.4f} (p > 0.05 indica normalidade)")
    
    # QQ Plot
    plt.figure(figsize=(8, 6))
    qqplot(modelo.resid, line='s')
    plt.title('QQ Plot dos Resíduos')
    plt.show()
    
    # 5. Multicolinearidade
    print("\n5. MULTICOLINEARIDADE (VIF):")
    vif = pd.DataFrame()
    vif["Variável"] = modelo.model.exog_names
    vif["VIF"] = [variance_inflation_factor(modelo.model.exog, i) 
                 for i in range(modelo.model.exog.shape[1])]
    print(vif)
    print("VIF < 5 indica baixa multicolinearidade")

verificar_pressupostos(modelo)

# 6) Resultados e visualização
def visualizar_resultados(df, modelo):
    """Gera visualizações dos resultados"""
    # Adicionar previsões e métricas ao DataFrame
    df['previsao_2024'] = modelo.predict(sm.add_constant(df[['sigilosos_2022', 'sigilosos_2023']]))
    df['crescimento_abs'] = df['sigilosos_2024'] - df['sigilosos_2023']
    df['crescimento_rel'] = (df['crescimento_abs'] / df['sigilosos_2023'].replace(0, 1)) * 100
    
    # Corrigir valores negativos para tamanho (usar valor absoluto e adicionar um mínimo)
    df['tamanho_marcador'] = df['crescimento_abs'].abs() + 1  # +1 para evitar tamanho zero
    
    # Gráfico de dispersão com tendência
    fig = px.scatter(
        df.reset_index(),
        x='sigilosos_2023',
        y='sigilosos_2024',
        size='tamanho_marcador',  # Usar a coluna corrigida
        color='crescimento_rel',
        hover_name='oab',
        trendline='ols',
        title='Relação entre Processos Sigilosos (2023 vs 2024)',
        labels={
            'sigilosos_2023': 'Processos Sigilosos em 2023',
            'sigilosos_2024': 'Processos Sigilosos em 2024',
            'crescimento_rel': 'Crescimento (%)',
            'tamanho_marcador': 'Magnitude do Crescimento'
        },
        size_max=20  # Limitar o tamanho máximo dos marcadores
    )
    
    # Personalizar a legenda de cores
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Crescimento (%)",
            tickvals=[-100, -50, 0, 50, 100, 150, 200],
            ticktext=["-100%", "-50%", "0%", "50%", "100%", "150%", "200%"]
        )
    )
    
    fig.show()
    
    # [O resto da função permanece igual...]
    # Tabela de resultados
    resultados = df.reset_index()[
        ['oab', 'sigilosos_2022', 'sigilosos_2023', 'sigilosos_2024', 
         'crescimento_rel', 'previsao_2024']
    ].sort_values('crescimento_rel', ascending=False)
    
    fig_tabela = go.Figure(data=[go.Table(
        header=dict(
            values=['OAB', '2022', '2023', '2024', 'Crescimento (%)', 'Previsão 2024'],
            fill_color='#203864',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                resultados['oab'],
                resultados['sigilosos_2022'],
                resultados['sigilosos_2023'],
                resultados['sigilosos_2024'],
                resultados['crescimento_rel'].round(2),
                resultados['previsao_2024'].round(2)
            ],
            fill_color='lavender',
            align='left'
        )
    )])
    fig_tabela.update_layout(
        title='Resultados da Análise por Advogado',
        height=800,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig_tabela.show()
    
    return df

df_resultados = visualizar_resultados(df_temporal, modelo)

# 7) Resumo estatístico
print("\n=== RESUMO DO MODELO ===")
print(modelo.summary())

