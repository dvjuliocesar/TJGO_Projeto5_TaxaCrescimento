# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from datetime import datetime

# Configurações Iniciais
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# Carregar dados
df = pd.read_csv('uploads/dados_processos_sigilosos_2022-01-01_a_2025-06-30.csv', 
                 parse_dates=['data_distribuicao', 'data_baixa'])

# Verificar dados
print(df.head())
print(df.info())







