#CONFIGURACION DEL ENTORNO
#pip install pandas numpy matplotlib seaborn statsmodels scipy yfina nce requests beautifulsoup4 fbprophet
#Bibliotecas y Configuración Inicial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from fbprophet import Prophet
#EXTRACCIONES DE DATOS
#Lista de índices bursátiles y sus correspondientes tickers en Yahoo Finance
indices_tickers = {
    'SP500': '^GSPC',            # S&P 500
    'Nasdaq': '^IXIC',           # Nasdaq Composite
    'DowJones': '^DJI',          # Dow Jones Industrial Average
    'FTSE100': '^FTSE',          # FTSE 100
    'DAX': '^GDAXI',             # DAX (Alemania)
    'CAC40': '^FCHI',            # CAC 40 (Francia)
    'Nikkei225': '^N225',        # Nikkei 225 (Japón)
    'HangSeng': '^HSI',          # Hang Seng (Hong Kong)
    'ASX200': '^AXJO',           # ASX 200 (Australia)
    'ShanghaiComposite': '000001.SS'  # Shanghai Composite (China)
}
dataframes = {}
for index_name, ticker in indices_tickers.items():
    dataframes[index_name] = yf.download(ticker, start='2015-01-01', end='2024-08-15')
for index_name, df in dataframes.items():
    df.rename(columns={'Close': f'{index_name}_Close'}, inplace=True)
combined_df = pd.DataFrame()
for index_name, df in dataframes.items():
    if combined_df.empty:
        combined_df = df[[f'{index_name}_Close']]
    else:
        combined_df = combined_df.join(df[[f'{index_name}_Close']], how='outer')
print(combined_df.head())
combined_df.to_csv('indices_bursatiles.csv', index=True)
combined_df.plot(figsize=(14, 8), title='Evolución de los Índices Bursátiles Globales', grid=True)
corr_matrix = df_combined.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación de Índices Bursátiles')
plt.show()

# Configuración de la clave API de FRED
api_key = '6bdceec24a7e130561ca124e5c935e07'
def get_inflation_data():
    # Leer datos de inflación usando pandas_datareader
    inflation_df = pdr.get_data_fred('CPIAUCNS', api_key=api_key)
    inflation_df.reset_index(inplace=True)
    inflation_df.columns = ['Date', 'Inflation']
    return inflation_df
inflation_df = get_inflation_data()
print(inflation_df.head())
def get_gdp_data():
    # Leer datos de PIB usando pandas_datareader
    gdp_df = pdr.get_data_fred('GDP', api_key=api_key)
    gdp_df.reset_index(inplace=True)
    gdp_df.columns = ['Date', 'GDP']
    return gdp_df
gdp_df = get_gdp_data()
X = sm.add_constant(gdp_df['Interest_Rate'])
model = sm.OLS(gdp_df['GDP'], X).fit()
X = gdp_df[['Interest_Rate', 'Inflation']]
X = sm.add_constant(X)
model = sm.OLS(gdp_df['GDP'], X).fit()
print(model.summary())
print(model.summary())

# Modelos ARIMA para predicción del PIB
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(gdp_df['GDP'], order=(5,1,0))
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())
# Predicción con fbprophet
gdp_df_prophet = gdp_df[['Date', 'GDP']]
gdp_df_prophet.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(gdp_df_prophet)
future = model_prophet.make_future_dataframe(periods=365)
forecast = model_prophet.predict(future)
model_prophet.plot(forecast)
plt.title('Predicción del PIB con fbprophet')
plt.show()

# Correlación cruzada antes y después de una crisis financiera
pre_crisis = df_combined[df_combined.index < '2020-03-01']
post_crisis = df_combined[df_combined.index >= '2020-03-01']
correlation_pre_crisis = pre_crisis.corr()
correlation_post_crisis = post_crisis.corr()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(correlation_pre_crisis, annot=True, cmap='coolwarm', ax=ax[0])
ax[0].set_title('Correlación Pre-Crisis')
sns.heatmap(correlation_post_crisis, annot=True, cmap='coolwarm', ax=ax[1])
ax[1].set_title('Correlación Post-Crisis')
plt.show()

df_combined.index = pd.to_datetime(df_combined.index)
gdp_df['Date'] = pd.to_datetime(gdp_df['Date'])
inflation_df['Date'] = pd.to_datetime(inflation_df['Date'])
df_combined = df_combined.merge(gdp_df, left_index=True, right_on='Date', how='left')
df_combined = df_combined.merge(inflation_df, left_index=True, right_on='Date', how='left')
plt.figure(figsize=(12, 7))
plt.plot(df_combined['Date'], df_combined['GDP'], label='PIB')
plt.plot(df_combined.index, df_combined['S&P 500'], label='S&P 500')
plt.title('PIB vs S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.show()


# Simulación Monte Carlo para pronosticar crecimiento del PIB
simulations = 10000
simulated_gdp_growth = np.zeros(simulations)
for i in range(simulations):
    simulated_gdp_growth[i] = np.random.normal(loc=gdp_df['GDP'].mean(), scale=gdp_df['GDP'].std())
plt.figure(figsize=(10,6))
sns.histplot(simulated_gdp_growth, kde=True)
plt.title('Simulación Monte Carlo del Crecimiento del PIB')
plt.xlabel('Crecimiento del PIB Simulado')
plt.ylabel('Frecuencia')
plt.show()


# Commodities
oil = yf.download('CL=F', start='2015-01-01', end='2024-08-15')
gold = yf.download('GC=F', start='2015-01-01', end='2024-08-15')
plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
sns.lineplot(data=oil, x='Date', y='Close', label='Petróleo')
plt.title('Evolución del Precio del Petróleo')
plt.subplot(2, 1, 2)
sns.lineplot(data=gold, x='Date', y='Close', label='Oro', color='orange')
plt.title('Evolución del Precio del Oro')
plt.tight_layout()
plt.show()













