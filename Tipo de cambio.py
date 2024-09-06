import requests
import pandas as pd
import matplotlib.pyplot as plt

def obtener_tipo_cambio_banxico(token, fecha_inicio, fecha_fin):
    # URL de la API de Banxico para el tipo de cambio FIX del dólar
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF63528/datos/{fecha_inicio}/{fecha_fin}"

    # Encabezados con el token de autenticación
    headers = {
        "Bmx-Token": token
    }

    try:
        # Realiza la solicitud GET a la API de Banxico
        response = requests.get(url, headers=headers)

        # Verifica si hay errores HTTP
        response.raise_for_status()

        # Obtiene los datos en formato JSON
        data = response.json()

        # Extrae los datos de tipo de cambio y fechas
        datos = data['bmx']['series'][0]['datos']
        fechas = [item['fecha'] for item in datos]
        tipos_cambio = [float(item['dato']) for item in datos]

        return fechas, tipos_cambio
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"Error de conexión: {err}")
    except KeyError:
        print("Error al procesar los datos de respuesta, verifica la estructura de la respuesta JSON.")
    except Exception as e:
        print("Error al obtener el tipo de cambio:", e)

# Usa tu token de la API
api_key = "ef39fd41f5fda6b7c7dc1783cf830d962a5487a1f8146ee8ac0c4850c44d728d"
fecha_inicio = "2024-08-02"  # Fecha de inicio en formato YYYY-MM-DD
fecha_fin = "2024-09-05"     # Fecha de fin en formato YYYY-MM-DD

fechas, tipos_cambio = obtener_tipo_cambio_banxico(api_key, fecha_inicio, fecha_fin)

# Convertir las fechas a formato datetime
fechas = pd.to_datetime(fechas, format='%d/%m/%Y')

# Crear un DataFrame
df = pd.DataFrame({
    'Fecha': fechas,
    'Tipo de Cambio': tipos_cambio
})

# Graficar los datos
plt.figure(figsize=(10, 6))
plt.plot(df['Fecha'], df['Tipo de Cambio'], marker='o', linestyle='-', color='b')
plt.title('Tipo de Cambio USD/MXN Diario')
plt.xlabel('Fecha')
plt.ylabel('Tipo de Cambio')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

