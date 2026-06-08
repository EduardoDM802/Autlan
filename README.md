# Dashboard de Cobertura AUTLAN

Dashboard ejecutivo en Streamlit para analizar el modelo Monte Carlo de cobertura de Autlan.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app/montecarlo_dashboard.py
```

## Entrada de datos

La aplicacion usa archivos locales versionados en el repositorio:

- `gold_forecasts.csv`
- `gold_historical_prices.csv`
- `usdmxn_forwards_forecasts.csv`
- `usdmxn_historical_prices.csv`
- `LOGOS/tec_logo.png`
- `LOGOS/auntlan_logo.png`

No depende de una conexion activa con Refinitiv para correr en produccion.

## Despliegue recomendado

Streamlit Community Cloud:

- Repository: este repositorio en GitHub
- Branch: `main`
- Main file path: `app/montecarlo_dashboard.py`
- Python: 3.12

## Nota sobre Vercel

La app es Streamlit y requiere el runtime interactivo de Streamlit. Vercel Python Functions estan orientadas a apps ASGI/WSGI como FastAPI, Flask o Django, por lo que Streamlit Community Cloud es la plataforma adecuada para publicar este dashboard sin cambiar su arquitectura.
