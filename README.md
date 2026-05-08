# ⚡ Electrification Readiness Map

State-level bivariate choropleth: **ACS B25117 heating fuel** × **EIA electricity rates**

Live: [your-app.streamlit.app](https://your-app.streamlit.app)

## What it shows

- **Fossil heat share** — % of occupied units using gas, fuel oil, or propane (ACS B25117)
- **Electricity rate** — residential avg ¢/kWh by state (EIA Retail Sales API)
- **Priority score** — 0.6 × fossil% + 0.4 × normalized rate
- **Bivariate view** — 3×3 color matrix encoding both dimensions simultaneously

## Local setup

```bash
git clone https://github.com/rmkenv/electrification
cd electrification
pip install -r requirements.txt

# Add API keys
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml with your keys

streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo + `app.py`
4. **Settings → Secrets** → paste:
```toml
CENSUS_API_KEY = "your_key"
EIA_API_KEY = "your_key"
```
5. Deploy — done

Keys are free:
- Census: https://api.census.gov/data/key_signup.html
- EIA: https://www.eia.gov/opendata/

## Author
Ryan Kmetz · [IQSpatial](https://github.com/rmkenv)
