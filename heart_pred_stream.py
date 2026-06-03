# ================ #
# CARGAR LIBRERÍAS #
# ================ #
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ====================== #
# CONFIGURACIÓN DE RUTAS #
# ====================== #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "heart_disease_features.joblib"

# ============= #
# CARGAR MODELO #
# ============= #
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# ========================= #
# EXTRAER INFO DEL PIPELINE #
# ========================= #
preprocessor = model.named_steps['preprocess']

num_cols = next(
    cols for name, transformer, cols in preprocessor.transformers_
    if name == 'num'
)

cat_cols = next(
    cols for name, transformer, cols in preprocessor.transformers_
    if name == 'cat'
)

encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorias = encoder.categories_

cat_values = {
    col: list(vals)
    for col, vals in zip(cat_cols, categorias)
}

# ============================== #
# CONFIGURACIÓN DE LA APLICACIÓN #
# ============================== #
st.set_page_config(
    page_title="Predicción de enfermedad cardíaca",
    layout="wide"
)

# ======================= #
# ESTILO DE LA APLICACIÓN #
# ======================= #
st.markdown("""
<style>
.big-title {
    font-size:100px;
    font-weight:700;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="font-size:45px; font-weight:800; text-align:center;">Predicción de Enfermedad Cardíaca ❤️</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="text-align:justify; font-size:18px; font-weight:400;">Las enfermedades cardiovasculares representan una de las principales causas de mortalidad a nivel mundial, siendo responsables de aproximadamente 17.9 millones de muertes cada año, lo que equivale al 31% de las muertes globales. Entre las principales causas se encuentran los ataques cardíacos y los accidentes cerebrovasculares, afectando incluso a personas menores de 70 años. Debido a esto, la detección temprana de factores de riesgo cardiovasculares es fundamental para prevenir complicaciones y mejorar la calidad de vida de los pacientes.</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="text-align:center; font-size:25px; font-weight:400;">Complete la información para estimar el riesgo de enfermedad cardíaca</p>',
    unsafe_allow_html=True
)

# ======================================================= #
# CONFIGURACIÓN DE FILTROS PARA VARIABLES NUMÉRICAS      #
# Ajusta min_value, max_value, step y value por columna  #
# ======================================================= #
NUM_CONFIG = {
    "Age":      {"min_value": 20.0,   "max_value": 100.0, "step": 1.0,   "value": 40.0},
    "RestingBP": {"min_value": 0.0,  "max_value": 220.0, "step": 1.0,   "value": 120.0},
    "Cholesterol":     {"min_value": 0.0, "max_value": 700.0, "step": 1.0,   "value": 200.0},
    "MaxHR":  {"min_value": 60.0,  "max_value": 220.0, "step": 1.0,   "value": 150.0},
    "Oldpeak":  {"min_value": 0.0, "max_value": 10.0, "step": 0.1, "value": 0.0},
    "FastingBS":  {"min_value": 0.0, "max_value": 1.0, "step": 1.0, "value": 0.0},
}

# ============================================= #
# ENTRADAS DE LA BARRA LATERAL (SIDEBAR INPUTS) #
# ============================================= #
st.sidebar.markdown(
    '<p style="font-size:22px; font-weight:700;">📊 Variables de Entrada</p>',
    unsafe_allow_html=True
)

input_data = {}

for col in features:

    # Numéricas
    if col in num_cols:
        cfg = NUM_CONFIG.get(col, {})
        input_data[col] = st.sidebar.number_input(
            col,
            min_value=cfg.get("min_value"),
            max_value=cfg.get("max_value"),
            step=cfg.get("step"),
            value=cfg.get("value", 0.0),
        )

    # Categóricas
    elif col in cat_cols:
        input_data[col] = st.sidebar.selectbox(
            col,
            cat_values[col]
        )

# ===================== #
# PREDICCIÓN DEL MODELO #
# ===================== #
if st.sidebar.button("🔮 Predecir"):

    df = pd.DataFrame([input_data])

    try:
        pred = model.predict(df)[0]
        enfermedad = "Sí" if pred == 1 else "No"

        st.success(f"❤️ Predicción de enfermedad cardíaca: {enfermedad}")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")

# ============= #
# MOSTRAR INPUT #
# ============= #

df_input = pd.DataFrame([input_data]).T
df_input.columns = ["Valor"]
df_input["Valor"] = df_input["Valor"].astype(str)

styled = df_input.style \
    .set_properties(**{
        'text-align': 'left',
        'font-size': '14px'
    }) \
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('font-weight', 'bold'), ('text-align', 'left')]},
        {'selector': 'td', 'props': [('padding', '6px')]}
    ])

st.markdown(
    '<p style="text-align:center;font-size:22px; font-weight:700;">📄 Datos Ingresados</p>',
    unsafe_allow_html=True
)
st.dataframe(styled, height=420)

# ======================================== #
#        Ejecución desde la Terminal       #
# streamlit run src/housing_pred_stream.py #
# ======================================== #
