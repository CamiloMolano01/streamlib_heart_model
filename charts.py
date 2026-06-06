import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


@st.cache_data
def _load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def render_dataset_charts(data_path: str | Path) -> None:
    """Load dataset from `data_path` and render dataset-level charts.

    Currently shows a table preview and a correlation heatmap (Plotly).
    """
    path = Path(data_path)
    if not path.exists():
        st.error(f"Dataset not found: {path}")
        return

    df = _load_dataframe(path)

    st.markdown("**Vista previa del dataset**")
    st.dataframe(df.head(), height=240)

    # Histograma por categoría vs target
    # Detectar columnas categóricas disponibles
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    all_cols = df.columns.tolist()
    if cat_cols:
        default_cat = "ChestPainType" if "ChestPainType" in cat_cols else cat_cols[0]
        cat_col = st.selectbox("Columna categórica", cat_cols, index=cat_cols.index(default_cat))

        # target (color) column
        default_target = "HeartDisease" if "HeartDisease" in all_cols else all_cols[-1]
        target_col = st.selectbox("Columna objetivo (color)", all_cols, index=all_cols.index(default_target))

        # Construir histograma estilo Plotly (adaptado de la plantilla)
        try:
            data = df
            fig = px.histogram(
                data,
                x=cat_col,
                color=target_col,
                text_auto=True,
                color_discrete_sequence=['#66C2A5', '#FC8D62']
            )

            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside',
                marker_line_color='white',
                marker_line_width=2,
                opacity=0.9
            )

            fig.update_layout(
                template='plotly_white',
                title={
                    'text': f'Distribución de {target_col} según {cat_col}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'family': 'Arial', 'color': '#333'}
                },
                xaxis_title=cat_col,
                yaxis_title='Frecuencia',
                width=950,
                height=550,
                bargap=0.25,
                legend_title=target_col,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial', size=14, color='#444')
            )

            fig.update_xaxes(showline=True, linewidth=2, linecolor='lightgray')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='lightgray', gridcolor='rgba(200,200,200,0.3)')

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"No se pudo generar el histograma: {e}")


    # Correlation heatmap
    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        st.info("No hay suficientes columnas numéricas para calcular correlaciones.")
        return

    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(x="feature", y="feature", color="correlation")
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
