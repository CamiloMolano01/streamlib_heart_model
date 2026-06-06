import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency


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
    # Selector para controlar cuántas filas mostrar en la vista previa
    max_rows = min(100, len(df)) if len(df) > 0 else 5
    default_rows = min(10, max_rows)
    num_preview = st.slider("Filas a mostrar en la vista previa", min_value=5, max_value=max_rows, value=default_rows, step=5)
    height = min(600, 30 * num_preview)
    st.dataframe(df.head(num_preview), height=height)

    # Histograma por categoría vs target
    # Detectar columnas categóricas disponibles
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    all_cols = df.columns.tolist()
    if cat_cols:
        st.markdown("---")
        st.markdown("### Histograma: Distribución por categoría")
        st.write("")
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
    st.markdown("---")
    st.markdown("### Matriz de correlación")
    st.write("")
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

    # Spearman correlation matrix (dynamic selection)
    st.markdown("---")
    st.markdown("### Matriz de correlación (Spearman)")
    st.write("")
    try:
        numeric_all = df.select_dtypes(include=["number"]).columns.tolist()
        default_numeric = [c for c in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'] if c in numeric_all]
        sel_numeric = st.multiselect("Selecciona variables numéricas", numeric_all, default=default_numeric if default_numeric else numeric_all[:5])

        if len(sel_numeric) >= 2:
            corr_spearman = df[sel_numeric].corr(method='spearman')

            # Plotly heatmap (interactive)
            fig_corr = px.imshow(
                corr_spearman,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                labels=dict(x='feature', y='feature', color='spearman')
            )
            fig_corr.update_layout(title='Matriz de Correlación de Spearman', height=600)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Dataframe + download
            st.markdown('**Tabla de correlaciones (Spearman)**')
            st.dataframe(corr_spearman, height=300)
            csv_bytes = corr_spearman.to_csv().encode('utf-8')
            st.download_button('Descargar Spearman CSV', csv_bytes, file_name='spearman_correlation.csv', mime='text/csv')
        else:
            st.info('Selecciona al menos dos variables numéricas para calcular la correlación de Spearman.')
    except Exception as e:
        st.error(f'Error al generar la matriz de Spearman: {e}')

    # Violin plot: elegir columna numérica (y) y columna objetivo (x/color)
    st.markdown("---")
    st.markdown("### Violin plot: Distribución numérica por categoría")
    st.write("")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        default_y = "Age" if "Age" in numeric_cols else numeric_cols[0]
        y_col = st.selectbox("Columna numérica (eje Y)", numeric_cols, index=numeric_cols.index(default_y))

        default_x = target_col if 'target_col' in locals() else ("HeartDisease" if "HeartDisease" in df.columns else df.columns[-1])
        x_col = st.selectbox("Columna categórica (eje X / color)", df.columns.tolist(), index=df.columns.tolist().index(default_x))

        try:
            fig_v = px.violin(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                box=True,
                points='outliers',
                color_discrete_sequence=['#66C2A5', '#FC8D62']
            )

            fig_v.update_traces(
                meanline_visible=True,
                opacity=0.85,
                hovertemplate='<b>' + x_col + '</b>: %{x}<br><b>' + y_col + '</b>: %{y}<extra></extra>'
            )

            fig_v.update_layout(
                template='plotly_white',
                title={
                    'text': f'Violin Plot de {y_col} según {x_col}',
                    'x': 0.5,
                    'font': {'size': 22, 'family': 'Arial', 'color': '#333'}
                },
                xaxis_title=f'{x_col}',
                yaxis_title=f'{y_col}',
                width=950,
                height=550,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial', size=14, color='#444'),
                legend_title=x_col
            )

            fig_v.update_xaxes(showline=True, linewidth=2, linecolor='lightgray')
            fig_v.update_yaxes(showline=True, linewidth=2, linecolor='lightgray', gridcolor='rgba(200,200,200,0.3)')

            st.plotly_chart(fig_v, use_container_width=True)

        except Exception as e:
            st.error(f"No se pudo generar el violin plot: {e}")

    # Cramér's V dinámico entre variables categóricas
    st.markdown("---")
    st.markdown("### Cramérs V y tablas de contingencia")
    st.write("")

    # Considerar como categóricas también las columnas numéricas de baja cardinalidad
    num_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    num_as_cat = [c for c in num_cols_all if df[c].nunique() <= 10]
    cat_columns_all = df.select_dtypes(exclude=["number"]).columns.tolist() + num_as_cat
    # mantener orden y unicidad
    cat_columns_all = list(dict.fromkeys(cat_columns_all))
    default_vars = [v for v in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease'] if v in cat_columns_all]
    vars_selected = st.multiselect("Selecciona variables categóricas para comparar", cat_columns_all, default=default_vars if default_vars else cat_columns_all[:3])

    def cramers_v(tabla):
        chi2 = chi2_contingency(tabla)[0]
        n = tabla.sum().sum()
        r, k = tabla.shape
        return np.sqrt(chi2 / (n * min(r - 1, k - 1))) if n > 0 and min(r - 1, k - 1) > 0 else np.nan

    def interpretar_cramer(v):
        if np.isnan(v):
            return 'N/A'
        if v < 0.10:
            return 'Muy débil'
        elif v < 0.30:
            return 'Débil'
        elif v < 0.50:
            return 'Moderada'
        elif v < 0.70:
            return 'Fuerte'
        else:
            return 'Muy fuerte'

    if len(vars_selected) >= 2:
        resultados = []
        for var1, var2 in combinations(vars_selected, 2):
            try:
                tabla = pd.crosstab(df[var1], df[var2])
                chi2, p, _, _ = chi2_contingency(tabla)
                v = cramers_v(tabla)
                fuerza = interpretar_cramer(v)

                resultados.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Chi2': round(float(chi2), 4),
                    'p-value': round(float(p), 6),
                    "Cramer's V": round(float(v) if not np.isnan(v) else np.nan, 3),
                    'Dependencia': 'Sí' if p < 0.05 else 'No',
                    'Fuerza': fuerza
                })

                # Mostrar heatmap dentro de un expander para mantener la vista compacta
                with st.expander(f'Tabla: {var1} vs {var2} — V={v:.3f} | p={p:.5f}'):
                    # Convertir tabla a matriz numérica para plotly
                    try:
                        ct = tabla.copy()
                        fig_ct = px.imshow(
                            ct,
                            text_auto=True,
                            color_continuous_scale='YlOrRd',
                        )
                        fig_ct.update_layout(
                            title=f'{var1} vs {var2} — V={v:.3f} | p={p:.5f}',
                            xaxis_title=var2,
                            yaxis_title=var1,
                            height=500
                        )
                        fig_ct.update_traces(hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Count: %{z}<extra></extra>')
                        st.plotly_chart(fig_ct, use_container_width=True)
                    except Exception as e:
                        st.error(f'No se pudo mostrar la tabla interactiva para {var1} vs {var2}: {e}')
            except Exception as e:
                resultados.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Chi2': None,
                    'p-value': None,
                    "Cramer's V": None,
                    'Dependencia': f'Error: {e}'
                })

        # Mostrar resumen de resultados y permitir descarga
        df_res = pd.DataFrame(resultados).sort_values(by='p-value', na_position='last')
        st.markdown('**Resumen Chi-cuadrado (pares seleccionados)**')
        st.dataframe(df_res, height=min(400, 60 * len(df_res)))
        csv_bytes = df_res.to_csv(index=False).encode('utf-8')
        st.download_button('Descargar resultados Chi-cuadrado CSV', csv_bytes, file_name='chi2_results.csv', mime='text/csv')
    else:
        st.info('Selecciona al menos dos variables categóricas para calcular Cramér\'s V.')
