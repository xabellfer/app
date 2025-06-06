import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Cargar y preparar datos ---
df = pd.read_csv("pip (4).csv")
df_national = df[df['reporting_level'] == 'national']

# --- Diccionarios y listas ---
traducciones = {
    'headcount': 'Tasa de pobreza (headcount)',
    'poverty_gap': 'Brecha de pobreza',
    'poverty_severity': 'Severidad de la pobreza',
    'watts': 'Índice de Watts',
    'mean': 'Ingreso promedio',
    'median': 'Ingreso mediano',
    'mld': 'Desviación Logarítmica Media',
    'gini': 'Índice de Gini',
    'reporting_gdp': 'PIB per cápita'
}

traducciones_inv = {v: k for k, v in traducciones.items()}
años = ["Todos los años"] + sorted(df['reporting_year'].dropna().unique().astype(int))
variables = [v for v in traducciones if v in df.columns]
variables_traducidas = [traducciones[v] for v in variables]
regiones = sorted(df['region_name'].unique())
paises = sorted(df['country_name'].unique())
indicadores = [v for v in ['headcount', 'poverty_gap', 'poverty_severity', 'watts', 'gini', 'mean'] if v in df.columns]
indicadores_traducidos = [traducciones[v] for v in indicadores]

parejas_variables = [
    ("headcount", "mean"), ("headcount", "median"), ("gini", "headcount"),
    ("gini", "poverty_gap"), ("mld", "headcount"), ("mean", "gini"),
    ("median", "gini"),
]
etiquetas_parejas = {
    f"{traducciones[x]} vs {traducciones[y]}": (x, y) for x, y in parejas_variables
}

explicaciones_parejas = {
    "Tasa de pobreza (headcount) vs Ingreso promedio":
        "Evalúa si los países con mayor ingreso promedio tienen menor pobreza. A mayor ingreso medio, se espera una menor proporción de personas bajo la línea de pobreza.",
    "Tasa de pobreza (headcount) vs Ingreso mediano":
        "Analiza si la mediana del ingreso, que representa mejor al individuo típico, se asocia con menores niveles de pobreza.",
    "Índice de Gini vs Tasa de pobreza (headcount)":
        "Explora cómo la desigualdad en la distribución del ingreso influye en la proporción de personas pobres en una población.",
    "Índice de Gini vs Brecha de pobreza":
        "Relaciona la desigualdad con la profundidad de la pobreza: más desigualdad puede llevar a una mayor brecha para superar la pobreza.",
    "Desviación Logarítmica Media vs Tasa de pobreza (headcount)":
        "Evalúa si la desigualdad con mayor peso en los más pobres se asocia con un mayor porcentaje de personas pobres.",
    "Ingreso promedio vs Índice de Gini":
        "Explora si el crecimiento económico (ingreso medio) se asocia con menor o mayor desigualdad.",
    "Ingreso mediano vs Índice de Gini":
        "Evalúa si el ingreso del ciudadano promedio se ve afectado por la concentración de riqueza en los extremos.",
}

# --- NUEVA PESTAÑA: RESUMEN VISUAL GLOBAL ---
... # todo el código igual hasta dentro de la función resumen_visual()

def resumen_visual(año, variable):
    df_filtrado = df[df['reporting_year'] == año].copy()

    # Seleccionar sólo el reporting_level dominante por país
    df_filtrado['dominante'] = df_filtrado.groupby('country_name')['reporting_level'].transform(lambda x: x.mode()[0])
    df_filtrado = df_filtrado[df_filtrado['reporting_level'] == df_filtrado['dominante']]

    df_var = df_filtrado[["country_name", "region_name", variable]].dropna()
    df_var = df_var[df_var[variable] > 0]  # Filtrar solo valores mayores que cero

    # Treemap por región
    df_region = df_var.groupby("region_name").agg({
        variable: ['mean'],
        'country_name': 'count'
    }).reset_index()
    df_region.columns = ["region_name", "valor_medio", "n_paises"]

    fig_treemap = px.treemap(
        df_region,
        path=["region_name"],
        values="n_paises",
        color="valor_medio",
        color_continuous_scale="Tealgrn",
        labels={"valor_medio": traducciones[variable]},
        hover_data={"valor_medio": ':.4f'}
    )
    fig_treemap.update_layout(title=f"Relación {traducciones[variable]}/Desigualdad")

    # Barplot por país (todos los países con valor > 0)
    df_var_sorted = df_var.sort_values(by=variable, ascending=False)
    fig_bar = px.bar(
        df_var_sorted,
        x=variable,
        y="country_name",
        orientation="h",
        color=variable,
        color_continuous_scale="Blues",
        labels={"country_name": "País", variable: traducciones[variable]},
    )
    fig_bar.update_layout(
        title="Comparación de pobreza entre países",
        yaxis=dict(autorange="reversed")
    )

    return fig_treemap, fig_bar
r

# --- Gráficos interactivos con Plotly ---
def plot_box_hist_violin(df_base, variable, tipo):
    fig = None
    if tipo == "Boxplot":
        fig = px.box(df_base, y=variable, points="all", color_discrete_sequence=["#3D9970"])
    elif tipo == "Histograma":
        fig = px.histogram(df_base, x=variable, nbins=30, color_discrete_sequence=["#0074D9"])
    elif tipo == "Gráfico de Violín":
        fig = px.violin(df_base, y=variable, box=True, points="all", color_discrete_sequence=["#FF851B"])

    fig.update_layout(
        title=f"{tipo} de {traducciones[variable]}",
        height=500,
        template="plotly_white"
    )
    return fig

def evolucion_pais(pais, indicador):
    df_pais = df[df['country_name'] == pais]
    if df_pais.empty:
        return None, pd.DataFrame()
    reporting_level = df_pais['reporting_level'].mode()[0]
    df_pais = df_pais[df_pais['reporting_level'] == reporting_level]
    df_pais = df_pais.sort_values('reporting_year')
    df_filtrado = df_pais[['reporting_year', indicador]].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtrado['reporting_year'],
        y=df_filtrado[indicador],
        mode='lines+markers',
        marker=dict(color='blue'),
        name=traducciones[indicador]
    ))
    fig.update_layout(
        title=f"Evolución de {traducciones[indicador]} en {pais} ({reporting_level})",
        xaxis_title="Año",
        yaxis_title=traducciones[indicador],
        height=500,
        template="plotly_white"
    )
    return fig, df_filtrado.rename(columns={'reporting_year': 'Año', indicador: 'Valor'})

def graficar_relacion_variables_seleccion(x_var, y_var):
    df_plot = df_national[[x_var, y_var, 'region_name']].dropna()
    fig = px.scatter(
        df_plot, x=x_var, y=y_var, color='region_name',
        title=f"{traducciones[x_var]} vs {traducciones[y_var]}",
        labels={x_var: traducciones[x_var], y_var: traducciones[y_var]},
        template="plotly_white"
    )
    return fig

def crear_matriz_correlacion():
    corr_vars = df_national[variables].dropna()
    corr_matrix = corr_vars.corr().round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlación"),
        x=[traducciones[v] for v in corr_matrix.columns],
        y=[traducciones[v] for v in corr_matrix.index],
        title="Matriz de correlación",
        template="plotly_white"
    )
    return fig

def crear_mapa_mundial(variable, año):
    df_filtrado = df[
        (df['reporting_level'] == 'national') &
        (df['reporting_year'] == int(año)) &
        (df[variable].notna())
    ][['country_name', variable]]

    fig = px.choropleth(
        df_filtrado,
        locations='country_name',
        locationmode='country names',
        color=variable,
        hover_name='country_name',
        color_continuous_scale='Viridis',
        title=f'{traducciones[variable]} en {año}',
        labels={variable: traducciones[variable]},
        template="plotly_white"
    )
    fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
    return fig

# --- Interfaz Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Pobreza Global")
st.title("🌍 Análisis de Pobreza Global")

tabs = st.tabs(["📊 Gráficos por Año", "📈 Evolución por País", "🧩 Resumen Visual", "🔗 Relaciones", "🗺️ Mapa Mundial"])

with tabs[0]:
    st.subheader("Visualización por Año y Variable")
    col1, col2, col3 = st.columns(3)
    tipo = col1.selectbox("Tipo de gráfico", ["Boxplot", "Histograma", "Gráfico de Violín"])
    año = col2.selectbox("Año", años)
    variable_trad = col3.selectbox("Variable", variables_traducidas)
    variable = traducciones_inv[variable_trad]
    df_base = df if año == "Todos los años" else df[df['reporting_year'] == int(año)]
    st.plotly_chart(plot_box_hist_violin(df_base, variable, tipo), use_container_width=True)

with tabs[1]:
    st.subheader("Evolución temporal por país")
    pais = st.selectbox("País", paises)
    indicador_trad = st.selectbox("Indicador", indicadores_traducidos)
    indicador = traducciones_inv[indicador_trad]
    fig, tabla = evolucion_pais(pais, indicador)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tabla)

with tabs[2]:
    st.subheader("Resumen visual global")
    col1, col2 = st.columns(2)
    variable_trad = col1.selectbox("Variable para resumen", variables_traducidas, key="resumen_var")
    año_resumen = col2.selectbox("Año", sorted(df['reporting_year'].dropna().unique().astype(int)), key="resumen_año")
    variable = traducciones_inv[variable_trad]
    fig_treemap, fig_bar = resumen_visual(año_resumen, variable)
    st.plotly_chart(fig_treemap, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with tabs[3]:
    st.subheader("Relación entre variables")
    sub_tabs = st.tabs(["Gráfica de dispersión", "Matriz de correlación"])

    with sub_tabs[0]:
        relacion_trad = st.selectbox("Relación", list(etiquetas_parejas.keys()))
        x_var, y_var = etiquetas_parejas[relacion_trad]
        st.plotly_chart(graficar_relacion_variables_seleccion(x_var, y_var), use_container_width=True)
        st.markdown(f"**Explicación:** {explicaciones_parejas.get(relacion_trad, '')}")

    with sub_tabs[1]:
        st.plotly_chart(crear_matriz_correlacion(), use_container_width=True)

with tabs[4]:
    st.subheader("Mapa mundial por variable")
    col1, col2 = st.columns(2)
    variable_trad = col1.selectbox("Variable para mapa", variables_traducidas)
    año_map = col2.selectbox("Año", sorted(df['reporting_year'].dropna().unique().astype(int)))
    variable = traducciones_inv[variable_trad]
    st.plotly_chart(crear_mapa_mundial(variable, año_map), use_container_width=True)
