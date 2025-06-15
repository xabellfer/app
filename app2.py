import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pycountry

# --- Cargar y preparar datos ---
df = pd.read_csv("datos_pobreza_filtrados.csv")
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


# --- Funciones para gráficas ---
def violin_plot(df_base, variable, variable_trad):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(y=df_base[variable], ax=ax, inner='quartile', color='skyblue')
    ax.set_title(f"Diagrama de Violín de {variable_trad}")
    ax.set_ylabel(variable_trad)
    ax.set_xlabel("")
    return fig

def box_plot(df_base, variable, variable_trad):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(y=df_base[variable], ax=ax, color='lightgreen')
    ax.set_title(f"Boxplot de {variable_trad}")
    ax.set_ylabel(variable_trad)
    return fig

def histograma(df_base, variable, variable_trad):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(df_base[variable].dropna(), kde=False, bins=30, color='skyblue', ax=ax)
    ax.set_title(f"Histograma de {variable_trad}")
    ax.set_xlabel(variable_trad)
    ax.set_ylabel("Frecuencia")
    return fig

def evolucion_pais(pais, indicador):
    df_pais = df[df['country_name'] == pais]
    if df_pais.empty:
        return None, pd.DataFrame()
    reporting_level = df_pais['reporting_level'].mode()[0]
    df_pais = df_pais[df_pais['reporting_level'] == reporting_level]
    df_pais = df_pais.sort_values('reporting_year')
    df_filtrado = df_pais[['reporting_year', indicador]].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_filtrado['reporting_year'], df_filtrado[indicador], marker='o', color='blue')
    ax.set_title(f"Evolución de {traducciones[indicador]} en {pais} a nivel '{reporting_level}'")
    ax.set_xlabel("Año")
    ax.set_ylabel(traducciones[indicador])
    return fig, df_filtrado.rename(columns={'reporting_year': 'Año', indicador: 'Valor'})

def graficar_relacion_variables_seleccion(x_var, y_var):
    df_plot = df_national[[x_var, y_var, 'region_name']].dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=df_plot, x=x_var, y=y_var, hue='region_name', palette='Set2', ax=ax)
    ax.set_title(f"{traducciones[x_var]} vs {traducciones[y_var]}")
    ax.set_xlabel(traducciones[x_var])
    ax.set_ylabel(traducciones[y_var])
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
        color_continuous_scale='Reds',
        title=f'{traducciones[variable]} en {año}',
        labels={variable: traducciones[variable]}
    )

    # Ajuste del tamaño del mapa
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=700  # <--- MÁS ALTO
    )

    return fig


# --- Interfaz Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Pobreza Global")
st.title("🌍 Análisis de Pobreza Global")

tabs = st.tabs(["📊 Gráficos Variables", "📈 Evolución por País", "📌 Comparación por País", "🔗 Relaciones", "🗺️ Mapa Mundial"])

with tabs[0]:
    st.subheader("Visualización por Año y Variable")
    col1, col2, col3 = st.columns(3)
    tipo = col1.selectbox("Tipo de gráfico", ["Boxplot", "Histograma", "Gráfico de Violín"])
    año = col2.selectbox("Año", años)
    variable_trad = col3.selectbox("Variable", variables_traducidas)
    variable = traducciones_inv[variable_trad]
    df_base = df if año == "Todos los años" else df[df['reporting_year'] == int(año)]
    if tipo == "Boxplot":
        st.pyplot(box_plot(df_base, variable, variable_trad))
    elif tipo == "Histograma":
        st.pyplot(histograma(df_base, variable, variable_trad))
    else:
        st.pyplot(violin_plot(df_base, variable, variable_trad))

with tabs[1]:
    st.subheader("Evolución temporal por país")
    pais = st.selectbox("País", paises)
    indicador_trad = st.selectbox("Indicador", indicadores_traducidos)
    indicador = traducciones_inv[indicador_trad]
    fig, tabla = evolucion_pais(pais, indicador)
    if fig:
        st.pyplot(fig)
        st.dataframe(tabla)

with tabs[3]:
    st.subheader("Relación entre variables")
    sub_tabs = st.tabs(["Gráfica de dispersión", "Matriz de correlación"])

    with sub_tabs[0]:
        relacion_trad = st.selectbox("Relación", list(etiquetas_parejas.keys()))
        x_var, y_var = etiquetas_parejas[relacion_trad]
        st.pyplot(graficar_relacion_variables_seleccion(x_var, y_var))
        st.markdown(f"**Explicación:** {explicaciones_parejas.get(relacion_trad, '')}")

    with sub_tabs[1]:
        st.markdown("### Matriz de correlación")
        corr_vars = df_national[variables].dropna()
        corr_matrix = corr_vars.corr()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=[traducciones[v] for v in corr_matrix.columns],
                    yticklabels=[traducciones[v] for v in corr_matrix.index],
                    ax=ax)
        st.pyplot(fig)


with tabs[2]:
    st.subheader("Comparación entre países por variable y año")
    col1, col2 = st.columns(2)
    año_seleccionado = col1.selectbox("Año", sorted(df['reporting_year'].dropna().unique().astype(int)), key="año_comp")
    variable_traducida = col2.selectbox("Variable", variables_traducidas, key="var_comp")
    variable = traducciones_inv[variable_traducida]

    df_año = df[df['reporting_year'] == año_seleccionado].copy()

    # Obtener reporting_level más común por país
    niveles_pred = df_año.groupby('country_name')['reporting_level'].agg(lambda x: x.mode()[0])
    df_año['nivel_dominante'] = df_año['country_name'].map(niveles_pred)
    df_filtrado = df_año[df_año['reporting_level'] == df_año['nivel_dominante']]

    # Filtrar valores válidos
    df_filtrado = df_filtrado[['country_name', 'region_name', variable]].dropna()
    df_filtrado = df_filtrado[df_filtrado[variable] > 0]

    # Treemap: promedio y máximo por región, con país incluido
    df_max = df_filtrado.sort_values(variable, ascending=False).groupby('region_name').first().reset_index()
    df_region = df_filtrado.groupby('region_name')[variable].mean().reset_index()
    df_region = df_region.merge(df_max[['region_name', 'country_name', variable]], on='region_name', suffixes=('_mean', '_max'))
    df_region['custom_label'] = df_region.apply(
        lambda row: f"{row['region_name']}<br>Promedio: {row[variable + '_mean']:.2f}<br>Máximo ({row['country_name']}): {row[variable + '_max']:.2f}",
        axis=1
    )

    fig_region = px.treemap(
        df_region,
        path=['region_name'],
        values=variable + '_mean',
        hover_data={
            variable + '_mean': False,
            variable + '_max': True,
        },
        color=variable + '_mean',
        color_continuous_scale='Viridis',
        title=f"{variable_traducida} - Promedio y país destacado por región ({año_seleccionado})"
    )
    fig_region.data[0].texttemplate = None  # Eliminar texto central ("promedio_sum")
    st.plotly_chart(fig_region, use_container_width=True)

    # Gráfico de barras por país (orden descendente)
    st.markdown("### Comparación entre países")
    df_ordenado = df_filtrado.sort_values(by=variable, ascending=False)
    fig_bar = px.bar(
        df_ordenado,
        x=variable,
        y='country_name',
        orientation='h',
        color='region_name',
        labels={variable: variable_traducida, 'country_name': 'País', 'region_name': 'Región'},
        height=700,
        hover_data={'region_name': True}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Media mundial ---
    media_mundial = df_filtrado[variable].mean()
    st.markdown("---")
    st.markdown(f"### 🌍 Media mundial de **{variable_traducida}** en {año_seleccionado}: **{media_mundial:.4f}**")




with tabs[4]:
    st.subheader("Mapa mundial por variable")
    col1, col2 = st.columns(2)
    variable_trad = col1.selectbox("Variable para mapa", variables_traducidas)
    año_map = col2.selectbox("Año", sorted(df['reporting_year'].dropna().unique().astype(int)))
    variable = traducciones_inv[variable_trad]
    st.plotly_chart(crear_mapa_mundial(variable, año_map), use_container_width=True)
