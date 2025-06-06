import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pycountry

# --- Cargar y preparar datos ---
df = pd.read_csv("pip (4).csv")
# df_national = df[df['reporting_level'] == 'national'] # Mantendremos el df completo y filtraremos según la lógica requerida

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
años = [str(año) for año in sorted(df['reporting_year'].dropna().unique().astype(int))] # Convertir a string para el selectbox
años.insert(0, "Todos los años") # Añadir "Todos los años" al principio
variables = [v for v in traducciones if v in df.columns]
variables_traducidas = [traducciones[v] for v in variables]
regiones = sorted(df['region_name'].dropna().unique()) # Añadir dropna() por si hay nulos
paises = sorted(df['country_name'].dropna().unique()) # Añadir dropna()
indicadores = [v for v in ['headcount', 'poverty_gap', 'poverty_severity', 'watts',
                           'mean', 'median', 'mld', 'gini', 'reporting_gdp'] if v in df.columns] # Asegurarse de que estén en df.columns

etiquetas_parejas = {
    "Tasa de pobreza vs. Ingreso promedio": ("headcount", "mean"),
    "Tasa de pobreza vs. Desviación Logarítmica Media": ("headcount", "mld"),
    "Ingreso promedio vs. PIB per cápita": ("mean", "reporting_gdp"),
    "Índice de Gini vs. Ingreso promedio": ("gini", "mean")
}

explicaciones_parejas = {
    "Tasa de pobreza vs. Ingreso promedio": "Esta gráfica muestra la relación entre la tasa de pobreza y el ingreso promedio. Generalmente, a mayor ingreso promedio, menor es la tasa de pobreza.",
    "Tasa de pobreza vs. Desviación Logarítmica Media": "Aquí se observa la relación entre la tasa de pobreza y la desigualdad de ingresos (medida por la Desviación Logarítmica Media). Usualmente, una mayor desigualdad se asocia con mayores tasas de pobreza.",
    "Ingreso promedio vs. PIB per cápita": "Esta gráfica explora la relación entre el ingreso promedio de los hogares y el Producto Interno Bruto (PIB) per cápita del país, indicadores clave del desarrollo económico.",
    "Índice de Gini vs. Ingreso promedio": "Muestra cómo el índice de Gini (medida de desigualdad) se relaciona con el ingreso promedio. Puede haber países con alto ingreso promedio pero también alta desigualdad."
}


# --- Funciones de graficado ---
def graficar_region(region, indicador):
    # Aquí se mantiene el filtro por 'national' ya que es para evolución de región
    df_filtered = df[(df['region_name'] == region) & (df['reporting_level'] == 'national')]
    if df_filtered.empty:
        st.warning(f"No hay datos para la región '{region}' y el indicador '{traducciones.get(indicador, indicador)}' a nivel nacional.")
        return None

    fig = px.line(df_filtered,
                  x='reporting_year',
                  y=indicador,
                  color='country_name',
                  title=f'{traducciones.get(indicador, indicador)} en {region} por país',
                  labels={'reporting_year': 'Año', indicador: traducciones.get(indicador, indicador), 'country_name': 'País'})
    fig.update_layout(hovermode="x unified")
    return fig

def evolucion_pais(pais, indicador):
    df_pais = df[df['country_name'] == pais].copy()
    if df_pais.empty:
        st.warning(f"No hay datos para el país '{pais}'.")
        return None, None

    # Filtrar por el reporting_level más frecuente para ese país
    # Nota: si el reporting_level más frecuente varía mucho por año para un mismo país, esto simplifica.
    # Se toma el reporting_level globalmente más frecuente para ese país en todo el dataset.
    
    # Primero, asegúrate de que 'reporting_level' esté presente y no sea nulo antes de calcular la moda
    if 'reporting_level' in df_pais.columns and not df_pais['reporting_level'].dropna().empty:
        most_freq_level_for_country = df_pais['reporting_level'].mode()[0]
        df_pais_filtered = df_pais[df_pais['reporting_level'] == most_freq_level_for_country]
    else:
        # Si no hay 'reporting_level' o es todo nulo, intenta usar 'national' si existe o simplemente el df_pais
        if 'reporting_level' in df_pais.columns and 'national' in df_pais['reporting_level'].unique():
            df_pais_filtered = df_pais[df_pais['reporting_level'] == 'national']
        else:
            df_pais_filtered = df_pais # Si no hay 'reporting_level' o un 'national', usa todos los datos del país

    if df_pais_filtered.empty:
        st.warning(f"No hay datos suficientes para el país '{pais}' con el nivel de reporte más frecuente.")
        return None, None

    fig = px.line(df_pais_filtered,
                  x='reporting_year',
                  y=indicador,
                  title=f'Evolución de {traducciones.get(indicador, indicador)} en {pais}',
                  labels={'reporting_year': 'Año', indicador: traducciones.get(indicador, indicador)})
    fig.update_layout(hovermode="x unified")
    return fig, df_pais_filtered[['reporting_year', indicador]].set_index('reporting_year')

def graficar_relacion_variables_seleccion(x_var, y_var):
    # Aquí se usa 'national' como un nivel de reporte base para la relación entre variables
    df_plot = df[df['reporting_level'] == 'national'].dropna(subset=[x_var, y_var])
    fig = px.scatter(df_plot,
                     x=x_var,
                     y=y_var,
                     color='region_name',
                     hover_name='country_name',
                     title=f'Relación entre {traducciones.get(x_var, x_var)} y {traducciones.get(y_var, y_var)}',
                     labels={x_var: traducciones.get(x_var, x_var), y_var: traducciones.get(y_var, y_var)})
    fig.update_layout(hovermode="closest")
    return fig

def graficar_comparativa_anual(df_data, variable_seleccionada, titulo):
    fig = px.scatter(df_data,
                     x='country_name',
                     y=variable_seleccionada,
                     color='region_name',
                     hover_name='country_name',
                     title=titulo,
                     labels={'country_name': 'País', variable_seleccionada: traducciones.get(variable_seleccionada, variable_seleccionada), 'region_name': 'Región'})
    fig.update_layout(xaxis_tickangle=-45) # Inclinar las etiquetas del eje x para mejor legibilidad
    return fig


# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Pobreza e Ingresos")
st.title("Análisis de Pobreza e Ingresos Globales")

# Definir las pestañas, insertando la nueva en la posición correcta
tabs = st.tabs(["Información General", "Evolución por País", "Comparativa por Año y Variable", "Relación entre variables", "Mapa mundial por indicador"])

with tabs[0]:
    st.header("Información General del Dataset")
    st.write("Este dashboard presenta datos de pobreza e ingresos a nivel nacional y subnacional.")
    st.write(f"Número total de registros: {len(df)}")
    st.write(f"Columnas disponibles: {', '.join(df.columns.tolist())}")

    st.subheader("Primeras filas del dataset (con reporting_level)")
    # Muestra el df completo para que se vea el reporting_level
    st.dataframe(df.head())

with tabs[1]:
    st.subheader("Evolución de Indicadores por País")
    pais_seleccionado = st.selectbox("Selecciona un País", paises)
    indicador_seleccionado_pais = st.selectbox("Selecciona un Indicador", variables_traducidas, key='pais_indicador')
    indicador_seleccionado_pais_key = traducciones_inv[indicador_seleccionado_pais]

    fig, tabla = evolucion_pais(pais_seleccionado, indicador_seleccionado_pais_key)
    if fig:
        st.plotly_chart(fig, use_container_width=True) # Usar plotly_chart para interactividad
        if tabla is not None:
            st.dataframe(tabla)

with tabs[2]: # ¡Esta es la nueva pestaña!
    st.subheader("Comparativa de Variables por Año")
    año_comparativa = st.selectbox("Selecciona un Año", años, key='año_comparativa_selector')
    variable_comparativa = st.selectbox("Selecciona una Variable", variables_traducidas, key='var_comparativa_selector')
    variable_comparativa_key = traducciones_inv[variable_comparativa]

    if año_comparativa != "Todos los años":
        df_anual = df[df['reporting_year'] == int(año_comparativa)].copy()

        # Lógica para seleccionar el valor por país basado en el reporting_level más frecuente
        df_filtrado_final = pd.DataFrame()
        for country in df_anual['country_name'].unique():
            df_country_year = df_anual[df_anual['country_name'] == country].copy()
            if not df_country_year.empty:
                # Calcular la frecuencia de cada reporting_level para este país en este año
                reporting_level_counts = df_country_year['reporting_level'].value_counts()
                if not reporting_level_counts.empty:
                    most_freq_level = reporting_level_counts.idxmax() # Obtiene el nivel más frecuente
                    
                    # Filtra por el nivel más frecuente
                    df_seleccionado = df_country_year[df_country_year['reporting_level'] == most_freq_level]
                    
                    # Si hay múltiples entradas para el mismo país/año/reporting_level (duplicados de datos),
                    # tomamos el promedio o una fila representativa. Aquí tomamos el promedio para la variable.
                    # Esto es importante para asegurar un único punto por país en el scatter plot.
                    if not df_seleccionado.empty:
                        # Si la variable es numérica, promediar. Si no, tomar la primera.
                        # Para este caso, asumimos que 'variable_comparativa_key' es numérica.
                        # Si no es numérica, se debería definir cómo se agrega.
                        row_to_add = {
                            'country_name': country,
                            'region_name': df_seleccionado['region_name'].iloc[0], # Tomar la primera región
                            variable_comparativa_key: df_seleccionado[variable_comparativa_key].mean() # Promediar el valor
                        }
                        # Añadir otras columnas necesarias si se usan en el gráfico (ej. reporting_year, que ya está filtrado)
                        df_filtrado_final = pd.concat([df_filtrado_final, pd.DataFrame([row_to_add])], ignore_index=True)
                else:
                    # Si no hay reporting_level para este país en este año, se podría optar por omitirlo o tomar otra decisión
                    pass # Actualmente se omite

        if not df_filtrado_final.empty:
            st.write(f"Mostrando datos para el año **{año_comparativa}** y la variable **{variable_comparativa}**.")
            st.write("Solo se considera el valor correspondiente al 'reporting_level' más frecuente por país para ese año. Si hay múltiples valores para ese nivel, se promedian.")
            fig_comparativa = graficar_comparativa_anual(df_filtrado_final, variable_comparativa_key,
                                                         f'{variable_comparativa} por País en el año {año_comparativa}')
            st.plotly_chart(fig_comparativa, use_container_width=True)
        else:
            st.warning(f"No hay datos para mostrar para el año {año_comparativa} y la variable {variable_comparativa} con la lógica de nivel de reporte.")
    else:
        st.info("Por favor, selecciona un año específico para ver la comparativa anual.")


with tabs[3]: # Las pestañas originales se han desplazado
    st.subheader("Relación entre variables")
    sub_tabs = st.tabs(["Gráfica de dispersión", "Matriz de correlación"])

    with sub_tabs[0]:
        relacion_trad = st.selectbox("Relación", list(etiquetas_parejas.keys()))
        x_var, y_var = etiquetas_parejas[relacion_trad]
        st.plotly_chart(graficar_relacion_variables_seleccion(x_var, y_var), use_container_width=True)
        st.markdown(f"**Explicación:** {explicaciones_parejas.get(relacion_trad, '')}")

    with sub_tabs[1]:
        st.markdown("### Matriz de correlación")
        # Asegurarse de que las variables para la matriz de correlación sean solo las numéricas relevantes
        # Aquí se usa 'variables' que ya filtra por las que tienen traducción. Se filtra por reporting_level = 'national'
        corr_vars_df = df[df['reporting_level'] == 'national'][variables].dropna()
        
        if not corr_vars_df.empty:
            corr_matrix = corr_vars_df.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6)) # Aumentar tamaño para mejor legibilidad
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                        xticklabels=[traducciones[v] for v in corr_matrix.columns],
                        yticklabels=[traducciones[v] for v in corr_matrix.index],
                        ax=ax_corr, cbar_kws={'label': 'Coeficiente de Correlación'})
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.title("Matriz de Correlación de Variables Nacionales")
            st.pyplot(fig_corr)
        else:
            st.warning("No hay suficientes datos a nivel nacional para generar la matriz de correlación.")


with tabs[4]: # Y esta también
    st.subheader("Mapa mundial por Indicador")
    año_mapa = st.selectbox("Selecciona un Año", años, key='mapa_año')
    indicador_mapa = st.selectbox("Selecciona un Indicador", variables_traducidas, key='mapa_indicador')
    indicador_mapa_key = traducciones_inv[indicador_mapa]

    df_mapa = df[df['reporting_level'] == 'national'].copy() # Solo usar datos nacionales para el mapa por simplicidad
    if año_mapa != "Todos los años":
        df_mapa = df_mapa[df_mapa['reporting_year'] == int(año_mapa)]

    if df_mapa.empty:
        st.warning(f"No hay datos para mostrar el mapa con el año '{año_mapa}' y el indicador '{indicador_mapa}'.")
    else:
        # Intenta mapear nombres de países a códigos ISO 3 para Plotly
        df_mapa['iso_alpha'] = df_mapa['country_name'].apply(lambda x: pycountry.countries.get(name=x).alpha_3 if pycountry.countries.get(name=x) else None)
        df_mapa_filtered = df_mapa.dropna(subset=['iso_alpha', indicador_mapa_key]) # Usar indicador_mapa_key

        if not df_mapa_filtered.empty:
            fig_mapa = px.choropleth(df_mapa_filtered,
                                     locations="iso_alpha",
                                     color=indicador_mapa_key,
                                     hover_name="country_name",
                                     projection="natural earth",
                                     title=f'{indicador_mapa} Mundial en {año_mapa}',
                                     color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig_mapa, use_container_width=True)
        else:
            st.warning(f"No hay datos suficientes con códigos de país válidos para generar el mapa para el año '{año_mapa}' y el indicador '{indicador_mapa}'.")
