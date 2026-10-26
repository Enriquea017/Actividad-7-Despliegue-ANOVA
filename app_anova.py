# ANOVA de un factor sobre el histórico de ventas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="ANOVA", page_icon="📊", layout="wide")

# Colores en tonos de azul
AZULES = "Blues"
AZUL_OSCURO = "#08306b"
MESES = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}

# Carga de datos
@st.cache_data
def load_data():
    # Lee la hoja de ventas
    df = pd.read_excel("Datos_Complementarios_CLV.xlsx", sheet_name=2)
    # Quita espacios en los nombres de columnas
    df.columns = df.columns.str.strip()
    # Crea la columna de mes
    df['mes'] = pd.to_datetime(df['fecha_documento'], format='%d.%m.%Y').dt.month
    return df

df = load_data()

# Variables disponibles
Lista_num = ['total_neto', 'precio_unitario', 'total_unidades']
Lista_cat = ['marc', 'folio_abreviado', 'vendedor', 'estado', 'ciudad', 'mes']

# Barra lateral
st.sidebar.title("HISTÓRICO DE VENTAS")
st.sidebar.subheader("ANOVA de un factor")
Variable_y = st.sidebar.selectbox(label="Variable dependiente (y)", options=Lista_num)
Variable_x = st.sidebar.selectbox(
    label="Variable independiente (x)",
    options=[c for c in Lista_cat if c != Variable_y])
top_n = st.sidebar.slider("Grupos a comparar (los más frecuentes)", min_value=2, max_value=20, value=8)
alfa = st.sidebar.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

# Cálculos
st.title("📊 ANOVA")
st.markdown(f"**Variable dependiente:** {Variable_y} &nbsp;&nbsp;|&nbsp;&nbsp; **Variable independiente:** {Variable_x}")

datos = df[[Variable_x, Variable_y]].dropna()
datos[Variable_x] = datos[Variable_x].astype(str)
# Conserva los grupos más frecuentes
datos = datos[datos[Variable_x].isin(datos[Variable_x].value_counts().head(top_n).index)]
grupos_dict = {nombre: g[Variable_y].values for nombre, g in datos.groupby(Variable_x)}
# Descarta grupos con menos de 2 observaciones
grupos_dict = {k: v for k, v in grupos_dict.items() if len(v) >= 2}

if len(grupos_dict) < 2:
    st.error("La variable independiente necesita al menos 2 grupos con 2 o más observaciones.")
    st.stop()

grupos = list(grupos_dict.values())
f_valor, p_valor = stats.f_oneway(*grupos)

# Sumas de cuadrados
n_total = sum(len(g) for g in grupos)
k = len(grupos)
media_global = np.concatenate(grupos).mean()
ss_entre = sum(len(g) * (g.mean() - media_global) ** 2 for g in grupos)
ss_dentro = sum(((g - g.mean()) ** 2).sum() for g in grupos)
ss_total = ss_entre + ss_dentro
gl_entre, gl_dentro = k - 1, n_total - k
ms_dentro = ss_dentro / gl_dentro

# Indicadores
M1, M2, M3, M4 = st.columns(4)
M1.metric("🧮 Estadístico F", f"{f_valor:.3f}")
M2.metric("🎲 p-valor", f"{p_valor:.4f}" if p_valor >= 0.0001 else "< 0.0001")
M3.metric("👥 Grupos", k)
M4.metric("🧾 Observaciones", n_total)

if p_valor < alfa:
    st.success("Hay diferencias significativas entre los grupos", icon="✅")
else:
    st.warning("No hay diferencias significativas entre los grupos", icon="⚠️")

# Gráfico de barras y diagrama de cajas
Contenedor_A, Contenedor_B = st.columns(2)

resumen = (datos.groupby(Variable_x)[Variable_y]
           .agg(['count', 'mean', 'std', 'sem', 'min', 'median', 'max']).reset_index())

with Contenedor_A:
    st.subheader("📊 Gráfico de barras: media por grupo")
    fig_bar = go.Figure(go.Bar(
        x=resumen[Variable_x], y=resumen['mean'],
        error_y=dict(type='data', array=resumen['sem'], visible=True),
        marker=dict(color=resumen['mean'], colorscale=AZULES, line=dict(color=AZUL_OSCURO, width=1))))
    fig_bar.add_hline(y=media_global, line_dash="dash", line_color=AZUL_OSCURO,
                      annotation_text="Media global", annotation_position="top left")
    fig_bar.update_layout(height=380, xaxis_title=Variable_x,
                          yaxis_title=f"Media de {Variable_y}",
                          title="Media por grupo con error estándar")
    st.plotly_chart(fig_bar, use_container_width=True)

with Contenedor_B:
    st.subheader("📦 Diagrama de cajas: distribución por grupo")
    orden = sorted(datos[Variable_x].unique())
    fig_box = px.box(datos, x=Variable_x, y=Variable_y, color=Variable_x, points="outliers",
                     category_orders={Variable_x: orden},
                     color_discrete_sequence=px.colors.sample_colorscale(
                         AZULES, [0.35 + 0.65 * i / max(len(orden) - 1, 1) for i in range(len(orden))]))
    fig_box.update_layout(height=380, showlegend=False,
                          title="Dispersión y valores atípicos")
    st.plotly_chart(fig_box, use_container_width=True)

# Tabla ANOVA y descriptivos
Contenedor_C, Contenedor_D = st.columns(2)

with Contenedor_C:
    st.subheader("🧾 Tabla ANOVA")
    tabla_anova = pd.DataFrame({
        'Fuente': ['Entre grupos', 'Dentro de grupos', 'Total'],
        'Suma de cuadrados': [ss_entre, ss_dentro, ss_total],
        'gl': [gl_entre, gl_dentro, n_total - 1],
        'Cuadrado medio': [ss_entre / gl_entre, ms_dentro, np.nan],
        'F': [f_valor, np.nan, np.nan],
        'p-valor': [p_valor, np.nan, np.nan]})
    st.dataframe(tabla_anova.style.format(precision=4, na_rep=""), use_container_width=True, hide_index=True)

with Contenedor_D:
    st.subheader("📋 Estadísticos descriptivos por grupo")
    desc = resumen.rename(columns={Variable_x: 'Grupo', 'count': 'n', 'mean': 'Media', 'std': 'Desv. est.',
                                   'sem': 'Error est.', 'min': 'Mín', 'median': 'Mediana', 'max': 'Máx'})
    st.dataframe(desc.style.format(precision=3), use_container_width=True, hide_index=True)

# Heatmap de la media por grupo y mes
st.subheader("🔥 Heatmap: media por grupo y mes" if Variable_x != 'mes' else "🔥 Heatmap: media por mes y marca")
Variable_col = 'mes' if Variable_x != 'mes' else 'marc'
mapa = df[[Variable_x, Variable_col, Variable_y]].dropna()
mapa[Variable_x] = mapa[Variable_x].astype(str)
mapa = mapa[mapa[Variable_x].isin(datos[Variable_x].unique())]
if Variable_col == 'marc':
    mapa = mapa[mapa[Variable_col].isin(mapa[Variable_col].value_counts().head(top_n).index)]
tabla_mapa = mapa.pivot_table(index=Variable_x, columns=Variable_col, values=Variable_y, aggfunc='mean')
if Variable_col == 'mes':
    tabla_mapa.columns = [MESES.get(m, str(m)) for m in tabla_mapa.columns]

# Escala de color entre los percentiles 5 y 95
valores = tabla_mapa.values[~np.isnan(tabla_mapa.values)]
fig_heat = go.Figure(go.Heatmap(
    z=tabla_mapa.values, x=[str(c) for c in tabla_mapa.columns], y=tabla_mapa.index.astype(str),
    colorscale=AZULES, zmin=np.percentile(valores, 5), zmax=np.percentile(valores, 95),
    text=np.round(tabla_mapa.values, 1), texttemplate="%{text}", hoverongaps=False,
    colorbar=dict(title=f"Media de {Variable_y}")))
fig_heat.update_layout(height=max(320, 40 * len(tabla_mapa) + 120),
                       xaxis_title=Variable_col, yaxis_title=Variable_x,
                       title=f"Media de {Variable_y} por {Variable_x} y {Variable_col}")
st.plotly_chart(fig_heat, use_container_width=True)
