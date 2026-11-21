import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Simulador de Bombeo",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# ESTILO VISUAL - DASHBOARD INDUSTRIAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07111f; --panel:#101e30; --panel2:#14263b; --line:#243a52; --cyan:#22d3ee; --blue:#3b82f6; --muted:#91a4b9; --white:#edf6ff; }
.stApp { background: radial-gradient(circle at 15% 0%, #102943 0%, #07111f 42%, #050c16 100%); color:var(--white); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(5,12,22,.78); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1b2c,#091321); border-right:1px solid #20364d; }
[data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif; }
.block-container { max-width:1450px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.4px; color:#f1f7ff !important; }
h2 { font-size:1.35rem !important; }
p, label, .stCaption { color:#b6c7d9; }
[data-testid="stMetric"] { background:transparent; }
.stButton>button { border-radius:10px; border:1px solid #31506d; background:#142b42; color:#eaf7ff; font-weight:700; min-height:2.7rem; transition:.2s ease; }
.stButton>button:hover { border-color:#22d3ee; color:#22d3ee; background:#17344d; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#22d3ee; }
hr { border-color:#263c53 !important; }
[data-testid="stExpander"] { background:#101e30; border:1px solid #243a52; border-radius:14px; }
[data-testid="stAlert"] { border-radius:12px; }
.metric-card { background:linear-gradient(145deg,rgba(20,38,59,.98),rgba(12,27,44,.98)); border:1px solid #29445f; border-radius:16px; padding:18px 18px 16px; min-height:126px; box-shadow:0 10px 28px rgba(0,0,0,.16); position:relative; overflow:hidden; }
.metric-card:after { content:''; position:absolute; width:90px; height:90px; right:-35px; top:-35px; border-radius:50%; background:rgba(34,211,238,.08); }
.metric-label { color:#9eb3c8; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.7px; }
.metric-value { color:#f3f9ff; font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; margin-top:9px; line-height:1.2; }
.metric-unit { color:#22d3ee; font-size:.82rem; font-weight:600; margin-top:5px; }
.hero { background:linear-gradient(115deg,rgba(16,43,66,.98),rgba(12,27,45,.96)); border:1px solid #2b4d68; border-radius:20px; padding:25px 28px; margin-bottom:20px; box-shadow:0 18px 40px rgba(0,0,0,.2); }
.hero-kicker { color:#22d3ee; font-size:.75rem; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; }
.hero-title { color:#f4f9ff; font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin:7px 0 5px; }
.hero-sub { color:#a9bfd3; font-size:.96rem; }
.status-pill { display:inline-block; padding:7px 12px; border-radius:999px; font-size:.78rem; font-weight:800; letter-spacing:.5px; border:1px solid; }
.status-on { color:#5eead4; background:rgba(13,148,136,.13); border-color:#0f766e; }
.status-off { color:#fbbf24; background:rgba(217,119,6,.12); border-color:#92400e; }
.section-note { color:#8fa7bd; font-size:.9rem; margin-top:-8px; margin-bottom:14px; }
.panel-heading { color:#e9f5ff; font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; margin:8px 0 4px; }
.small-tag { color:#8ca5bc; font-size:.78rem; letter-spacing:.5px; text-transform:uppercase; font-weight:700; }
footer {visibility:hidden;}
@media (max-width: 700px) { .hero-title {font-size:1.5rem;} .metric-value {font-size:1.4rem;} .block-container {padding-left:1rem;padding-right:1rem;} }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. PARÁMETROS FÍSICOS DEL PROYECTO
# ============================================================

RHO = 1000.0
G = 9.81
H = 30.0
EFICIENCIA = 0.70

TAU = 1800.0
ZETA = 0.5

OMEGA = 2 * np.pi / 86400

SEGUNDOS_DIA = 86400
SEGUNDOS_HORA = 3600

VELOCIDAD_SIMULACION = 3.0

# ============================================================
# 3. PARÁMETROS DE LA SOLUCIÓN DIFERENCIAL
# ============================================================

P_MEDIA = (
    RHO * G * H * (6 / SEGUNDOS_HORA)
) / EFICIENCIA

P_AMPLITUD = (
    RHO * G * H * (3 / SEGUNDOS_HORA)
) / EFICIENCIA

D = 1 - TAU**2 * OMEGA**2
E = 2 * ZETA * TAU * OMEGA

B = P_AMPLITUD * D / (D**2 + E**2)
C = -P_AMPLITUD * E / (D**2 + E**2)

ALPHA = -ZETA / TAU
BETA = np.sqrt(1 - ZETA**2) / TAU

# Condiciones iniciales asumidas
P0 = 0.0
V0 = 0.0


# ============================================================
# 4. FUNCIONES MATEMÁTICAS
# ============================================================

def caudal(t):
    return 6 + 3 * np.sin(OMEGA * np.asarray(t))


def potencia_requerida(t):

    q = caudal(t) / SEGUNDOS_HORA

    return (RHO * G * H * q) / EFICIENCIA


def potencia_particular(t):

    t = np.asarray(t)

    return (
        P_MEDIA
        + B * np.sin(OMEGA * t)
        + C * np.cos(OMEGA * t)
    )


def derivada_particular(t):

    t = np.asarray(t)

    return (
        B * OMEGA * np.cos(OMEGA * t)
        - C * OMEGA * np.sin(OMEGA * t)
    )


# Cálculo de constantes de la solución homogénea

C1 = P0 - float(potencia_particular(0))

C2 = (
    V0
    - float(derivada_particular(0))
    - ALPHA * C1
) / BETA


def potencia_homogenea(t):

    t = np.asarray(t)

    return np.exp(ALPHA * t) * (
        C1 * np.cos(BETA * t)
        + C2 * np.sin(BETA * t)
    )


def potencia_dinamica(t):

    return (
        potencia_homogenea(t)
        + potencia_particular(t)
    )


# ============================================================
# 5. ESTADO DE LA SIMULACIÓN
# ============================================================

if "tiempo_total" not in st.session_state:
    st.session_state.tiempo_total = 0.0

if "simulacion_activa" not in st.session_state:
    st.session_state.simulacion_activa = False

if "ultimo_tick" not in st.session_state:
    st.session_state.ultimo_tick = None

if "modo" not in st.session_state:
    st.session_state.modo = "Manual"

if "hora_manual" not in st.session_state:
    st.session_state.hora_manual = 12


# ============================================================
# 6. PANEL DE CONTROL
# ============================================================

st.sidebar.markdown("<div style='padding:10px 0 16px'><div style='font-size:1.5rem;font-weight:800;color:#eaf7ff'>💧 Control central</div><div style='font-size:.78rem;color:#8fa7bd;letter-spacing:1px;text-transform:uppercase'>Sistema de bombeo</div></div>", unsafe_allow_html=True)

modo = st.sidebar.radio(
    "Modo de funcionamiento",
    ["Manual", "Automático"],
    key="selector_modo"
)

if modo != st.session_state.modo:

    st.session_state.modo = modo
    st.session_state.simulacion_activa = False
    st.session_state.ultimo_tick = None

st.sidebar.divider()

if modo == "Manual":

    st.sidebar.subheader("Control manual")

    hora_manual = st.sidebar.slider(
        "Seleccionar hora del día",
        min_value=0,
        max_value=24,
        value=int(st.session_state.hora_manual),
        step=1,
        key="control_hora"
    )

    st.session_state.hora_manual = hora_manual

    tiempo_actual = hora_manual * SEGUNDOS_HORA

    st.sidebar.info(
        "Selecciona una hora para observar "
        "cómo cambian el caudal y la potencia."
    )

else:

    st.sidebar.subheader("Control automático")

    st.sidebar.caption(
        "Cada 3 segundos reales avanza una hora simulada."
    )

    col1, col2 = st.sidebar.columns(2)

    with col1:

        if st.button("▶ Iniciar", use_container_width=True):

            if not st.session_state.simulacion_activa:

                st.session_state.simulacion_activa = True
                st.session_state.ultimo_tick = time.monotonic()

    with col2:

        if st.button("⏸ Pausar", use_container_width=True):

            st.session_state.simulacion_activa = False
            st.session_state.ultimo_tick = None

    if st.sidebar.button(
        "↻ Reiniciar ciclo",
        use_container_width=True
    ):

        st.session_state.tiempo_total = 0.0
        st.session_state.simulacion_activa = False
        st.session_state.ultimo_tick = None

    if st.session_state.simulacion_activa:

        st.sidebar.success("Simulación en funcionamiento")

    else:

        st.sidebar.warning("Simulación pausada")

    st.sidebar.divider()

    tiempo_actual = st.session_state.tiempo_total


# ============================================================
# 7. REPRESENTACIÓN VISUAL DEL SISTEMA
# ============================================================

def punto_en_recorrido(fraccion):

    puntos = np.array([
        [3.0, 1.2],
        [4.2, 1.2],
        [5.2, 1.2],
        [6.25, 2.0],
        [7.0, 2.0],
        [7.0, 3.7],
        [10.2, 3.7],
        [10.2, 4.4]
    ])

    distancias = np.sqrt(
        np.sum(np.diff(puntos, axis=0)**2, axis=1)
    )

    acumuladas = np.concatenate(([0], np.cumsum(distancias)))

    distancia_total = acumuladas[-1]

    distancia = (fraccion % 1) * distancia_total

    indice = np.searchsorted(
        acumuladas,
        distancia,
        side="right"
    ) - 1

    indice = max(0, min(indice, len(distancias) - 1))

    proporcion = (
        distancia - acumuladas[indice]
    ) / distancias[indice]

    posicion = (
        puntos[indice]
        + proporcion * (puntos[indice + 1] - puntos[indice])
    )

    return posicion[0], posicion[1]


def volumen_transferido(t):
    """Volumen acumulado transferido desde las 00:00 (m³).

    Integra Q(t) = 6 + 3 sen(OMEGA*t), expresado en m³/h,
    respecto al tiempo en segundos.
    """
    t = np.asarray(t, dtype=float)
    return (
        6.0 * (t / SEGUNDOS_HORA)
        + (3.0 / (OMEGA * SEGUNDOS_HORA))
        * (1.0 - np.cos(OMEGA * t))
    )


def niveles_tanques(tiempo_ciclo):
    """Niveles visuales derivados del balance de volumen.

    Supuestos ilustrativos explícitos:
    - Área transversal de cada tanque: 30 m².
    - Nivel inicial inferior: 5 m.
    - Nivel inicial elevado: 1 m.
    - Sin consumo ni descarga durante el ciclo.

    El ciclo visual vuelve a sus niveles iniciales al completar 24 h,
    representando un nuevo ciclo con los depósitos restablecidos.
    """
    area_tanque = 30.0  # m², supuesto para visualizar niveles
    nivel_inferior_inicial = 5.0  # m
    nivel_elevado_inicial = 1.0  # m

    volumen = float(volumen_transferido(tiempo_ciclo))
    cambio_nivel = volumen / area_tanque

    nivel_inferior = nivel_inferior_inicial - cambio_nivel
    nivel_elevado = nivel_elevado_inicial + cambio_nivel

    return (
        float(np.clip(nivel_inferior, 0.0, 6.0)),
        float(np.clip(nivel_elevado, 0.0, 6.0)),
        volumen
    )


def crear_esquema(caudal_actual, flujo_activo, tiempo_ciclo, hora_entera):

    fig = go.Figure()

    nivel_inferior, nivel_elevado, volumen_transferido_actual = niveles_tanques(tiempo_ciclo)

    # Límites geométricos de los tanques en el diagrama
    inferior_y_base, inferior_y_tope = 0.56, 2.44
    elevado_y_base, elevado_y_tope = 4.47, 6.43

    # Convertir niveles físicos (0 a 6 m) a coordenadas del dibujo
    inferior_y_agua = inferior_y_base + (nivel_inferior / 6.0) * (inferior_y_tope - inferior_y_base)
    elevado_y_agua = elevado_y_base + (nivel_elevado / 6.0) * (elevado_y_tope - elevado_y_base)

    # TANQUE INFERIOR: estructura y agua variable
    fig.add_shape(
        type="rect", x0=0.7, y0=0.5, x1=3.0, y1=2.5,
        line=dict(color="#94A3B8", width=4),
        fillcolor="rgba(0,0,0,0)"
    )
    fig.add_shape(
        type="rect", x0=0.76, y0=inferior_y_base,
        x1=2.94, y1=inferior_y_agua,
        line=dict(width=0), fillcolor="#1687D9"
    )
    fig.add_shape(
        type="line", x0=0.79, y0=inferior_y_agua,
        x1=2.91, y1=inferior_y_agua,
        line=dict(color="#67DDF5", width=2)
    )
    fig.add_annotation(
        x=1.85, y=max(0.75, inferior_y_base + (inferior_y_agua-inferior_y_base)*0.48),
        text="<b>TANQUE INFERIOR</b>", showarrow=False,
        font=dict(color="white", size=11)
    )

    # TANQUE ELEVADO: el nivel aumenta con el volumen recibido
    fig.add_shape(
        type="rect", x0=9.1, y0=4.4, x1=11.3, y1=6.5,
        line=dict(color="#94A3B8", width=4),
        fillcolor="rgba(0,0,0,0)"
    )
    fig.add_shape(
        type="rect", x0=9.17, y0=elevado_y_base,
        x1=11.23, y1=elevado_y_agua,
        line=dict(width=0), fillcolor="#1687D9"
    )
    fig.add_shape(
        type="line", x0=9.20, y0=elevado_y_agua,
        x1=11.20, y1=elevado_y_agua,
        line=dict(color="#67DDF5", width=2)
    )
    fig.add_annotation(
        x=10.2, y=6.05, text="<b>TANQUE ELEVADO</b>",
        showarrow=False, font=dict(color="white", size=11)
    )

    # ESTRUCTURA DEL TANQUE ELEVADO
    for x in [9.3, 11.1]:
        fig.add_shape(
            type="line", x0=x, y0=1.8, x1=x, y1=4.4,
            line=dict(color="#94A3B8", width=7)
        )
    fig.add_shape(
        type="line", x0=9.3, y0=1.8, x1=11.1, y1=1.8,
        line=dict(color="#94A3B8", width=7)
    )

    # TUBERÍAS
    segmentos = [
        ((3.0, 1.2), (4.2, 1.2)),
        ((4.2, 1.2), (5.2, 1.2)),
        ((6.25, 2.0), (7.0, 2.0)),
        ((7.0, 2.0), (7.0, 3.7)),
        ((7.0, 3.7), (10.2, 3.7)),
        ((10.2, 3.7), (10.2, 4.4))
    ]
    for inicio, fin in segmentos:
        fig.add_shape(
            type="line", x0=inicio[0], y0=inicio[1],
            x1=fin[0], y1=fin[1],
            line=dict(color="#1687D9", width=12)
        )
        fig.add_shape(
            type="line", x0=inicio[0], y0=inicio[1] + 0.035,
            x1=fin[0], y1=fin[1] + 0.035,
            line=dict(color="#7DEBFF", width=4)
        )

    # MOTOR
    fig.add_shape(
        type="rect", x0=4.1, y0=1.55, x1=5.1, y1=2.4,
        line=dict(color="#64748B", width=3), fillcolor="#334155"
    )
    fig.add_annotation(
        x=4.6, y=1.97, text="<b>MOTOR</b>", showarrow=False,
        font=dict(color="white", size=10)
    )

    # BOMBA
    fig.add_shape(
        type="circle", x0=5.2, y0=1.45, x1=6.3, y1=2.55,
        line=dict(color="#93C5FD", width=4), fillcolor="#1D4ED8"
    )
    fig.add_annotation(
        x=5.75, y=2.0, text="<b>B</b>", showarrow=False,
        font=dict(color="white", size=20)
    )

    # La velocidad de las partículas es proporcional al caudal instantáneo.
    # Q varía entre 3 y 9 m³/h; el factor queda entre 0.5 y 1.5.
    factor_velocidad = max(0.0, caudal_actual / 6.0)
    reloj_animacion = time.monotonic()

    # PARTÍCULAS: se desplazan más rápido cuando aumenta el caudal.
    if flujo_activo:
        velocidad_particulas = 0.10 * factor_velocidad
        posiciones = [
            punto_en_recorrido(
                (reloj_animacion * velocidad_particulas + desplazamiento) % 1
            )
            for desplazamiento in [0, 0.16, 0.32, 0.48, 0.64, 0.80]
        ]
    else:
        posiciones = [
            punto_en_recorrido(v) for v in [0.12, 0.38, 0.64, 0.88]
        ]

    fig.add_trace(go.Scatter(
        x=[p[0] for p in posiciones],
        y=[p[1] for p in posiciones],
        mode="markers",
        marker=dict(size=10, color="#C8FAFF", line=dict(color="#00D9FF", width=2)),
        showlegend=False, hoverinfo="skip"
    ))

    # INDICADORES DISCRETOS DE DIRECCIÓN EN TODO EL RECORRIDO DE LAS TUBERÍAS.
    # Pequeños triángulos, alineados con cada tramo para no tapar el esquema.
    # El flujo va desde el tanque inferior, atraviesa la bomba y sube al tanque elevado.
    fig.add_trace(go.Scatter(
        x=[3.65, 4.85, 6.72, 7.0, 7.0, 8.0, 9.15, 10.2],
        y=[1.2, 1.2, 2.0, 2.65, 3.35, 3.7, 3.7, 4.12],
        mode="markers",
        marker=dict(
            symbol=["triangle-right", "triangle-right", "triangle-right",
                    "triangle-up", "triangle-up", "triangle-right",
                    "triangle-right", "triangle-up"],
            size=10,
            color="#B8F7FF",
            line=dict(color="#00BFEA", width=1)
        ),
        showlegend=False,
        hoverinfo="skip"
    ))

    # ETIQUETAS Y LECTURAS DEL DIAGRAMA
    fig.add_annotation(
        x=7.5, y=4.08, text="Tubería de impulsión",
        showarrow=False, font=dict(color="#60A5FA", size=11)
    )
    fig.add_annotation(
        x=5.7, y=0.9, text="<b>BOMBA Y MOTOR ELÉCTRICO</b>",
        showarrow=False, font=dict(color="#60A5FA", size=11)
    )
    fig.add_annotation(
        x=5.65, y=3.0,
        text=f"<b>Caudal: {caudal_actual:.2f} m³/h</b>",
        showarrow=False, font=dict(color="#FACC15", size=14)
    )
    fig.add_annotation(
        x=5.65, y=3.38,
        text=f"<b>HORA: {hora_entera:02d}:00</b>",
        showarrow=False,
        font=dict(color="#E2E8F0", size=12),
        bgcolor="rgba(30, 64, 100, 0.85)",
        bordercolor="#3B82F6",
        borderwidth=1,
        borderpad=5
    )
    fig.add_annotation(
        x=1.85, y=2.68,
        text=f"Nivel: {nivel_inferior:.2f} m",
        showarrow=False, font=dict(color="#93C5FD", size=10)
    )
    fig.add_annotation(
        x=10.2, y=6.68,
        text=f"Nivel: {nivel_elevado:.2f} m",
        showarrow=False, font=dict(color="#93C5FD", size=10)
    )

    fig.update_layout(
        template="plotly_dark", height=540, showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(visible=False, range=[0, 12], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 7], fixedrange=True),
        plot_bgcolor="#0b1928", paper_bgcolor="#0b1928",
        dragmode=False,
        transition=dict(duration=0)
    )

    return fig


# ============================================================
# 8. SIMULACIÓN PRINCIPAL
# ============================================================

def flujo_activo_estado(modo_actual):
    return modo_actual == "Automático" and st.session_state.simulacion_activa


def mostrar_simulador():

    # ACTUALIZAR RELOJ AUTOMÁTICO

    if modo == "Automático" and st.session_state.simulacion_activa:

        ahora = time.monotonic()

        if st.session_state.ultimo_tick is None:

            st.session_state.ultimo_tick = ahora

        transcurrido = (
            ahora - st.session_state.ultimo_tick
        )

        pasos = int(
            transcurrido // VELOCIDAD_SIMULACION
        )

        if pasos > 0:

            st.session_state.tiempo_total += (
                pasos * SEGUNDOS_HORA
            )

            st.session_state.ultimo_tick += (
                pasos * VELOCIDAD_SIMULACION
            )

    # TIEMPO INTERNO Y HORA VISIBLE

    if modo == "Automático":

        tiempo_total = float(
            st.session_state.tiempo_total
        )

    else:

        tiempo_total = float(tiempo_actual)

    tiempo_ciclo = tiempo_total % SEGUNDOS_DIA

    hora_actual = tiempo_ciclo / SEGUNDOS_HORA

    hora_entera = int(hora_actual)

    # VALORES ACTUALES

    caudal_actual = float(
        caudal(tiempo_ciclo)
    )

    potencia_actual = float(
        potencia_requerida(tiempo_ciclo)
    )

    potencia_modelo = float(
        potencia_dinamica(tiempo_total)
    )

    # ========================================================
    # ENCABEZADO
    # ========================================================

    estado_texto = "EN FUNCIONAMIENTO" if flujo_activo_estado(modo) else ("MODO MANUAL" if modo == "Manual" else "EN PAUSA")
    estado_clase = "status-on" if flujo_activo_estado(modo) else "status-off"
    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">Ingeniería · Monitoreo y análisis</div>
      <div class="hero-title">Sistema de bombeo de agua</div>
      <div class="hero-sub">Modelo matemático interactivo para el análisis del caudal y la respuesta de potencia.</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:18px">
        <span class="status-pill {estado_clase}">● {estado_texto}</span>
        <span style="color:#b6c7d9;font-size:.9rem">Modo: <b>{modo}</b></span>
        <span style="color:#b6c7d9;font-size:.9rem">Hora simulada: <b style="color:#22d3ee">{hora_entera:02d}:00</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # INDICADORES
    # ========================================================

    st.markdown('<div class="panel-heading">Indicadores del sistema</div><div class="section-note">Lecturas principales del estado actual de la simulación</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    tarjetas = [
        ("Caudal actual", f"{caudal_actual:.2f}", "m³/h", "💧"),
        ("Potencia requerida", f"{potencia_actual:.2f}", "W · considerando eficiencia", "⚡"),
        ("Potencia dinámica", f"{potencia_modelo:.2f}", "W · respuesta del modelo", "📈"),
        ("Altura de bombeo", f"{H:.0f}", "metros", "↥"),
    ]
    for col, (titulo, valor, unidad, icono) in zip([c1,c2,c3,c4], tarjetas):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{icono} &nbsp;{titulo}</div><div class="metric-value">{valor}</div><div class="metric-unit">{unidad}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right;color:#8fa7bd;font-size:.82rem;margin-top:10px">RELOJ DE SIMULACIÓN &nbsp; <b style="color:#22d3ee;font-size:1rem">{hora_entera:02d}:00</b></div>', unsafe_allow_html=True)
    st.divider()

    # ========================================================
    # REPRESENTACIÓN DEL SISTEMA
    # ========================================================

    st.markdown('<div class="panel-heading">Diagrama del sistema</div><div class="section-note">Recorrido ilustrativo del agua desde el tanque inferior hasta el tanque elevado</div>', unsafe_allow_html=True)

    flujo_activo = (
        modo == "Automático"
        and st.session_state.simulacion_activa
    )

    fig_sistema = crear_esquema(
        caudal_actual,
        flujo_activo,
        tiempo_ciclo,
        hora_entera
    )

    st.plotly_chart(
        fig_sistema,
        use_container_width=True,
        key="esquema_bombeo"
    )

    if flujo_activo:

        st.success(
            "Flujo activo · La simulación está avanzando."
        )

    elif modo == "Automático":

        st.warning(
            "Simulación pausada."
        )

    else:

        st.info(
            "Modo manual: selecciona una hora para "
            "consultar el comportamiento del sistema."
        )

    st.divider()

    # ========================================================
    # GRÁFICA DEL CAUDAL
    # ========================================================

    st.markdown('<div class="panel-heading">Comportamiento del caudal</div><div class="section-note">Variación del caudal a lo largo de un ciclo de 24 horas</div>', unsafe_allow_html=True)

    horas = np.linspace(0, 24, 145)  # Menos puntos: gráficas más ligeras

    tiempos_grafica = (
        horas * SEGUNDOS_HORA
    )

    caudales_grafica = caudal(
        tiempos_grafica
    )

    fig_caudal = go.Figure()

    fig_caudal.add_trace(go.Scatter(
        x=horas,
        y=caudales_grafica,
        mode="lines",
        name="Caudal",
        line=dict(color="#3B82F6", width=3)
    ))

    fig_caudal.add_vline(
        x=hora_actual,
        line_dash="dash",
        line_color="#F87171",
        line_width=2
    )

    fig_caudal.add_trace(go.Scatter(
        x=[hora_actual],
        y=[caudal_actual],
        mode="markers",
        name="Hora actual",
        marker=dict(color="#F87171", size=11)
    ))

    fig_caudal.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,25,41,0.55)",
        font=dict(family="DM Sans, sans-serif", color="#c5d7e8"),
        height=390,
        xaxis_title="Hora del día",
        yaxis_title="Caudal (m³/h)",
        xaxis=dict(
            range=[0, 24],
            dtick=2
        ),
        yaxis=dict(range=[0, 10]),
        margin=dict(l=20, r=20, t=35, b=20),
        transition=dict(duration=0)
    )

    st.plotly_chart(
        fig_caudal,
        use_container_width=True,
        key="grafica_caudal"
    )

    # ========================================================
    # GRÁFICA DE POTENCIA
    # ========================================================

    st.markdown('<div class="panel-heading">Potencia requerida y respuesta dinámica</div><div class="section-note">Comparación entre la potencia calculada y la respuesta del modelo diferencial</div>', unsafe_allow_html=True)

    potencia_req_grafica = potencia_requerida(
        tiempos_grafica
    )

    # Se utiliza el tiempo transcurrido acumulado para la
    # respuesta dinámica, manteniendo la evolución transitoria.

    inicio_ciclo = tiempo_total - tiempo_ciclo

    tiempos_dinamicos = (
        inicio_ciclo + tiempos_grafica
    )

    potencia_din_grafica = potencia_dinamica(
        tiempos_dinamicos
    )

    st.caption(
        "La respuesta dinámica conserva su evolución entre ciclos de 24 horas. "
        "Al repetirse el reloj, no se reinician las condiciones iniciales "
        "del modelo matemático."
    )

    fig_potencia = go.Figure()

    fig_potencia.add_trace(go.Scatter(
        x=horas,
        y=potencia_req_grafica,
        mode="lines",
        name="Potencia requerida",
        line=dict(color="#A78BFA", width=3)
    ))

    fig_potencia.add_trace(go.Scatter(
        x=horas,
        y=potencia_din_grafica,
        mode="lines",
        name="Respuesta dinámica",
        line=dict(color="#22C55E", width=3)
    ))

    fig_potencia.add_vline(
        x=hora_actual,
        line_dash="dash",
        line_color="#F87171",
        line_width=2
    )

    fig_potencia.add_trace(go.Scatter(
        x=[hora_actual],
        y=[potencia_actual],
        mode="markers",
        name="Potencia requerida actual",
        marker=dict(color="#F87171", size=10)
    ))

    fig_potencia.add_trace(go.Scatter(
        x=[hora_actual],
        y=[potencia_modelo],
        mode="markers",
        name="Potencia dinámica actual",
        marker=dict(color="#22C55E", size=10)
    ))

    fig_potencia.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,25,41,0.55)",
        font=dict(family="DM Sans, sans-serif", color="#c5d7e8"),
        height=440,
        xaxis_title="Hora del día",
        yaxis_title="Potencia (W)",
        xaxis=dict(
            range=[0, 24],
            dtick=2
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=20, r=20, t=65, b=20),
        transition=dict(duration=0)
    )

    st.plotly_chart(
        fig_potencia,
        use_container_width=True,
        key="grafica_potencia"
    )

    st.divider()

    # ========================================================
    # INFORMACIÓN MATEMÁTICA
    # ========================================================

    st.markdown('<div class="panel-heading">Parámetros matemáticos</div><div class="section-note">Constantes utilizadas en la solución de la ecuación diferencial</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Coeficiente B",
        f"{B:.2f} W"
    )

    m2.metric(
        "Coeficiente C",
        f"{C:.2f} W"
    )

    m3.metric(
        "Frecuencia angular Ω",
        f"{OMEGA:.8f} rad/s"
    )

    m4.metric(
        "Amortiguamiento ζ",
        f"{ZETA:.2f}"
    )

    with st.expander("Consultar ecuaciones del proyecto"):

        st.markdown(
            """
            **1. Modelo del caudal**

            Q(t) = 6 + 3 sen(Ωt)

            **2. Potencia requerida considerando la eficiencia**

            P_req(t) = ρgHQ(t)/η

            **3. Ecuación diferencial**

            3240000 P''(t) + 1800 P'(t) + P(t) = P_req(t)

            **4. Solución particular**

            P_p(t) = P_media + B sen(Ωt) + C cos(Ωt)

            **5. Solución general**

            P(t) = P_h(t) + P_p(t)
            """
        )

        st.caption(
            f"ρ={RHO:.0f} kg/m³ · g={G:.2f} m/s² · "
            f"H={H:.0f} m · η={EFICIENCIA:.2f} · "
            f"τ={TAU:.0f} s · ζ={ZETA:.2f}"
        )

        st.caption(
            f"Condiciones iniciales asumidas: "
            f"P(0)={P0:.1f} W y P'(0)={V0:.1f} W/s."
        )

        st.caption(
            "Los niveles visuales se calculan integrando el caudal "
            "del modelo y aplicando un área transversal supuesta de "
            "30 m² por tanque, con niveles iniciales de 5 m y 1 m. "
            "Se supone que no hay consumo durante el ciclo y que los "
            "depósitos se restablecen al iniciar un nuevo ciclo de 24 h. "
            "Son supuestos de visualización: las dimensiones reales y "
            "el reinicio físico de los depósitos no forman parte de los "
            "datos medidos del proyecto. La altura H=30 m del modelo "
            "matemático se mantiene constante."
        )

    st.divider()

    st.markdown("<div style='border:1px solid #24516a;background:rgba(8,145,178,.08);padding:13px 16px;border-radius:12px;color:#9eddec;font-size:.9rem'>✓ Modelo matemático cargado correctamente · Parámetros y ecuaciones disponibles en el panel inferior.</div>", unsafe_allow_html=True)

    st.caption(
        "Proyecto académico: Modelación de la potencia "
        "de un sistema de bombeo."
    )


# ============================================================
# 9. ACTUALIZACIÓN AUTOMÁTICA ESTABLE
# ============================================================

# El intervalo permanece fijo para evitar problemas
# al iniciar o pausar la simulación.

@st.fragment(run_every="1s")
def simulacion():

    mostrar_simulador()


simulacion()

