import streamlit as st
import base64
import pickle
import pandas as pd
import numpy as np
import io
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# ==================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==================================================

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Clasificador Pokémon",
    layout="wide",
)

# --- GESTIÓN DE RUTAS (SOLUCIÓN AL ERROR) ---
# Obtenemos la ruta absoluta del directorio donde se encuentra este script (app.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONSTANTES DE ESTILO Y MODELO ---

# Colores de tipo Pokémon para el display de la predicción
POKEMON_TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0", "Grass": "#78C850",
    "Electric": "#F8D030", "Ice": "#98D8D8", "Fighting": "#C03028", "Poison": "#A040A0",
    "Ground": "#E0C068", "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8", "Steel": "#B8B8D0",
    "Dark": "#705848", "Fairy": "#EE99AC", "Unknown": "#68A090"
}

# ==================================================
# CARGA DE MODELO, SCALER Y ENCODER
# ==================================================

@st.cache_resource
def load_artifacts():
    try:
        # Construimos las rutas completas usando os.path.join
        model_path = os.path.join(CURRENT_DIR, "modelo_pokemon_lr.joblib")
        scaler_path = os.path.join(CURRENT_DIR, "scaler_pokemon.joblib")
        encoder_path = os.path.join(CURRENT_DIR, "abilities_encoder.joblib")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        abilities_encoder = joblib.load(encoder_path)

        return model, scaler, abilities_encoder
    except Exception as e:
        st.error(f"Error cargando archivos del modelo. Verifica que los archivos .joblib estén en GitHub en la misma carpeta que app.py. Detalle: {e}")
        return None, None, None

MODEL, SCALER, ABILITIES_ENCODER = load_artifacts()
MLB = ABILITIES_ENCODER


@st.cache_resource
def load_abilities():
    try:
        # Si ABILITIES_ENCODER es None (falló la carga anterior), esto daría error, 
        # así que verificamos antes.
        if ABILITIES_ENCODER is None:
            return ["None"]
        return ABILITIES_ENCODER.classes_

    except Exception as e:
        st.error(f"Error al leer las clases del encoder: {e}")
        return ["None"]

ABILITIES_OPTIONS = load_abilities()


# --- Funciones de Audio y Fondo ---

# FUNCIÓN MEJORADA PARA CARGAR FONDO DE MANERA SEGURA (Base64)
def set_background(image_filename):
    # Construimos la ruta completa a la imagen
    image_path = os.path.join(CURRENT_DIR, image_filename)
    
    try:
        with open(image_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()

        css = f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_img}");
                background-size: cover !important;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            .st-emotion-cache-18ni7ap,
            .st-emotion-cache-1jicfl2,
            .st-emotion-cache-1gulkj5 {{
                background: transparent !important;
            }}

            .main {{
                padding-top: 2rem !important;
            }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"No se encontró la imagen de fondo en: {image_path}. Verifica el nombre y que esté subida a GitHub.")

# IMPORTANTE: Asegúrate de que el nombre del archivo en GitHub sea EXACTAMENTE este.
# Si puedes, renómbralo a 'imagen_pokemon.png' (sin espacios ni tildes) para evitar problemas en Linux.
set_background("Imagen del Pokémon.png")


# ==================================================
# 2. ESTILOS CSS
# ==================================================
st.markdown("""
    <style>
        @import url('https://fonts.cdnfonts.com/css/pokemon-solid');
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

        /* TÍTULO PRINCIPAL */
        .pokemon-title {
            font-family: 'Pokemon Solid', sans-serif;
            color: #ffcb05; /* Amarillo */
            -webkit-text-stroke: 4px #2a75bb; /* Azul */
            font-size: 80px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        /* SUBTÍTULOS Y TEXTO GENERAL */
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: white;
            font-weight: 600;
            text-shadow: 2px 2px 6px #000000;
            font-family: 'Press Start 2P', cursive;
            margin-top: 15px;
        }

        /* SPRITES */
        .sprite {
            width: 150px;
            image-rendering: pixelated;
        }
        .sprite-container {
            display: flex;
            justify-content: center;
            gap: 60px;
            margin-top: 10px;
            margin-bottom: 25px;
        }
        .float {
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
            100% { transform: translateY(0px); }
        }

        /* ESTILO PARA BOTONES DE STREAMLIT (Mejorado) */
        .stButton > button {
            background: #ffcb05 !important;
            color: #2a75bb !important;
            border: 4px solid #2a75bb !important;
            font-family: 'Pokemon Solid', sans-serif !important;
            padding: 14px 20px !important;
            border-radius: 15px !important;
            font-size: 20px !important;
            box-shadow: 4px 4px 0px #000;
            cursor: pointer !important;
            transition: 0.1s;
        }

        .stButton > button:hover {
            background: #ffe66d !important;
            transform: translateY(-2px) scale(1.05);
            box-shadow: 6px 6px 0px #000;
        }

        /* CENTRADO */
        .center {
            display: flex;
            justify-content: center;
        }

        /* MODAL */
        div[data-testid="stModal"] {
            border: 4px solid #2a75bb !important;
            border-radius: 20px !important;
            background: #fffa !important;
            backdrop-filter: blur(6px);
            padding: 20px;
        }
        .modal-title {
            font-family: 'Pokemon Solid', sans-serif;
            color: #2a75bb;
            -webkit-text-stroke: 2px #ffcb05;
            font-size: 40px;
            text-align: center;
        }
        .stNumberInput > label {
            color: #2a75bb !important;
            font-weight: bold;
        }

        /* BADGE DE PREDICCIÓN (Nuevo) */
        .type-badge {
            font-family: 'Pokemon Solid', sans-serif;
            font-size: 32px;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            margin: 20px auto;
            display: inline-block;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            border: 4px solid #fff;
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            text-align: center;
        }

        /* Contenedor de Inputs */
        .input-container {
            border: 3px solid #ffcb05;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# 3. FUNCIONES DEL MODELO Y PREDICCIÓN
# ==================================================
def prepare_input_for_model(data):
    # Verificar si los artefactos cargaron correctamente antes de procesar
    if MODEL is None or SCALER is None or ABILITIES_ENCODER is None:
        return None

    num_cols = ["hp", "atk", "def", "spa", "spd", "speed", "height", "weight"]

    num_values = [[
        data["hp"], data["atk"], data["def"],
        data["spa"], data["spd"], data["speed"],
        data["height"], data["weight"]
    ]]

    scaled_nums = SCALER.transform(num_values)

    df_scaled = pd.DataFrame(
        scaled_nums,
        columns=[f"sc_{c}" for c in num_cols]
    )

    # --- FIX: usar ABILITIES_ENCODER en lugar de MLB ---
    abil_list = [data["abilities"]]
    abil_encoded = ABILITIES_ENCODER.transform([abil_list])
    df_abil = pd.DataFrame(
        abil_encoded,
        columns=[f"abi_{a}" for a in ABILITIES_ENCODER.classes_]
    )

    df_color = pd.get_dummies([data["color"]], prefix="color")
    df_gen = pd.get_dummies([data["generation"]], prefix="gen")

    df_final = pd.concat([df_scaled, df_abil, df_color, df_gen], axis=1)

    missing_cols = [c for c in MODEL.feature_names_in_ if c not in df_final.columns]
    for col in missing_cols:
        df_final[col] = 0

    df_final = df_final[MODEL.feature_names_in_]

    return df_final




def predict_pokemon_type(features_array):
    """Predice con el modelo de regresión logística."""
    if MODEL is None or features_array is None:
        return "Unknown"

    try:
        prediction = MODEL.predict(features_array)
        return prediction[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return "Unknown"

# ==================================================
# 4. UI Y LÓGICA PRINCIPAL
# ==================================================

# TÍTULO
st.markdown("<h1 class='pokemon-title float'>Clasificador de Pokémon</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ingresa las 12 características para predecir el tipo primario.</p>", unsafe_allow_html=True)
st.write("")

# SPRITES
st.markdown("<p class='subtitle'>Sprites animados:</p>", unsafe_allow_html=True)
st.markdown("""
<div class="sprite-container">
    <img class="sprite float" src="https://play.pokemonshowdown.com/sprites/ani/charizard.gif">
    <img class="sprite float" src="https://play.pokemonshowdown.com/sprites/ani/pikachu.gif">
    <img class="sprite float" src="https://play.pokemonshowdown.com/sprites/ani/gengar.gif">
</div>
""", unsafe_allow_html=True)


# --- INICIALIZACIÓN DE ESTADO ---
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "input_data" not in st.session_state:
    # Inicialización de todos los 12 inputs
    st.session_state.input_data = {
        'name': "Pikachu", 'hp': 50, 'atk': 50, 'def': 50, 'spa': 50, 'spd': 50, 'speed': 50,
        'height': 1.0, 'weight': 50.0, 'abilities': "Static", 'color': "Yellow", 'generation': 1
    }

def open_modal():
    st.session_state.show_modal = True

def close_modal():
    st.session_state.show_modal = False

def handle_predict():
    """Maneja el clic del botón Predecir."""
    if MODEL is None:
        st.session_state.prediction_result = ("Error", "No se pudo cargar el modelo (revisa los archivos en GitHub).")
        return

    # 1. Preparar los datos del estado de la sesión
    features_array = prepare_input_for_model(st.session_state.input_data)
    
    # 2. Realizar la predicción
    if features_array is not None:
        predicted_type = predict_pokemon_type(features_array)
        # 3. Almacenar el resultado para mostrarlo
        st.session_state.prediction_result = (predicted_type, "Predicción exitosa.")
    else:
        st.session_state.prediction_result = ("Error", "Error procesando los datos de entrada.")


def handle_restart():
    """Reinicia la aplicación."""
    st.session_state.prediction_result = None
    st.session_state.input_data = {
        'name': "Pikachu", 'hp': 50, 'atk': 50, 'def': 50, 'spa': 50, 'spd': 50, 'speed': 50,
        'height': 1.0, 'weight': 50.0, 'abilities': "Static", 'color': "Yellow", 'generation': 1
    }
    st.session_state.show_modal = False


# --- BOTONES ---
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.button("⚡ Predecir", on_click=handle_predict, use_container_width=True)

with col2:
    if st.button("📂 Cargar Datos", on_click=open_modal, use_container_width=True):
        pass # La función on_click maneja el estado

with col3:
    st.button("🔄 Reiniciar", on_click=handle_restart, use_container_width=True)


# --- CONTENIDO DEL MODAL (Carga de Datos) ---
if st.session_state.show_modal:
    with st.expander("Cargar Datos", expanded=False):

        st.markdown("<h2 class='modal-title'>Características del Pokémon</h2>", unsafe_allow_html=True)
        
        # Contenedor para el formulario dentro del modal
        with st.container(border=True):
            st.markdown("### Estadísticas Base (Valores entre 1-255)")
            col_hp, col_atk, col_def = st.columns(3)
            with col_hp:
                st.session_state.input_data['hp'] = st.number_input("HP", value=st.session_state.input_data['hp'], min_value=1, max_value=255, key="modal_hp")
            with col_atk:
                st.session_state.input_data['atk'] = st.number_input("Ataque", value=st.session_state.input_data['atk'], min_value=1, max_value=255, key="modal_atk")
            with col_def:
                st.session_state.input_data['def'] = st.number_input("Defensa", value=st.session_state.input_data['def'], min_value=1, max_value=255, key="modal_def")
            
            col_spa, col_spd, col_speed = st.columns(3)
            with col_spa:
                st.session_state.input_data['spa'] = st.number_input("Ataque Especial", value=st.session_state.input_data['spa'], min_value=1, max_value=255, key="modal_spa")
            with col_spd:
                st.session_state.input_data['spd'] = st.number_input("Defensa Especial", value=st.session_state.input_data['spd'], min_value=1, max_value=255, key="modal_spd")
            with col_speed:
                st.session_state.input_data['speed'] = st.number_input("Velocidad", value=st.session_state.input_data['speed'], min_value=1, max_value=255, key="modal_speed")

            st.markdown("---")
            st.markdown("Más Características")

            # Fila 1: Nombre y Generación
            col_name, col_gen = st.columns(2)
            with col_name:
                st.session_state.input_data['name'] = st.text_input("Nombre (ej: Pikachu)", value=st.session_state.input_data['name'], key="modal_name")
            with col_gen:
                st.session_state.input_data['generation'] = st.number_input("Generación", value=st.session_state.input_data['generation'], min_value=1, max_value=9, key="modal_gen")

            # Fila 2: Altura y Peso
            col_h, col_w = st.columns(2)
            with col_h:
                st.session_state.input_data['height'] = st.number_input("Altura (m)", value=st.session_state.input_data['height'], min_value=0.1, max_value=20.0, step=0.1, key="modal_h")
            with col_w:
                st.session_state.input_data['weight'] = st.number_input("Peso (kg)", value=st.session_state.input_data['weight'], min_value=0.1, max_value=1000.0, step=0.1, key="modal_w")

            # Fila 3: Habilidad y Color (Categóricas - Asumimos un OHE)
            col_abi, col_color = st.columns(2)
            with col_abi:
                # Opciones de ejemplo, ajusta a tus datos
                # Convertimos a formato capitalizado para que el usuario lo vea bonito
                abilities_display = [str(a).capitalize() for a in ABILITIES_OPTIONS]

                try:
                    current_index = abilities_display.index(st.session_state.input_data['abilities'].capitalize())
                except ValueError:
                    current_index = 0

                selected_ability = st.selectbox(
                    "Habilidad Principal",
                    options=abilities_display,
                    index=current_index,
                    key="modal_abi"
                )

                # Guardar en minúscula, tal como el encoder lo espera
                st.session_state.input_data['abilities'] = selected_ability.lower()


            with col_color:
                # Opciones de ejemplo, ajusta a tus datos
                color_options = ["Yellow", "Blue", "Red", "Green", "White", "Black"]
                try:
                    current_index = color_options.index(st.session_state.input_data['color'])
                except ValueError:
                    current_index = 0
                st.session_state.input_data['color'] = st.selectbox("Color", options=color_options, index=current_index, key="modal_color")

        st.markdown("---")
        st.button("Aceptar y Cerrar", on_click=close_modal, use_container_width=True)


# --- SECCIÓN INFERIOR (OUTPUT) ---
st.write("---")

st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>📌 Características Ingresadas</p>", unsafe_allow_html=True)

# Mostrar un resumen de los datos de entrada
col_in1, col_in2, col_in3 = st.columns(3)
with col_in1:
    st.markdown(f"**Nombre:** {st.session_state.input_data['name']}")
    st.markdown(f"**HP:** {st.session_state.input_data['hp']}")
    st.markdown(f"**Ataque:** {st.session_state.input_data['atk']}")
    st.markdown(f"**Defensa:** {st.session_state.input_data['def']}")
with col_in2:
    st.markdown(f"**Ataque Esp.:** {st.session_state.input_data['spa']}")
    st.markdown(f"**Defensa Esp.:** {st.session_state.input_data['spd']}")
    st.markdown(f"**Velocidad:** {st.session_state.input_data['speed']}")
with col_in3:
    st.markdown(f"**Generación:** {st.session_state.input_data['generation']}")
    st.markdown(f"**Altura (m):** {st.session_state.input_data['height']}")
    st.markdown(f"**Peso (kg):** {st.session_state.input_data['weight']}")
    st.markdown(f"**Habilidad:** {st.session_state.input_data['abilities']}")
    st.markdown(f"**Color:** {st.session_state.input_data['color']}")

st.markdown("</div>", unsafe_allow_html=True)


st.write("---")
st.markdown("<p class='subtitle'>Resultado de la Predicción del Modelo</p>", unsafe_allow_html=True)

# --- DISPLAY DE PREDICCIÓN ---
if st.session_state.prediction_result:
    predicted_type, message = st.session_state.prediction_result
    
    if predicted_type == "Error":
        st.error(message)
    else:
        color = POKEMON_TYPE_COLORS.get(predicted_type, "#68A090") # Default: Unknown/Placeholder
        
        st.success(f"¡El modelo ha hecho su predicción!")
        st.markdown(f"""
            <div style="text-align: center;">
                <p style="font-size: 20px; color: white; text-shadow: 2px 2px 4px #000;">
                    El Tipo Primario Predicho es:
                </p>
                <div class="type-badge" style="background-color: {color};">
                    {predicted_type}
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("Presiona **'Cargar Datos'** para ingresar las características y luego **'Predecir'**.")
