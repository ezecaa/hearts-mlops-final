import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import torch
import joblib
import os
from pathlib import Path
import traceback
import subprocess
import webbrowser

# ==== 1. DEFINICIONES DE FUNCIONES ====

# ---- Modelo de Red Neuronal Feedforward ----
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(Feedforward, self).__init__()
        
        layers = []
        prev_size = input_size 

        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3)
            ])
            prev_size = hidden_size        

        layers.append(torch.nn.Linear(prev_size, 1))
        layers.append(torch.nn.Sigmoid())        

        self.layers = torch.nn.Sequential(*layers)    

    def forward(self, x):
        return self.layers(x)

# ---- Funciones de ML para predicciones ----
@st.cache_resource
def cargar_modelo_ml():
    """Carga el modelo de ML y el preprocesador de manera optimizada"""
    try:
        # Rutas a los archivos del modelo
        model_path = Path('ModeloEntre/stroke_model.pth')
        preprocessor_path = Path('ModeloEntre/stroke_preprocessor.pkl')
        
        # Verificar que existan los archivos
        if not model_path.exists():
            st.error(f"No se encontró el archivo del modelo en {model_path}")
            return None, None
        if not preprocessor_path.exists():
            st.error(f"No se encontró el archivo del preprocesador en {preprocessor_path}")
            return None, None
        
        # Cargar el preprocesador 
        preprocessor = joblib.load(preprocessor_path)
        
        # Configurar dispositivo para PyTorch checkpoint = torch.load(model_path, map_location=device)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            st.info("Usando Apple Metal (MPS) para aceleración de hardware")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            st.info("Usando CUDA GPU para aceleración de hardware")
        #elif torch.cuda.is_available():    # https://github.com/pytorch/pytorch/issues/76039    # https://pypi.org/project/torch-npu/ 
        #    device = torch.device("npu") # El docker tarda mucho al cargar la librería
        #    st.info("Tiene un nuevo CPU con NPU ??? testing ") 
        else:
            device = torch.device("cpu")
            st.info("Usando CPU para cálculos (sin aceleración de hardware)")

        # Cargar el modelo directamente desde el checkpoint
        try:
            # Cargar el checkpoint
            #checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            checkpoint = torch.load(model_path, map_location=device)
            
            # Intenta input_size Extraer parámetros del modelo Intentar cargar desde el checkpoint
            input_size = 16  # Valor por defecto basado en el análisis de errores hardcode
            input_size_from_checkpoint = checkpoint.get('input_size')
            if input_size_from_checkpoint is not None:
                input_size = input_size_from_checkpoint 

            # se indica el parámetro del modelo
            hidden_sizes = checkpoint.get('hidden_sizes', [128, 64, 32])
            
            # Debug-testing Mostrar información sobre el modelo
            st.info(f"Modelo cargado con éxito. Arquitectura: input_size={input_size}, hidden_sizes={hidden_sizes}")
            
            # Crear y cargar el modelo
            model = Feedforward(input_size, hidden_sizes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Poner el modelo en modo evaluación
            
            return model, preprocessor
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.error(traceback.format_exc())
            return None, None
    except Exception as e:
        st.error(f"Error general al cargar recursos: {e}")
        st.error(traceback.format_exc())
        return None, None

def predecir_stroke(paciente, model, preprocessor):
    """Realiza la predicción de accidente cerebrovascular"""
    if model is None or preprocessor is None:
        return None
    
    try:
        # Convertir el paciente (que es una serie de pandas) a un DataFrame
        paciente_df = pd.DataFrame([paciente])
        
        # Seleccionar solo las columnas que el preprocesador espera
        # Esto dependerá de cómo se entrenó el modelo, ajusta según sea necesario
        columnas_modelo = ['gender', 'age', 'hypertension', 'heart_disease', 
                         'ever_married', 'work_type', 'residence_type', 
                         'avg_glucose_level', 'bmi', 'smoking_status']
        
        # Asegurarnos de que tenemos las columnas necesarias
        columnas_disponibles = [col for col in columnas_modelo if col in paciente_df.columns]
        paciente_df = paciente_df[columnas_disponibles]
        
        # Debug-Testing  información complementaria  
        #st.info(f"Columnas disponibles para predicción: {columnas_disponibles}")
        
        # Preprocesar los datos
        X = preprocessor.transform(paciente_df)
        
        # Debug-Testing información complementaria 
        #st.info(f"Forma de los datos preprocesados: {X.shape}")
        
        # Convertir a tensor de PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Realizar la predicción
        with torch.no_grad():
            y_pred = model(X_tensor)
        
        # Convertir a probabilidad
        probabilidad = y_pred.item()
        
        # Devolver la probabilidad y la predicción binaria (1 si prob > 0.5, 0 si no)
        return probabilidad, 1 if probabilidad > 0.4 else 0
    
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ---- Funciones de conexión y consulta básica ----
def conectar_bd():
    try:
        conn = psycopg2.connect(
            host="localhost",
            #host="postgres", # conexión a docker 
            port="5432",
            user="airflow",#postgres
            password="airflow",#postgres
            database="stroke"
        )
        return conn
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

# ---- Funciones para obtener datos y buscar ----
def obtener_datos():
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stroke_data")
            datos = cursor.fetchall()
            
            # Obtener nombres de columnas
            columnas = [desc[0] for desc in cursor.description]
            
            # Cerrar cursor y conexión
            cursor.close()
            conn.close()
            
            # Crear DataFrame de pandas
            df = pd.DataFrame(datos, columns=columnas)
            return df
        except Exception as e:
            st.error(f"Error al consultar datos: {e}")
            return None
    return None

def buscar_por_id(id_paciente):
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stroke_data WHERE id = %s", (id_paciente,))
            datos = cursor.fetchall()
            
            # Obtener nombres de columnas
            columnas = [desc[0] for desc in cursor.description]
            
            # Cerrar cursor y conexión
            cursor.close()
            conn.close()
            
            # Crear DataFrame de pandas
            df = pd.DataFrame(datos, columns=columnas)
            return df
        except Exception as e:
            st.error(f"Error al buscar paciente: {e}")
            return None
    return None

# ---- Funciones para navegación entre registros ----
def obtener_id_minimo():
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(id) FROM stroke_data")
            min_id = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return min_id
        except Exception as e:
            st.error(f"Error al obtener ID mínimo: {e}")
            return None
    return None

def obtener_id_maximo():
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM stroke_data")
            max_id = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return max_id
        except Exception as e:
            st.error(f"Error al obtener ID máximo: {e}")
            return None
    return None

def obtener_siguiente_id(id_actual):
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(id) FROM stroke_data WHERE id > %s", (id_actual,))
            siguiente_id = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return siguiente_id
        except Exception as e:
            st.error(f"Error al obtener siguiente ID: {e}")
            return None
    return None

def obtener_anterior_id(id_actual):
    conn = conectar_bd()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM stroke_data WHERE id < %s", (id_actual,))
            anterior_id = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return anterior_id
        except Exception as e:
            st.error(f"Error al obtener ID anterior: {e}")
            return None
    return None

# ---- Función para mostrar la información del paciente con predicción ML ----
def mostrar_informacion_paciente(paciente, model=None, preprocessor=None):
    # Crear columnas para mostrar la información
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**ID historia clínica:**", paciente['id'])
        st.write("**Género del paciente:**", paciente['gender'])
        st.write("**Edad del paciente:**", paciente['age'])
        st.write("**Ha sufrido hipertensión:**", "Sí" if paciente['hypertension'] == 1 else "No")
        st.write("**Ha sufrido cardiopatía:**", "Sí" if paciente['heart_disease'] == 1 else "No")
        st.write("**Está o estuvo casado:**", paciente['ever_married'])
    
    with col2:
        st.write("**Tipo de trabajo:**", paciente['work_type'])
        st.write("**Tipo de residencia:**", paciente['residence_type'])
        st.write("**Nivel promedio de glucosa:**", paciente['avg_glucose_level'])
        st.write("**Índice de Masa Corporal (IMC):**", paciente['bmi'])
        st.write("**Fuma:**", paciente['smoking_status'])
        st.write("**Sufrió accidente cerebrovascular:**", "Sí" if paciente['stroke'] == 1 else "No")
    
    # Evaluación de riesgo manual por medio de una tabla de riesgo simple 
    st.subheader("Evaluación de Riesgo")
    
    factores_riesgo = 0
    if paciente['hypertension'] == 1:
        factores_riesgo += 1
    if paciente['heart_disease'] == 1:
        factores_riesgo += 1
    if paciente['age'] > 65:
        factores_riesgo += 1
    if paciente['avg_glucose_level'] > 120:
        factores_riesgo += 1
    if paciente['bmi'] > 30:
        factores_riesgo += 1
    if paciente['smoking_status'] == 'smokes':
        factores_riesgo += 1
    
    # Determinar nivel de riesgo
    if factores_riesgo >= 4:
        nivel_riesgo = "Alto"
        color = "red"
    elif factores_riesgo >= 2:
        nivel_riesgo = "Medio"
        color = "orange"
    else:
        nivel_riesgo = "Bajo"
        color = "green"
    
    st.markdown(f"**Nivel de riesgo basado en factores: <span style='color:{color}'>{nivel_riesgo}</span>**", unsafe_allow_html=True)
    st.write(f"Factores de riesgo identificados: {factores_riesgo}/6 (hiper/IMC/edad/....)")
    st.write("---")
    
    # Predicción con el modelo ML
    if model is not None and preprocessor is not None:
        st.subheader("Predicción con Modelo de ML ")
        resultado = predecir_stroke(paciente, model, preprocessor)
        
        if resultado is not None:
            probabilidad, prediccion = resultado
            
            # Mostrar la predicción
            if prediccion == 1:
                pred_texto = "Positivo"
                pred_color = "red"
            else:
                pred_texto = "Negativo"
                pred_color = "green"
            
            # Crear visualización más atractiva
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Predicción: <span style='color:{pred_color}'>{pred_texto}</span>**", unsafe_allow_html=True)
                # Comparar con el valor real
                if 'stroke' in paciente and paciente['stroke'] == prediccion:
                    st.success("✓ La predicción coincide con el valor real")
                elif 'stroke' in paciente:
                    st.warning("⚠ La predicción NO coincide con el valor real")
            
            with col2:
                # Mostrar probabilidad con barra de progreso
                st.write(f"**Probabilidad de accidente cerebrovascular:**")
                st.progress(float(probabilidad))
                st.write(f"{probabilidad:.2%}")
        else:
            st.error("No se pudo realizar la predicción con el modelo de ML")

# ---- Función para ejecutar testing_model.py  adaptada para el -\U0001f3c3---
def ejecutar_testing_model():
    try:
        st.info("Ejecutando testing_model.py... Por favor espere.  ")
        
        # Método 1: Usar subprocess con env UTF-8
        import os
        import subprocess
        
        # Crear un entorno con codificación UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Ejecutar el proceso con redirección de salida a un archivo
        with open("testing_model_output.log", "w", encoding="utf-8") as f:
            result = subprocess.run(
                ["python", "testing_model.py"], 
                env=env,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True
            )
        
        # Mostrar si hubo errores
        if result.returncode != 0:
            st.error("Error durante la ejecución:")
            st.code(result.stderr)
        else:
            st.success("¡Ejecución completada con éxito!")
            
            # Leer y mostrar el log (omitiendo líneas con emojis)
            try:
                with open("testing_model_output.log", "r", encoding="utf-8") as f:
                    log_content = f.readlines()
                
                # Filtrar líneas con emojis problemáticos
                filtered_log = [line for line in log_content if '\U0001f3c3' not in line]
                
                st.subheader("Resultados de la ejecución:")
                st.code("".join(filtered_log))
            except Exception as e:
                st.warning(f"No se pudo leer el archivo de log: {e}")
                
            ## Mostrar enlaces a las imágenes generadas v1
            #st.subheader("Gráficos generados:")
            #col1, col2, col3 = st.columns(3)
            #
            ## Si existen las imágenes, mostrarlas
            #import os
            #if os.path.exists("correlation_with_target.png"):
            #    col1.image("correlation_with_target.png", caption="Correlación con Target")
            #
            #if os.path.exists("feature_importance.png"):
            #    col2.image("feature_importance.png", caption="Importancia de Características")
            #    
            #if os.path.exists("confusion_matrix.png"):
            #    col3.image("confusion_matrix.png", caption="Matriz de Confusión")

            ## Crear un checkbox opcional para mostrar las gráficas vV2
            #show_graphs = st.checkbox("Mostrar gráficas generadas", value=False)
            #
            #if show_graphs:
            #    st.subheader("Gráficos generados:")
            #    col1, col2, col3 = st.columns(3)
            #    
            #    # Si existen las imágenes, mostrarlas
            #    import os
            #    if os.path.exists("correlation_with_target.png"):
            #        col1.image("correlation_with_target.png", caption="Correlación con Target")
            #    
            #    if os.path.exists("feature_importance.png"):
            #        col2.image("feature_importance.png", caption="Importancia de Características")
            #        
            #    if os.path.exists("confusion_matrix.png"):
            #        col3.image("confusion_matrix.png", caption="Matriz de Confusión")                  
                
    except Exception as e:
        st.error(f"Error al ejecutar el script: {e}")
        import traceback
        st.error(traceback.format_exc())
## ---- Función para ejecutar testing_model.py ----
#
#** ERROR DE sys.stdout.write(f"\U0001f3c3 View run {run_name} at: {run_url}\n")
#MLflow sí intenta imprimir un emoji de persona corriendo (🏃) al final de una ejecución. 
#Esto ocurre en el código interno de MLflow cuando llama al método _log_url durante end_run().
# 
## def ejecutar_testing_model():
##     try:
##         st.info("Ejecutando testing_model.py... Por favor espere.")
##         result = subprocess.run(["python", "testing_model.py"], 
##                               capture_output=True, 
##                               text=True)
##         
##         # Mostrar salida del script
##         if result.returncode == 0:
##             st.success("¡Ejecución completada con éxito!")
##             st.code(result.stdout)
##         else:
##             st.error("Error durante la ejecución:")
##             st.code(result.stderr)
##     except Exception as e:
##         st.error(f"Error al ejecutar el script: {e}")


# ---- Función para abrir URL en nuevas pestañas ----
def abrir_url(url):
    # En Streamlit no podemos abrir directamente una pestaña,
    # pero podemos crear un botón que al hacer clic abra la URL
    js = f"""
    <a href="{url}" target="_blank">
        <button style="background-color:#4CAF50;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;">
            Abrir {url}
        </button>
    </a>
    """
    st.markdown(js, unsafe_allow_html=True)

# ==== 2. CONFIGURACIÓN DE LA PÁGINA y elementos iniciales ==== 

# Configuración de la página
st.set_page_config(
    page_title="Consulta de Datos de Pacientes",
    page_icon="🏥",
    layout="wide"
)

# Título de la aplicación
st.title("Sistema de Consulta de Pacientes con Riesgo de Accidente Cerebrovascular")

# Cargar el modelo y el preprocesador
model, preprocessor = cargar_modelo_ml()

# ==== 3. MENÚ LATERAL ====

# Sidebar para opciones de consulta
st.sidebar.title("Opciones de Consulta")
opcion = st.sidebar.radio(
    "Seleccione una opción:",
    ["Ver todos los pacientes", "Buscar por ID", "Recorrido entre pacientes", "MLFlow, Airflow, Minio"]
)

# ==== 4. BLOQUES CONDICIONALES PARA OPCIONES ====
# Mostrar todos los pacientes
if opcion == "Ver todos los pacientes":
    st.header("Lista de Todos los Pacientes")
    
    datos = obtener_datos()
    if datos is not None and not datos.empty:
        # Mostrar tabla con todos los datos
        st.dataframe(datos)
        
        # Estadísticas básicas
        st.subheader("Estadísticas Básicas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total de pacientes:** {len(datos)}")
            st.write(f"**Edad promedio:** {datos['age'].mean():.2f} años")
            st.write(f"**IMC promedio:** {datos['bmi'].mean():.2f}")
        
        with col2:
            st.write(f"**Pacientes con accidente cerebrovascular:** {datos['stroke'].sum()}")
            st.write(f"**Porcentaje con accidente cerebrovascular:** {(datos['stroke'].sum() / len(datos) * 100):.2f}%")
    else:
        st.warning("No se encontraron datos o hubo un problema al conectar con la base de datos.")

# Buscar por ID
elif opcion == "Buscar por ID":
    st.header("Buscar Paciente por ID")
    
    # Usar session_state para mantener el ID del paciente actual entre recargas
    if 'id_paciente_actual' not in st.session_state:
        st.session_state.id_paciente_actual = 1
    
    # Entrada para el ID con el valor actual
    id_paciente = st.number_input("Ingrese el ID del paciente:", 
                                min_value=1, 
                                step=1,
                                value=st.session_state.id_paciente_actual)
    
    # Botones de navegación arriba
    col_nav_top = st.columns([1, 1, 1, 1])
    
    with col_nav_top[0]:
        if st.button("Primero"):
            id_paciente = 1
            st.session_state.id_paciente_actual = id_paciente
    
    with col_nav_top[1]:
        if st.button("Anterior"):
            id_anterior = obtener_anterior_id(id_paciente)
            if id_anterior:
                id_paciente = id_anterior
                st.session_state.id_paciente_actual = id_paciente
            else:
                st.warning("Ya estás en el primer registro.")
    
    with col_nav_top[2]:
        if st.button("Siguiente"):
            id_siguiente = obtener_siguiente_id(id_paciente)
            if id_siguiente:
                id_paciente = id_siguiente
                st.session_state.id_paciente_actual = id_paciente
            else:
                st.warning("Ya estás en el último registro.")
    
    with col_nav_top[3]:
        if st.button("Último"):
            id_maximo = obtener_id_maximo()
            if id_maximo:
                id_paciente = id_maximo
                st.session_state.id_paciente_actual = id_paciente
    
    if st.button("Buscar"):
        st.session_state.id_paciente_actual = id_paciente
    
    # Mostrar datos del paciente
    datos = buscar_por_id(st.session_state.id_paciente_actual)
    if datos is not None and not datos.empty:
        st.subheader("Información del Paciente")
        
        # Obtener la primera fila (debería ser la única)
        paciente = datos.iloc[0]
        
        # Mostrar información usando la función actualizada
        mostrar_informacion_paciente(paciente, model, preprocessor)
    else:
        st.warning(f"No se encontró ningún paciente con el ID {st.session_state.id_paciente_actual}.")

# Recorrido entre pacientes
elif opcion == "Recorrido entre pacientes":
    st.header("Recorrido entre pacientes")
    
    # Inicializar session_state para el ID actual si no existe
    if 'paciente_actual_id' not in st.session_state:
        # Obtener el ID mínimo como punto de partida
        id_minimo = obtener_id_minimo()
        if id_minimo:
            st.session_state.paciente_actual_id = id_minimo
        else:
            st.session_state.paciente_actual_id = 1
    
    # Mostrar el ID actual
    st.subheader(f"Paciente ID: {st.session_state.paciente_actual_id}")
    
    # Botones de navegación
    col_nav = st.columns([1, 1])
    
    with col_nav[0]:
        if st.button("⏪ Anterior"):
            id_anterior = obtener_anterior_id(st.session_state.paciente_actual_id)
            if id_anterior:
                st.session_state.paciente_actual_id = id_anterior
                st.rerun()
            else:
                st.warning("Ya estás en el primer registro.")
    
    with col_nav[1]:
        if st.button("Siguiente ⏩"):
            id_siguiente = obtener_siguiente_id(st.session_state.paciente_actual_id)
            if id_siguiente:
                st.session_state.paciente_actual_id = id_siguiente
                st.rerun()
            else:
                st.warning("Ya estás en el último registro.")
    
    # Obtener y mostrar los datos del paciente actual
    datos = buscar_por_id(st.session_state.paciente_actual_id)
    if datos is not None and not datos.empty:
        st.subheader("Información del Paciente")
        paciente = datos.iloc[0]
        # Mostrar información usando la función actualizada
        mostrar_informacion_paciente(paciente, model, preprocessor)
    else:
        st.error(f"No se encontró ningún paciente con el ID {st.session_state.paciente_actual_id}.")
        # Intentar restablecer al ID mínimo
        id_minimo = obtener_id_minimo()
        if id_minimo:
            st.session_state.paciente_actual_id = id_minimo
            st.info(f"Se ha restablecido al primer paciente disponible (ID: {id_minimo}).")
            st.rerun()

# Nueva opción: MLFlow, Airflow, Minio
elif opcion == "MLFlow, Airflow, Minio":
    st.header("Ejecución de Programas")
    
    st.markdown("""
    ## Objetivo
    El objetivo principal es emular un ambiente de desarrollo controlado y de producción ejecutando estos programas que están en un contenedor Docker:
    """)
    
    # Pasos numerados con acciones y enlaces
    st.subheader("Pasos a seguir:")
    
    # Paso 1: Minio
    st.markdown("### 1. Ingresar a Minio")
    st.write("Presiona el siguiente enlace para acceder a Minio. usuario y contraseña es 'airflow'. Verificaremos que no contiene información.")
    abrir_url("http://localhost:9001/")
    
    # Paso 2: Airflow
    st.markdown("### 2. Ingresar a Apache Airflow")
    st.write("Accede a Airflow usuario y contraseña es 'airflow' y ejecuta el pipeline 'process_etl_stroke_data_v3'. Este pipeline realiza todo el proceso de Extraer, Transformar y Cargar (ETL) de los datos, almacenándolos en el simulador de Amazon S3 (Minio) y realiza la transformación de datos para nuestro Modelo de Testing.")
    abrir_url("http://localhost:8080/")
    
    # Paso 3: Actualizar Minio
    st.markdown("### 3. Actualizar Minio")
    st.write("Después de ejecutar el pipeline, actualiza Minio para ver los cambios:")
    abrir_url("http://localhost:9001/")
    
    # Paso 4: MLFlow
    st.markdown("### 4. Ingresar a MLFlow")
    st.write("Accede a MLFlow --> Experiments --> Stroke Prediction  y vemos cada uno de los procesos :")
    abrir_url("http://localhost:5000/")
    
    # Paso 5: Ejecutar testing_model.py
    st.markdown("### 5. Ejecutar testing_model.py")
    st.write("Este programa ejecuta un modelo simple que está vinculado a MLFlow, enviando las métricas del entrenamiento, gráficas y más información.")
    if st.button("Ejecutar testing_model.py"):
        ejecutar_testing_model()
    
    # Paso 6: Actualizar MLFlow
    st.markdown("### 6. Actualizar MLFlow")
    st.write("Después de ejecutar el modelo, actualizamos MLFlow --> Experiments --> Stroke Prediction -> experiment_id:  stroke_prediction :")
    abrir_url("http://localhost:5000/")
    
    st.info("Nota: Asegúrate de que todos los servicios (Minio, Airflow, MLFlow)  estén indicado Comando docker para iniciar todos los  procesos es: docker compose --profile all up")

# Pie de página
st.sidebar.markdown("---")
# Sistema de Consulta de Pacientes v ya ni me acuerdo 
st.sidebar.info("Sistema de Consulta de Pacientes con ML Optimizado y Herramientas de Flujo de Trabajo")
st.sidebar.title("El objetivo de esta app:")
st.sidebar.write("•	3 primeras opciones es implementar un ambiente de producción de un modelo ML ya entrenado (.pth y .pkl)")
st.sidebar.write("•	Opción MLFlow, Airflow, Minio hacer un circuito de trabajo donde interactuamos con dichas app ")
