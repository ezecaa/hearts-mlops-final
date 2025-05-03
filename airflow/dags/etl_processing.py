"""
Este DAG extrae datos de un archivo CSV fuente (healthcare-dataset-stroke-data.csv),
transforma los datos y los carga en un bucket S3/Minio de destino
como dos archivos CSV separados, uno para entrenamiento y otro para pruebas.
La división entre los datos de entrenamiento y prueba es 70/30.
"""
from datetime import timedelta
from airflow.decorators import dag, task
import logging
from airflow.models import Variable

# Definir valores predeterminados para las variables de Airflow
try:
    Variable.get("target_col")
except:
    Variable.set("target_col", "stroke")

try:
    Variable.get("test_size_stroke")
except:
    Variable.set("test_size_stroke", "0.3")

MARKDOWN_TEXT = """
# Pipeline ETL para Datos de Stroke

Este DAG extrae datos de un archivo CSV fuente (healthcare-dataset-stroke-data.csv),
transforma los datos y los carga en un bucket S3/Minio de destino
como dos archivos CSV separados, uno para entrenamiento y otro para pruebas.
La división entre los datos de entrenamiento y prueba es 70/30.
"""

default_args = {
    'owner': 'Brain Stroke Project',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15)
}

# Inicializar el registro para la ejecución del DAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dag(
    dag_id='process_etl_stroke_data_v3',
    description='ETL process for stroke prediction data, splitting data into training and testing datasets',
    doc_md=MARKDOWN_TEXT,
    tags=['etl', 'stroke_prediction'],
    default_args=default_args,
    catchup=False
)
def etl_processing():
    """
    Proceso ETL para los datos de predicción de stroke, extrayendo datos de un archivo CSV fuente,
    transformando los datos y cargándolos en un bucket S3/Minio de destino como dos archivos CSV separados.
    """
    @task.virtualenv(
        task_id='get_original_data',
        requirements=["pandas", "awswrangler==3.9.1", "mlflow>=2.8.0"],
        system_site_packages=True
    )

    def get_data() -> None:
        """
        Carga el conjunto de datos original desde un archivo local y lo guarda en S3/Minio.
        Punto de origen de los datos: El DAG obtiene los datos originales directamente desde un archivo CSV local
        en la carpeta /opt/airflow/dataset/healthcare-dataset-stroke-data.csv. Luego, estos datos crudos 
        se guardan en S3/Minio en la ruta s3://data/raw/stroke_data.csv para los pasos siguientes del proceso ETL.
        """
        # Importar las bibliotecas necesarias
        import logging
        import awswrangler as wr
        import pandas as pd
        import os
        import mlflow
        import datetime

        # Inicializar el registro
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Definir la ruta del archivo original y la ruta S3 donde se guardará
        source_path = '/opt/airflow/dataset/healthcare-dataset-stroke-data.csv'
        data_path = 's3://data/raw/stroke_data.csv'

        logger.info(f"Starting to read the dataset from {source_path}")

        try:
            # Verificar si el archivo existe
            if not os.path.exists(source_path):
                logger.error(f"File not found: {source_path}")
                raise FileNotFoundError(f"The file {source_path} was not found")
                
            # Leer el archivo CSV
            df = pd.read_csv(source_path)
            logger.info(f"Dataset read successfully from {source_path}")
            logger.info(f"Dataset shape: {df.shape}")
            
            # Verificar que el archivo tenga el formato esperado
            expected_columns = [
                'id', 'gender', 'age', 'hypertension', 'heart_disease', 
                'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 
                'bmi', 'smoking_status', 'stroke'
            ]
            
            # Normalizar nombres de columnas (para manejar posibles diferencias en mayúsculas/minúsculas)
            df.columns = [col.lower() for col in df.columns]
            
            # Renombrar residence_type si es necesario para mantener consistencia
            if 'residence_type' not in df.columns and 'residence_type' in [col.lower() for col in df.columns]:
                df.rename(columns={'residence_type': 'residence_type'}, inplace=True)
                
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            # Registrar información en MLflow
            mlflow.set_tracking_uri('http://mlflow:5000')
            experiment = mlflow.set_experiment("Stroke Prediction")
            
            # Iniciar una ejecución de MLflow para la tarea de obtención de datos
            run_name = 'get_data_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with mlflow.start_run(run_name=run_name):
                # Registrar parámetros
                mlflow.log_params({
                    'source_file': source_path,
                    'destination_path': data_path,
                    'row_count': df.shape[0],
                    'column_count': df.shape[1],
                })
                
                # Registrar métricas
                column_metrics = {
                    f'null_count_{col}': df[col].isnull().sum() for col in df.columns
                }
                mlflow.log_metrics(column_metrics)
                
                # Registrar información adicional
                mlflow.set_tags({
                    'task': 'data_ingestion',
                    'file_type': 'csv',
                    'airflow_dag_id': 'process_etl_stroke_data_v3'
                })
                
                logger.info(f"Logged dataset information to MLflow in run: {run_name}")
            
        except Exception as e:
            logger.error(f"Error al leer el archivo CSV: {e}")
            raise

        logger.info(f"Starting to save the dataset to S3/Minio at {data_path}")

        try:
            # Guardar el conjunto de datos original en S3/Minio
            wr.s3.to_csv(df, data_path, index=False)
            logger.info(f"Dataset saved successfully to {data_path}")
        except Exception as e:
            logger.error(f"Error al guardar el conjunto de datos en S3/Minio: {e}")
            raise
        
    @task.virtualenv(
        task_id='feature_engineering',
        requirements=["awswrangler==3.9.1", "mlflow>=2.8.0", "boto3"],
        system_site_packages=True
    )
    def feature_engineering() -> None:
        """
        Realiza ingeniería de características en el conjunto de datos de stroke
        """
        import json
        import datetime
        import logging
        import boto3
        import botocore.exceptions
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from airflow.models import Variable
        
        # Inicializar el registro
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def one_hot_encode(df: pd.DataFrame, one_hot_cols: list) -> pd.DataFrame:
            """
            Aplica One-Hot Encoding a las columnas especificadas de un DataFrame.

            Parameters:
            - df (pd.DataFrame): El DataFrame de entrada.
            - one_hot_cols (list): Lista de nombres de columnas a las que aplicar One-Hot Encoding.

            Returns:
            - pd.DataFrame: El DataFrame con las columnas codificadas con One-Hot.
            """
            logger.info(f"Applying One-Hot Encoding to columns: {one_hot_cols}")
            # Aplicar One-Hot Encoding y devolver el DataFrame modificado
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
            return df
        
        # Cargar Conjunto de Datos
        logger.info("Loading dataset from S3/Minio")
        # Definir rutas para datos originales y procesados
        data_original_path = 's3://data/raw/stroke_data.csv'
        data_processed_path = 's3://data/processed/stroke_data.csv'
        
        try:
            # Leer el conjunto de datos original desde S3/Minio
            df = wr.s3.read_csv(data_original_path)
            logger.info(f"Dataset loaded successfully from {data_original_path}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {list(df.columns)}")
        except Exception as e:  
            logger.error(f"Failed to load dataset from {data_original_path}: {e}")
            raise
        
        # Data Cleaning
        logger.info("Limpiando datos eliminando duplicados y manejando valores nulos")
        
        # Guardar número de filas original para información
        original_row_count = len(df)
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        duplicates_removed = original_row_count - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Manejar valores nulos
        # Para BMI, que suele tener valores nulos, imputar con la mediana
        bmi_nulls = df['bmi'].isnull().sum()
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        logger.info(f"Imputed {bmi_nulls} null values in 'bmi' column with median value")
        
        # Verificar si hay otros valores nulos y reportarlos
        null_counts = df.isnull().sum()
        null_rows_removed = 0
        if null_counts.sum() > 0:
            logger.info(f"Remaining null values by column:\n{null_counts[null_counts > 0]}")
            # Eliminar filas con cualquier valor nulo restante
            rows_before = len(df)
            df = df.dropna()
            null_rows_removed = rows_before - len(df)
            logger.info(f"Removed {null_rows_removed} rows with null values. New dataset shape: {df.shape}")
        else:
            logger.info("No null values remaining in the dataset")
            
        # Feature Engineering
        logger.info("Iniciando ingeniería de características")
        
        # Convertir variables categóricas a numéricas
        categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
        
        # Aplicar One-Hot Encoding
        df = one_hot_encode(df, categorical_cols)
        logger.info("Applied one-hot encoding to categorical features")
        
        # Eliminar la columna 'id' ya que no es relevante para la predicción
        id_column_removed = False
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            id_column_removed = True
            logger.info("Removed 'id' column as it's not relevant for prediction")
        
        # Verificar si hay valores extremos en variables numéricas y truncarlos si es necesario
        numeric_cols = ['age', 'avg_glucose_level', 'bmi']
        truncation_counts = {}
        for col in numeric_cols:
            if col in df.columns:
                # Truncar valores extremos (por ejemplo, por encima del percentil 99)
                upper_limit = df[col].quantile(0.99)
                rows_before = (df[col] > upper_limit).sum()
                df[col] = df[col].clip(upper=upper_limit)
                truncation_counts[col] = rows_before
                logger.info(f"Truncated {rows_before} extreme values in '{col}' column at {upper_limit}")
        
        logger.info("Feature engineering completed")
        
        # Save Processed Data
        logger.info(f"Saving processed dataset to {data_processed_path}")
        # Guardar el conjunto de datos procesado en S3/Minio
        try:
            wr.s3.to_csv(df, data_processed_path, index=False)
            logger.info(f"Processed dataset saved successfully to {data_processed_path}")
        except Exception as e:
            logger.error(f"Failed to save processed dataset to {data_processed_path}: {e}")
            raise

        # Actualizar Información del Conjunto de Datos
        logger.info("Updating dataset information in S3/Minio")
        # Inicializar cliente S3 para gestionar la información del conjunto de datos
        s3_client = boto3.client('s3')
        data_dict = {}
        
        # Intentar obtener información existente del conjunto de datos
        try:
            s3_client.head_object(Bucket='data', Key='data_info/stroke_data_info.json')
            result = s3_client.get_object(Bucket='data', Key='data_info/stroke_data_info.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
            logger.info("Existing dataset information loaded from S3/Minio")
        except botocore.exceptions.ClientError as e:
            # Si no se encuentra el objeto, continuar con un diccionario vacío
            if e.response['Error']['Code'] == "404":
                logger.info("No existing dataset information found, initializing new info dictionary")
            else:
                logger.error(f"Failed to get dataset information: {e}")
                raise

        # Obtener la columna objetivo (normalmente 'stroke')
        target_col = 'stroke'  # Variable predeterminada
        try:
            # Intentar obtener de la Variable de Airflow si está definida
            target_col = Variable.get("target_col")
        except:
            logger.info(f"Variable 'target_col' not found, using default: {target_col}")
            # Crear la variable si no existe
            Variable.set("target_col", target_col)
        
        logger.info(f"Target column for dataset: {target_col}")
        
        # Verificar que la columna objetivo exista en el DataFrame
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            # Establecer el valor correcto para los datos de stroke
            target_col = 'stroke'
            Variable.set("target_col", target_col)
            logger.info(f"Setting target column to default value: {target_col}")
            
            # Verificar nuevamente
            if target_col not in df.columns:
                logger.error(f"Default target column '{target_col}' still not found in DataFrame. Available columns: {df.columns.tolist()}")
                raise KeyError(f"Target column '{target_col}' not found in DataFrame")
        
        # Preparar la información de los datos para el registro
        dataset_log = df.drop(columns=[target_col])
        
        # Actualizar el diccionario de datos con los detalles del conjunto de datos
        data_dict['columns'] = dataset_log.columns.tolist()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categorical_cols
        data_dict['numeric_columns'] = numeric_cols
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}

        # Añadir la fecha y hora actual al diccionario de datos
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        
        # Convertir el diccionario de datos a cadena JSON
        data_string = json.dumps(data_dict, indent=2)

        # Guardar el diccionario de datos actualizado de nuevo en S3/Minio
        try:
            s3_client.put_object(
                Bucket='data',
                Key='data_info/stroke_data_info.json',
                Body=data_string
            )
            logger.info("Dataset information updated successfully in S3/Minio")
        except Exception as e:
            logger.error(f"Failed to update dataset information in S3/Minio: {e}")
            raise
        
        # Log Data to MLflow
        logger.info("Logging data to MLflow")

        # Configurar el servidor de seguimiento de MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Stroke Prediction")

        # Iniciar una nueva ejecución de MLflow
        run_name = 'feature_engineering_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros de procesamiento
            mlflow.log_params({
                'categorical_columns': ','.join(categorical_cols),
                'numeric_columns': ','.join(numeric_cols),
                'target_column': target_col,
                'id_column_removed': id_column_removed,
                'data_shape_after_processing': f"{df.shape[0]}x{df.shape[1]}",
                'one_hot_encoding_applied': True
            })
            
            # Registrar métricas
            mlflow.log_metrics({
                'duplicates_removed': duplicates_removed,
                'bmi_nulls_imputed': bmi_nulls,
                'null_rows_removed': null_rows_removed,
                'feature_count': len(dataset_log.columns),
                'processed_row_count': len(df)
            })
            
            # Registrar métricas de truncamiento
            for col, count in truncation_counts.items():
                mlflow.log_metric(f'truncated_values_{col}', count)
            
            # Registrar tags
            mlflow.set_tags({
                'task': 'feature_engineering',
                'processing_step': 'data_transformation',
                'airflow_dag_id': 'process_etl_stroke_data_v3'
            })
            
            # Registrar el conjunto de datos procesado
            mlflow_dataset = mlflow.data.from_pandas(
                df,
                source="local file: healthcare-dataset-stroke-data.csv",
                targets=target_col,
                name="stroke_data_processed"
            )
            
            # Registrar conjunto de datos en MLflow
            mlflow.log_input(mlflow_dataset, context="Dataset")
            logger.info(f"Dataset logged to MLflow successfully in run: {run_name}")
        
        logger.info("MLflow run ended")

    
    @task.virtualenv(
        task_id='split_dataset',
        requirements=["awswrangler==3.9.1", "mlflow>=2.8.0", "scikit-learn"],
        system_site_packages=True
    )
    def split_dataset() -> None:
        """
        Genera una división del conjunto de datos en una parte de entrenamiento y una parte de prueba, y las guarda en S3/Minio.
        """
        # Importar las bibliotecas necesarias
        import logging
        import awswrangler as wr
        import mlflow
        import datetime
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        # Inicializar el registro
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Definir rutas para datos procesados y divisiones de entrenamiento/prueba
        data_processed_path = 's3://data/processed/stroke_data.csv'
        X_train_path = 's3://data/train/stroke_data_X_train.csv'
        X_test_path = 's3://data/test/stroke_data_X_test.csv'
        y_train_path = 's3://data/train/stroke_data_y_train.csv'
        y_test_path = 's3://data/test/stroke_data_y_test.csv'

        logger.info(f"Starting to read processed dataset from {data_processed_path}")

        try:
            # Leer el conjunto de datos procesado desde S3/Minio
            data = wr.s3.read_csv(data_processed_path)
            logger.info(f"Processed dataset loaded successfully from {data_processed_path}")
        except Exception as e:
            logger.error(f"Failed to load processed dataset from {data_processed_path}: {e}")
            raise

        try:
            # Obtener la columna objetivo de la Variable de Airflow
            target_col = Variable.get("target_col")
            logger.info(f"Target column for the dataset: {target_col}")
            
            # Verificar que la columna objetivo exista en el DataFrame
            if target_col not in data.columns:
                logger.error(f"Target column '{target_col}' not found in DataFrame. Available columns: {data.columns.tolist()}")
                # Establecer el valor correcto para los datos de stroke
                target_col = 'stroke'
                Variable.set("target_col", target_col)
                logger.info(f"Setting target column to default value: {target_col}")
        except Exception as e:
            # Usar valor predeterminado si la variable no está definida
            target_col = 'stroke'
            logger.warning(f"Failed to get target column from Airflow Variable: {e}. Using default: {target_col}")
            # Crear la variable si no existe
            Variable.set("target_col", target_col)

        try:
            # Definir características y objetivo
            X = data.drop(columns=target_col, axis=1)
            y = data[target_col]
            logger.info("Features and target separated successfully")
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Verificar distribución de clases en el objetivo (importante para datasets desbalanceados)
            class_distribution = y.value_counts(normalize=True) * 100
            logger.info(f"Class distribution in target column:\n{class_distribution}")
        except Exception as e:
            logger.error(f"Failed to separate features and target: {e}")
            raise

        try:
            # Definir el tamaño de prueba (predeterminado: 0.3 = 30%)
            test_size = 0.3
            
            # Intentar obtener de Variable de Airflow si está definida
            try:
                test_size = float(Variable.get("test_size_stroke"))
            except:
                logger.info(f"Variable 'test_size_stroke' not found, using default: {test_size}")
                # Crear la variable si no existe
                Variable.set("test_size_stroke", str(test_size))
                
            logger.info(f"Test size for dataset split: {test_size}")

            # Dividir los datos en conjuntos de entrenamiento y prueba
            # Usar stratify para mantener la misma proporción de la clase objetivo en ambos conjuntos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("Dataset split into training and testing sets successfully")
            logger.info(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")
            
            # Verificar distribución de clases después de la división
            train_distribution = y_train.value_counts(normalize=True) * 100
            test_distribution = y_test.value_counts(normalize=True) * 100
            logger.info(f"Class distribution in training set:\n{train_distribution}")
            logger.info(f"Class distribution in testing set:\n{test_distribution}")
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            raise

        logger.info("Starting to save training and testing datasets to S3/Minio")

        try:
            # Guardar los conjuntos de datos de entrenamiento y prueba en S3/Minio
            wr.s3.to_csv(X_train, X_train_path, index=False)
            logger.info(f"Training features saved successfully to {X_train_path}")
            
            wr.s3.to_csv(X_test, X_test_path, index=False)
            logger.info(f"Testing features saved successfully to {X_test_path}")

            wr.s3.to_csv(y_train, y_train_path, index=False)
            logger.info(f"Training target saved successfully to {y_train_path}")

            wr.s3.to_csv(y_test, y_test_path, index=False)
            logger.info(f"Testing target saved successfully to {y_test_path}")
        except Exception as e:
            logger.error(f"Failed to save training or testing datasets to S3/Minio: {e}")
            raise

        # Log Data Split to MLflow
        logger.info("Logging data split information to MLflow")

        # Configurar el servidor de seguimiento de MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Stroke Prediction")

        # Iniciar una nueva ejecución de MLflow
        run_name = 'split_dataset_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros
            mlflow.log_params({
                'test_size': test_size,
                'random_state': 42,
                'stratified_split': True,
                'X_train_shape': f"{X_train.shape[0]}x{X_train.shape[1]}",
                'X_test_shape': f"{X_test.shape[0]}x{X_test.shape[1]}",
                'feature_count': X_train.shape[1]
            })
            
            # Registrar métricas
            mlflow.log_metrics({
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'train_ratio': 1 - test_size,
                'test_ratio': test_size,
                'class_1_proportion_train': float(train_distribution.get(1, 0)),
                'class_1_proportion_test': float(test_distribution.get(1, 0))
            })
            
            # Registrar tags
            mlflow.set_tags({
                'task': 'data_splitting',
                'processing_step': 'train_test_split',
                'airflow_dag_id': 'process_etl_stroke_data_v3'
            })
            
            logger.info(f"Data split information logged to MLflow successfully in run: {run_name}")

        logger.info("Dataset splitting and saving completed successfully")

    @task.virtualenv(
        task_id='normalize_data',
        requirements=["awswrangler==3.9.1", "mlflow>=2.8.0", "scikit-learn", "boto3"],
        system_site_packages=True
    )
    def normalize_data() -> None:
        """
        Normaliza los conjuntos de datos de entrenamiento y prueba usando StandardScaler y guarda los conjuntos de datos escalados en S3/Minio.
        Además, registra los parámetros del escalador y la información del conjunto de datos en MLflow.
        """
        # Importar las bibliotecas necesarias
        import logging
        import json
        import mlflow
        import boto3
        import botocore.exceptions
        import pandas as pd
        import numpy as np
        import awswrangler as wr
        import datetime
        from sklearn.preprocessing import StandardScaler

        # Inicializar el registro
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Definir rutas para datos de entrenamiento/prueba y datos escalados
        X_train_path = 's3://data/train/stroke_data_X_train.csv'
        X_test_path = 's3://data/test/stroke_data_X_test.csv'
        X_train_scaled_path = 's3://data/train/stroke_data_X_train_scaled.csv'
        X_test_scaled_path = 's3://data/test/stroke_data_X_test_scaled.csv'

        logger.info("Starting to read training and testing datasets from S3/Minio")

        try:
            # Leer los conjuntos de datos de entrenamiento y prueba desde S3/Minio
            X_train = wr.s3.read_csv(X_train_path)
            X_test = wr.s3.read_csv(X_test_path)
            logger.info("Training and testing datasets loaded successfully from S3/Minio")
            logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
        except Exception as e:
            logger.error(f"Failed to load training or testing datasets from S3/Minio: {e}")
            raise

        logger.info("Initializing the StandardScaler")

        try:
            # Inicializar el escalador
            scaler = StandardScaler(with_mean=True, with_std=True)
            
            # Registrar estadísticas antes del escalado
            pre_scaling_stats = {}
            for col in X_train.columns:
                pre_scaling_stats[f'train_{col}_mean'] = float(X_train[col].mean())
                pre_scaling_stats[f'train_{col}_std'] = float(X_train[col].std())
                pre_scaling_stats[f'train_{col}_min'] = float(X_train[col].min())
                pre_scaling_stats[f'train_{col}_max'] = float(X_train[col].max())

            # Ajustar el escalador en los datos de entrenamiento y transformar ambos conjuntos
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.info("Data normalization completed successfully using StandardScaler")
            
            # Registrar parámetros del escalador
            # Registrar parámetros del escalador
            logger.info(f"Scaler mean: {scaler.mean_[:5]}... (showing first 5 elements)")
            logger.info(f"Scaler scale: {scaler.scale_[:5]}... (showing first 5 elements)")
            
            # Registrar estadísticas después del escalado
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            post_scaling_stats = {}
            for col in X_train_scaled_df.columns:
                post_scaling_stats[f'train_scaled_{col}_mean'] = float(X_train_scaled_df[col].mean())
                post_scaling_stats[f'train_scaled_{col}_std'] = float(X_train_scaled_df[col].std())
                post_scaling_stats[f'train_scaled_{col}_min'] = float(X_train_scaled_df[col].min())
                post_scaling_stats[f'train_scaled_{col}_max'] = float(X_train_scaled_df[col].max())
                
        except Exception as e:
            logger.error(f"Failed during data normalization with StandardScaler: {e}")
            raise

        try:
            # Convertir los datos escalados a DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            logger.info("Scaled data successfully converted to DataFrame")
        except Exception as e:
            logger.error(f"Failed to convert scaled data to DataFrame: {e}")
            raise

        logger.info("Starting to save scaled training and testing datasets to S3/Minio")

        try:
            # Guardar los conjuntos de datos de entrenamiento y prueba escalados en S3/Minio
            wr.s3.to_csv(X_train_scaled, X_train_scaled_path, index=False)
            logger.info(f"Scaled training data saved successfully to {X_train_scaled_path}")

            wr.s3.to_csv(X_test_scaled, X_test_scaled_path, index=False)
            logger.info(f"Scaled testing data saved successfully to {X_test_scaled_path}")
        except Exception as e:
            logger.error(f"Failed to save scaled datasets to S3/Minio: {e}")
            raise

        logger.info("Updating dataset information with scaler details")

        try:
            # Inicializar cliente S3 e intentar cargar la información de datos existente
            client = boto3.client('s3') 
            data_dict = {}
            try:
                client.head_object(Bucket='data', Key='data_info/stroke_data_info.json')
                result = client.get_object(Bucket='data', Key='data_info/stroke_data_info.json')
                text = result["Body"].read().decode()
                data_dict = json.loads(text)
                logger.info("Existing dataset information loaded from S3/Minio")
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.info("No existing dataset information found, initializing new info dictionary")
                else:
                    logger.error(f"Failed to get dataset information: {e}")
                    raise

            # Actualizar el diccionario de datos con la información del escalador
            data_dict['standard_scaler_mean'] = scaler.mean_.tolist()
            data_dict['standard_scaler_std'] = scaler.scale_.tolist()
            data_string = json.dumps(data_dict, indent=2)

            # Guardar el diccionario de datos actualizado de nuevo en S3/Minio
            client.put_object(
                Bucket='data',
                Key='data_info/stroke_data_info.json',
                Body=data_string
            )
            logger.info("Dataset information updated successfully in S3/Minio with scaler details")
        except Exception as e:
            logger.error(f"Failed to update dataset information in S3/Minio: {e}")
            raise

        # Log Data to MLflow
        logger.info("Logging scaler details and dataset info to MLflow")

        # Configurar el servidor de seguimiento de MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Stroke Prediction")

        # Iniciar una nueva ejecución de MLflow
        run_name = 'normalize_data_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros de normalización
            scaler_params = {
                'scaler_type': 'StandardScaler',
                'with_mean': scaler.with_mean,
                'with_std': scaler.with_std,
                'feature_count': len(scaler.feature_names_in_)
            }
            mlflow.log_params(scaler_params)
            
            # Registrar algunas estadísticas antes y después del escalado como métricas
            # Estas métricas son útiles para entender cómo cambió la distribución de los datos
            for col in X_train.columns[:5]:  # Limitar a las primeras 5 columnas para no sobrecargar
                mlflow.log_metrics({
                    f'before_scaling_mean_{col}': pre_scaling_stats[f'train_{col}_mean'],
                    f'after_scaling_mean_{col}': post_scaling_stats[f'train_scaled_{col}_mean'],
                    f'before_scaling_std_{col}': pre_scaling_stats[f'train_{col}_std'],
                    f'after_scaling_std_{col}': post_scaling_stats[f'train_scaled_{col}_std']
                })
            
            # Registrar métricas generales
            mlflow.log_metrics({
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_count': X_train.shape[1]
            })
            
            # Registrar tags
            mlflow.set_tags({
                'task': 'data_normalization',
                'processing_step': 'feature_scaling',
                'airflow_dag_id': 'process_etl_stroke_data_v3'
            })
            
            logger.info(f"Scaler details and dataset info logged to MLflow successfully in run: {run_name}")

        logger.info("Data normalization process completed successfully")
    
    # Configurar las dependencias de tareas con registro
    logger.info("Setting up task dependencies for the DAG")
    
    # Definir el orden de las tareas en el DAG
    get_data() >> feature_engineering() >> split_dataset() >> normalize_data()
    
    logger.info("Task dependencies set up successfully")
    
# Inicializar el DAG
dag = etl_processing()