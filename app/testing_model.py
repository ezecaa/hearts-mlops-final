import mlflow
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow import MlflowClient
import warnings
import awswrangler as wr
import traceback
import logging
import sys

# Force PYTHONIOENCODING to UTF-8 solventa error de emoji ERROR DE sys.stdout.write(f"\U0001f3c3 View run {run_name} at: {run_url}\n")
os.environ["PYTHONIOENCODING"] = "utf-8"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# Configurar las variables de entorno para Minio (reemplazando los comandos %env)
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"

def main():
    logger.info("Iniciando el proceso de entrenamiento y evaluación del modelo...")
    
    # Configurar MLflow
    mlflow_server = "http://localhost:5000"
    mlflow.set_tracking_uri(mlflow_server)
    logger.info(f"MLflow configurado en: {mlflow_server}")
    
    # Función para graficar la correlación con el target
    def plot_correlation_with_target(X_df, y_df):
        try:
            # Combinar los dataframes
            combined_df = pd.concat([X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
            
            # Calcular la matriz de correlación
            correlation_matrix = combined_df.corr()
            
            # Obtener la correlación con el target (última columna)
            correlation_with_target = correlation_matrix.iloc[:-1, -1].sort_values(ascending=False)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index)
            plt.title('Correlación de características con Target (stroke)')
            plt.xlabel('Coeficiente de Correlación')
            plt.ylabel('Variables')
            plt.tight_layout()
            
            # Guardar la figura localmente pero no mostrarla
            plt.savefig("correlation_with_target.png")
            fig = plt.gcf()  # Obtener la figura actual
            plt.close()  # Cerrar la figura para no mostrarla
            
            return fig
        except Exception as e:
            logger.error(f"Error al generar gráfico de correlación: {str(e)}")
            return None

    # Función para graficar la importancia de características
    def plot_feature_importance(model, X_df, y_df):
        try:
            # Importancia de características
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            
            # Crear la figura
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), np.array(X_df.columns)[sorted_idx])
            plt.title('Importancia de Características (Random Forest)')
            plt.tight_layout()
            
            # Guardar la figura localmente pero no mostrarla
            plt.savefig("feature_importance.png")
            fig = plt.gcf()  # Obtener la figura actual
            plt.close()  # Cerrar la figura para no mostrarla
            
            return fig
        except Exception as e:
            logger.error(f"Error al generar gráfico de importancia de características: {str(e)}")
            return None

    # Función para obtener o crear un experimento
    def get_or_create_experiment(experiment_name):
        try:
            # Verificar si el experimento existe
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is not None:
                return experiment.experiment_id
            else:
                # Crear un nuevo experimento si no existe
                experiment_id = mlflow.create_experiment(experiment_name)
                return experiment_id
        except Exception as e:
            logger.error(f"Error al obtener/crear experimento: {str(e)}")
            return None
    
    try:
        logger.info("Cargando datos desde Minio...")
        # Cargar los datos procesados de Minio
        X_train_df = wr.s3.read_csv("s3://data/train/stroke_data_X_train_scaled.csv")
        y_train_df = wr.s3.read_csv("s3://data/train/stroke_data_y_train.csv")
        X_test_df = wr.s3.read_csv("s3://data/test/stroke_data_X_test_scaled.csv")
        y_test_df = wr.s3.read_csv("s3://data/test/stroke_data_y_test.csv")
        
        logger.info(f"Dimensiones de los datos: X_train={X_train_df.shape}, y_train={y_train_df.shape}, X_test={X_test_df.shape}, y_test={y_test_df.shape}")
        
        # Generar la gráfica de correlación pero no mostrarla
        logger.info("Generando gráficas de correlación...")
        corr_plot = plot_correlation_with_target(X_train_df, y_train_df)
        logger.info("Gráfica de correlación generada y guardada como 'correlation_with_target.png'")
        
        # Convertir los dataframes a arrays numpy
        X_train = X_train_df.to_numpy()
        y_train = y_train_df.to_numpy().ravel()
        X_test = X_test_df.to_numpy()
        y_test = y_test_df.to_numpy().ravel()
        
        # Crear o obtener el ID del experimento para MLflow
        experiment_id = get_or_create_experiment("Stroke Prediction")
        logger.info(f"Experiment ID: {experiment_id}")
        
        # Definir el nombre de la ejecución para MLflow
        run_name_parent = "stroke_prediction_" + datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
        
        # Mostrar el tamaño de los conjuntos de entrenamiento y prueba
        logger.info("Tamaño de los conjuntos:")
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Definir la cuadrícula de parámetros para Random Forest
        logger.info("Configurando Grid Search para Random Forest...")
        param_grid_rf = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]  # Importante para datasets desbalanceados
        }
        
        # Inicializar el modelo Random Forest Classifier (para clasificación)
        rf_model = RandomForestClassifier(random_state=42)
        
        # Configurar la búsqueda en cuadrícula con validación cruzada de 5 pliegues
        grid_search_rf = GridSearchCV(
            estimator=rf_model, 
            param_grid=param_grid_rf, 
            cv=5, 
            scoring='roc_auc',  # AUC ROC es una buena métrica para problemas desbalanceados
            n_jobs=-1  # Usar todos los núcleos disponibles
        )
        
        logger.info("Iniciando entrenamiento con GridSearchCV...")
        
        # Modificación para evitar el error de Unicode
        # Guardar la configuración actual de MLflow
        original_tracking_uri = mlflow.get_tracking_uri()
        
        # Iniciar manualmente el run en lugar de usar context manager
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent, nested=True)
        run_id = run.info.run_id
        # ERROR 'charmap' codec can't encode character '\U0001f3c3' in position 0: character maps to <undefined>
        logger.info(f"MLflow Run ID: {run_id}") 
        
        try:
            # Realizar la búsqueda en cuadrícula y ajustar el modelo
            grid_search_rf.fit(X_train, y_train)
            logger.info("Entrenamiento completado.")
            
            # Obtener el mejor modelo de la búsqueda en cuadrícula
            best_rf_model = grid_search_rf.best_estimator_
            
            # Realizar predicciones usando el mejor modelo
            logger.info("Evaluando el modelo en el conjunto de prueba...")
            rf_predictions_prob = best_rf_model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
            rf_predictions = best_rf_model.predict(X_test)  # Predicciones de clase
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, rf_predictions)
            precision = precision_score(y_test, rf_predictions)
            recall = recall_score(y_test, rf_predictions)
            f1 = f1_score(y_test, rf_predictions)
            roc_auc = roc_auc_score(y_test, rf_predictions_prob)
            
            # Crear matriz de confusión
            cm = confusion_matrix(y_test, rf_predictions)
            # -----------------
            # Graficar la matriz de confusión (pero no mostrarla)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            # Guardar la figura sin mostrarla
            plt.savefig("confusion_matrix.png")
            conf_matrix_plot = plt.gcf()
            plt.close()
            
            # Graficar la importancia de las características sin mostrarla
            logger.info("Generando gráfico de importancia de características...")
            feature_importance_plot = plot_feature_importance(best_rf_model, X_train_df, y_train_df)
            logger.info("Gráfico de importancia guardado como 'feature_importance.png'")
            
            # Registrar los mejores parámetros y métricas en MLflow
            logger.info("Registrando parámetros y métricas en MLflow...")
            mlflow.log_param("best_rf_n_estimators", best_rf_model.n_estimators)
            mlflow.log_param("best_rf_max_depth", best_rf_model.max_depth)
            mlflow.log_param("best_rf_min_samples_split", best_rf_model.min_samples_split)
            mlflow.log_param("best_rf_class_weight", best_rf_model.class_weight)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Registrar las figuras en MLflow
            if corr_plot:
                mlflow.log_figure(corr_plot, artifact_file="correlation_with_target.png")
            if feature_importance_plot:
                mlflow.log_figure(feature_importance_plot, artifact_file="feature_importance.png")
            if conf_matrix_plot:
                mlflow.log_figure(conf_matrix_plot, artifact_file="confusion_matrix.png")
            
            # Obtener un ejemplo de entrada
            input_example = X_test[0:1]
            
            # Definir la ruta del artefacto
            artifact_path = "best_rf_model"
            
            # Inferir el esquema del ejemplo de entrada
            signature = mlflow.models.infer_signature(X_train, best_rf_model.predict_proba(X_train))
            
            # Registrar el mejor modelo Random Forest en el servidor MLflow
            logger.info("Registrando el modelo en MLflow...")
            mlflow.sklearn.log_model(
                sk_model=best_rf_model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name='stroke_prediction_model_dev',
                metadata={'model_data_version': 1}
            )
            
            # Obtener la URI del modelo registrado
            model_uri = mlflow.get_artifact_uri(artifact_path)
            
            # testing para comparar resultados en MLFlow
            logger.info("\nResultados del mejor modelo Random Forest:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"\nMejores parámetros Random Forest: {grid_search_rf.best_params_}")
            
        finally:
            # Finalizar manualmente el run para evitar el error de emoji Unicode
            # Usamos el mlflow.end_run() en lugar de depender del context manager
            mlflow.end_run(status="FINISHED")
            logger.info("MLflow run finalizado correctamente")
        
        # Probar el modelo
        logger.info("\nProbando el modelo cargado...")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Obtener un elemento aleatorio del conjunto de pruebas
        random_idx = random.randint(0, X_test.shape[0] - 1)
        input_example = X_test[random_idx]
        actual_value = y_test[random_idx]
        
        logger.info(f"Ejemplo de entrada seleccionado: {input_example}")
        logger.info(f"Valor real: {actual_value}")
        
        # Calcular la probabilidad de stroke
        stroke_probability = loaded_model.predict_proba(input_example.reshape(1, -1))[0, 1]
        predicted_class = loaded_model.predict(input_example.reshape(1, -1))[0]
        
        logger.info(f"Probabilidad de stroke: {stroke_probability:.4f}")
        logger.info(f"Clase predicha: {predicted_class}")
        
        # Registrar el modelo para producción
        logger.info("\nRegistrando el modelo para producción...")
        client = MlflowClient()
        
        model_name = "stroke_prediction_model_prod"
        desc = "Modelo de producción para la predicción de stroke"
        
        # Verificar si el modelo ya existe, si no, crearlo
        try:
            client.get_registered_model(name=model_name)
            logger.info(f"El modelo registrado {model_name} ya existe")
        except Exception:
            client.create_registered_model(name=model_name, description=desc)
            logger.info(f"Se ha creado el modelo registrado {model_name}")
        
        # Crear etiquetas para el modelo
        tags = best_rf_model.get_params()
        tags = {k: str(v) for k, v in tags.items()}  # Convertir todos los valores a string
        tags["model"] = type(best_rf_model).__name__
        tags["accuracy"] = str(accuracy)
        tags["roc_auc"] = str(roc_auc)
        
        # Extraer el run_id del model_uri
        uri_parts = model_uri.split("/")
        extracted_run_id = None
        for i, part in enumerate(uri_parts):
            if part == "runs":
                extracted_run_id = uri_parts[i+1]
                break
        
        if not extracted_run_id:
            extracted_run_id = run_id
            
        logger.info(f"Usando run_id: {extracted_run_id}")
        
        # Crear una nueva versión del modelo
        try:
            result = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=extracted_run_id,
                tags=tags
            )
            
            # Establecer la versión actual como "champion"
            client.set_registered_model_alias(model_name, "champion", result.version)
            logger.info(f"Se ha registrado la versión {result.version} del modelo como 'champion'")
        except Exception as e:
            logger.error(f"Error al crear versión del modelo: {str(e)}")
        
        logger.info("\nProceso completado con éxito!")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()