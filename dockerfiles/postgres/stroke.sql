-- Primero crear las bases de datos
CREATE DATABASE stroke;

-- Conectar explícitamente a la base de datos stroke
\c stroke;

-- Eliminar la tabla si existe (para evitar el aviso)
DROP TABLE IF EXISTS stroke_data;

-- Crear la tabla
CREATE TABLE stroke_data (
    id INT,
    gender VARCHAR(10),
    age FLOAT,
    hypertension INT,
    heart_disease INT,
    ever_married VARCHAR(5),
    work_type VARCHAR(20),
    residence_type VARCHAR(10),
    avg_glucose_level FLOAT,
    bmi FLOAT,
    smoking_status VARCHAR(20),
    stroke INT
);

-- Cargar los datos stroke_data
\c stroke;
\copy stroke_data FROM '/docker-entrypoint-initdb.d/healthcare-dataset-stroke-data.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', NULL 'N/A');