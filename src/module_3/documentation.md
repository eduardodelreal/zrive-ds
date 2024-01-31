# Índice del Archivo

## Clasificadores

### KNN
- Regiones de decisión no lineales.
- Selección de los k puntos más cercanos (por distancia).
- Clasificación basada en la mayoría de los k vecinos.
- Ponderación: uniforme o por distancia.

### Árboles de Decisión (Decision Trees)
- Partición del espacio de decisión en regiones no lineales.
- Parámetros clave:
  - Profundidad máxima (Max Depth).

### Random Forest
- Conjunto de árboles de decisión con datos y características variadas.
- Técnicas utilizadas:
  - Bagging para diversificar los datos de entrenamiento.
  - Selección aleatoria de características basada en la raíz cuadrada del número total de características.

## Clasificadores Lineales

### Regresor Logístico
- Aplicable para clasificación binaria.
- Usa la función sigmoide para obtener valores entre 0 y 1.
- Enfoque en clasificación, no en regresión.
- Ajuste iterativo de pesos (weights).

### Análisis Discriminante Lineal (LDA)
- Clasificación multiclase.
- Reducción de dimensionalidad y generación de nuevas características.
- Creación de decisores lineales.

## Curva ROC y Área Bajo la Curva (AUC)
- Evaluación del rendimiento de clasificadores binarios.
- Interpretación de valores AUC.

## Hiperparámetros

### Parámetros Paramétricos
- LDA:
  - `n_components`: Número de dimensiones.
- Regresor Logístico (LR):
  - Penalizaciones: L1, L2, Ninguna.
  - `C`: Inversa de la fuerza de regularización.

### Parámetros No Paramétricos
- KNN:
  - `n_neighbors`: Número de vecinos.
  - `weights`: Ponderación uniforme o por distancia.
- Árboles de Decisión y Random Forest:
  - `max_depth`: Profundidad máxima del árbol.
  - `n_estimators` (RF): Número de árboles.

## Validación Cruzada
- Uso para la validación de hiperparámetros.
- Aplicación en datasets pequeños.
- Procedimiento detallado con enfoque en la asignación y evaluación de folds.

## Matriz de Confusión
- Estructura y uso.
- Interpretación de resultados.

## Fases del Aprendizaje Automático
- `fit`: Entrenamiento de modelos.
- `predict`: Generación de etiquetas predichas.
- `score`: Cálculo de precisión (accuracy).
- `predict_proba`: Probabilidades de clasificación (específico para LR).


