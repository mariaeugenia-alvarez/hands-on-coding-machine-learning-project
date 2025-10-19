# Proyecto Machine Learning: Predicción de Precios Airbnb Madrid

Este proyecto implementa un pipeline completo de machine learning para predecir precios de Airbnb en Madrid, aplicando técnicas de análisis exploratorio, ingeniería de características, selección de variables y evaluación de múltiples algoritmos de regresión.

## Descripción del Proyecto

### Objetivo
Desarrollar un modelo predictivo para estimar precios por noche de propiedades Airbnb en Madrid utilizando metodologías y mejores prácticas de ciencia de datos.

### Contexto de los Datos
- Dataset inicial: propiedades Airbnb globales
- Filtrado geográfico: Comunidad de Madrid
- Rango de precios analizado: hasta 300€/noche (99% de las muestras)
- División: conjunto de entrenamiento y test separados

## Metodología Implementada

### 1. Análisis Exploratorio de Datos (EDA)
- Filtrado geográfico para centrarse en Madrid
- Análisis de distribuciones y detección de outliers
- Identificación de valores nulos y patrones faltantes
- Visualización de correlaciones entre variables

### 2. Ingeniería de Características

#### Transformaciones Aplicadas:
- **Transformación logarítmica**: Price, Security Deposit, Cleaning Fee, Extra People, Reviews per Month
- **Imputación de valores nulos**: 
  - Moda para variables numéricas (Host Listings Count, Bathrooms, Bedrooms, Beds)
  - Cero para variables donde null indica ausencia (Security Deposit, Cleaning Fee, Extra People)
- **Codificación categórica**: Room Type, Bed Type mediante mean encoding
- **Creación de variables derivadas**: Days_Since_Last_Review (dias desde la última reseña), Booking_Rate (tasa de ocupación), Host_Since_Days (días desde que el host se registró), Avg_Review_Scores (promedio de puntuaciones de reseñas)

#### Selección de Variables:
- **Eliminadas por alta correlación**: Host Total Listings Count, Calculated Host Listings Count, Availability 30, Availability 60, Availability 90
- **Eliminadas por exceso de nulos**: Square Feet (>90%), Weekly/Monthly Price (>75%)
- **Métodos de selección**: Matriz de correlación, F-test y Mutual Information para ranking de importancia

### 3. Modelos Implementados

#### 3.1 Regresión LASSO
- **Optimización**: GridSearchCV con validación cruzada 5-fold
- **Mejor α**: 10^-3
- **Rendimiento**: RMSE = 0.381 (factor de error 1.46)
- **Variables más importantes**: Room Type (Entire home/apt), Accommodates, Bedrooms, Avg Review Scores

#### 3.2 Random Forest
- **Configuración**: 200 árboles, max_depth optimizado por CV
- **Prevención overfitting**: Reducción de profundidad respecto al óptimo (max_depth=10)
- **Métricas**: R² = 0.79 en test

#### 3.3 Gradient Boosting
- **Parámetros óptimos**: learning_rate=0.07, n_estimators=1000, max_depth=3
- **Rendimiento**: RMSE = 0.286 (factor de error 1.33) 
- **Mejora**: 25% respecto a LASSO

#### 3.4 XGBoost
- **Optimización**: RandomizedSearchCV
- **Configuración**: Búsqueda sobre learning_rate, n_estimators, max_depth
- **Validación**: Múltiples iteraciones para evitar overfitting

### 4. Evaluación y Validación
- **Métricas principales**: RMSE, R², MSE
- **Validación cruzada**: 3-fold y 5-fold según el modelo
- **Interpretabilidad**: Análisis de importancia de características
- **Diagnóstico**: Verificación de overfitting comparando train vs test

## Resultados Principales

### Rendimiento de Modelos (RMSE en test):
1. **XGBoost**: RMSE = 0.280 Mejor rendimiento general
2. **Gradient Boosting**: RMSE = 0.286 (factor error 1.33)  
3. **Random Forest**: R² = 0.79
4. **LASSO**: RMSE = 0.381 (factor error 1.46)

### Variables Más Predictivas:
- **Tipo de alojamiento** (Room Type:(Entire home/apt)): Mayor impacto
- **Capacidad** (Accommodates, Bedrooms): Relación directa con precio
- **Ubicación** (Latitude): Norte-Sur más relevante
- **Servicios adicionales** (Cleaning Fee): Factor importante
- **Calidad** (Avg Review Scores): Media de puntuación
- **Características físicas**: Bathrooms, Guest Included

### Interpretación del Error (XGBoost - Mejor Modelo):
Para una propiedad de 100€/noche, el modelo XGBoost predice en un rango de 76-132€ el 68% de las veces (±1 desviación estándar). El RMSE de 0.28 se traduce en un factor de error de 1.32 (e^0.281).

## Estructura del Proyecto

```
├── ml-airbnb.ipynb                           # Notebook principal con análisis completo
├── airbnb-listings-extract.csv              # Dataset original completo
├── airbnb-listings-extract-train.csv        # Conjunto de entrenamiento
├── airbnb-listings-extract-test.csv         # Conjunto de prueba
├── README.md                                 # Documentación del proyecto
└── your_report.html                          # Reporte de análisis
```

## Tecnologías Utilizadas

### Librerías de Análisis:
- **pandas, numpy**: Manipulación y análisis de datos
- **matplotlib, seaborn**: Visualización de datos
- **ydata-profiling**: Análisis exploratorio automatizado

### Machine Learning:
- **scikit-learn**: Modelos de regresión, validación cruzada, métricas
- **xgboost**: Algoritmo de gradient boosting avanzado
- **Modelos**: LASSO, Random Forest, Gradient Boosting, XGBoost

### Técnicas Aplicadas:
- Validación cruzada (k-fold)
- Búsqueda de hiperparámetros (Grid/RandomizedSearch)
- Regularización (L1 en LASSO)
- Ensemble methods (bagging y boosting)

## Conclusiones Técnicas

1. **Los modelos de ensemble superan significativamente** a la regresión lineal regularizada porque captan realciones mucho más complejas entre variables.
2. **Las variables categóricas** (tipo de alojamiento) tienen el mayor poder predictivo
3. **La ubicación geográfica** es relevante, especialmente la latitud (Norte-Sur)
4. **La capacidad del alojamiento** mantiene una relación directa con el precio
5. **Servicios adicionales** (Cleaning Fee): Factor importante en la determinación del precio
6. **Características físicas**: Beds, Bathrooms, Guest Included influyen moderadamente en las predicciones
7. **El modelo final generaliza bien** sin evidencia de overfitting significativo

## Aplicabilidad

El modelo desarrollado puede utilizarse para:
- Estimación automática de precios para nuevas propiedades
- Análisis de factores que influyen en la valoración
- Optimización de estrategias de pricing para anfitriones
- Benchmarking de precios en el mercado madrileño

## Siguientes Pasos

### Mejoras del Modelo:
- **Incorporar combinaciones de caracteristicas importantes**: Relaciones entre bedrooms, bathrooms, beds
- **Expandir características geográficas y codificar**: Añadir datos como zipcode o neighborhood
- **Análisis de amenities**: Procesamiento de texto para extraer características específicas de las comodidades, puede ser una buena caracterisitca de estudio
- **Ranking selección caracteristicas**: dentro de un bucle con validación cruzada. 
- **Reajustar hiperparámetros de modelos en cada iteración** 
