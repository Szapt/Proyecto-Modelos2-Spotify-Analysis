# Modelado y Predicción del Churn de Usuarios en Spotify mediante Técnicas de Aprendizaje Automático

**Modeling and Prediction of User Churn in Spotify Using Machine Learning Techniques**

---

## 📋 Descripción del Proyecto

Este proyecto presenta una evaluación exhaustiva de cinco algoritmos de aprendizaje automático para la predicción de abandono de usuarios (churn) en Spotify. Se implementan modelos de clasificación supervisada con técnicas de balanceo de clases y optimización de hiperparámetros, seguidos de un análisis detallado mediante reducción de dimensionalidad para diagnosticar limitaciones predictivas.

**Autores:** Santiago Zapata Coral & María de los Ángeles Agudelo Agudelo  
**Institución:** Universidad de Antioquia, Medellín, Colombia

---

## 🎯 Objetivos

1. Comparar el desempeño de múltiples algoritmos de aprendizaje automático en predicción de churn
2. Implementar técnicas de manejo de desbalance de clases (SMOTE)
3. Realizar optimización rigurosa de hiperparámetros mediante GridSearchCV
4. Analizar la estructura topológica del espacio de características mediante PCA, UMAP y LDA
5. Diagnosticar limitaciones fundamentales en el poder predictivo del dataset

---

## 📊 Dataset

- **Fuente:** [Kaggle - Spotify Churn Analysis Dataset](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)
- **Tamaño:** 8,000 usuarios
- **Variables:** 11 features (demográficas, suscripción, comportamiento de escucha)
- **Clases:** 74.1% activos vs. 25.9% churned (desbalance moderado)
- **Variables eliminadas:** `user_id`, `offline_listening` (multicolinealidad: r = -0.88)

### Variables incluidas:
- Numéricas: `skip_rate`, `listening_time`, `songs_played_per_day`, `ads_listened_per_week`, `age`
- Categóricas: `gender`, `country`, `subscription_type`, `device_type`
- Objetivo: `is_churned` (binaria)

---

## 🤖 Modelos Implementados

| Modelo | Tipo | AUC-ROC | Precisión | Exhaustividad | F1-score |
|--------|------|---------|-----------|---------------|----------|
| Regresión Logística | Paramétrico | 0.52 | 0.63 | 0.51 | 0.49 |
| Random Forest | Ensamble | 0.51 | 0.64 | 0.50 | 0.48 |
| SVM | No-paramétrico | 0.50 | 0.62 | 0.49 | 0.47 |
| MLP (Red Neuronal) | Deep Learning | 0.51 | 0.63 | 0.50 | 0.53 |
| KNN | Basado en distancia | **0.54** | 0.64 | 0.51 | 0.54 |

**Mejor modelo:** KNN con AUC-ROC = 0.54

---

## 🔧 Técnicas Aplicadas

### Preprocesamiento
- **Estandarización:** StandardScaler para variables numéricas
- **Codificación:** OneHotEncoding para variables categóricas
- **Balanceo:** SMOTE (Synthetic Minority Over-sampling Technique)

### Optimización
- **Búsqueda de hiperparámetros:** GridSearchCV
- **Validación:** 5-fold Cross-Validation
- **Métrica de optimización:** AUC-ROC Score

### Análisis de Dimensionalidad
1. **PCA (Análisis de Componentes Principales)**
   - 8 componentes requeridos para 80% de variabilidad
   - No mejoró AUC-ROC

2. **UMAP (Manifold Learning)**
   - Reveló bimodalidad por tipo de suscripción
   - Distribución aleatoria de clases sin separabilidad

3. **LDA (Linear Discriminant Analysis)**
   - Funciones de densidad idénticas y concéntricas
   - Confirmó ortogonalidad de variable objetivo

---

## 📁 Estructura del Repositorio

```
Proyecto-Modelos2-Spotify-Analysis/
│
├── 01_Regresion_Logistica.ipynb          # Implementación de Regresión Logística
├── 02_KNN.ipynb                          # K-Nearest Neighbors + PCA, UMAP, LDA
├── 03_Random_Forest__1_.ipynb            # Random Forest + análisis dimensional
├── 04_SVM__1_.ipynb                      # Support Vector Machine
├── 05_Red_Neuronal.ipynb                 # Multilayer Perceptron (MLP)
│
├── Comparacion_ROC_Modelos_Base.ipynb    # Gráfica comparativa de curvas ROC
│
├── Informe Final Modelos II              # Documento desarrollado en Latex
│
└── README.md                             # Este archivo
```

---

## 🚀 Cómo Ejecutar

### Requisitos
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### Ejecución
1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Szapt/Proyecto-Modelos2-Spotify-Analysis.git
   cd Proyecto-Modelos2-Spotify-Analysis
   ```

2. **Descargar el dataset:** Obtener `spotify_dataset.csv` de [Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)

3. **Ejecutar notebooks:**
   ```bash
   jupyter notebook 01_Regresion_Logistica.ipynb
   ```

4. **Generar reporte comparativo:**
   ```bash
   jupyter notebook Comparacion_ROC_Modelos_Base.ipynb
   ```

---

## 📈 Resultados Principales

### Hallazgo Crítico
**Todos los modelos convergieron hacia AUC-ROC ≈ 0.50 (desempeño de azar)**

- KNN: mejor desempeño con AUC-ROC = 0.54
- Diferencia marginal entre algoritmos (0.50 - 0.54)
- Aplicación de SMOTE no mejoró discriminabilidad
- GridSearchCV + 5-fold CV no compensó limitaciones del dataset

### Diagnóstico mediante Reducción de Dimensionalidad

**PCA:** 8 componentes explican 80% de varianza → sin mejora en AUC-ROC

**UMAP:** Revela dos clústers (Free vs. Premium) pero usuarios churned/no-churned distribuidos aleatoriamente

**LDA:** Funciones de densidad idénticas entre clases → imposible crear frontera de separación

### Conclusión
El dataset exhibe características de datos sintéticos o insuficientemente capturados. **La variable de churn es ortogonal al espacio de características disponibles**, haciendo la predicción fundamentalmente intractable con estos datos.


---

## 👥 Autores

**Santiago Zapata Coral**  
📧 santiagod.zapata@udea.edu.co

**María de los Ángeles Agudelo Agudelo**  
📧 maria.aagudelo@udea.edu.co

Facultad de Ingeniería, Universidad de Antioquia  
Medellín, Colombia

---

## 🎓 Nota Académica

Este proyecto demuestra que **excelente metodología ≠ excelentes resultados** cuando los datos fundamentales carecen de poder discriminativo. Es una lección valiosa en ciencia de datos: la calidad del dataset es más importante que la sofisticación del algoritmo.
