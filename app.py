import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Proyecto de IA", layout="wide")

# 1. MENÚ LATERAL
st.sidebar.title("Navegación")
opcion = st.sidebar.selectbox(
    "Elige el algoritmo:",
    ["Inicio", "Regresión Lineal", "K-Nearest Neighbors", "Naive Bayes"]
)

# 2. LÓGICA DE NAVEGACIÓN
if opcion == "Inicio":
    st.title(" Proyecto de Inteligencia Artificial")
    st.write("Selecciona un algoritmo en el menú de la izquierda para comenzar.")
    st.info("<-------------------------------------------------------------------")

elif opcion == "Regresión Lineal":
    st.title("Regresión Lineal Simple")
    st.markdown("Modelo: **y = mx + b**")

    # Opción para elegir cómo ingresar datos
    tipo_entrada = st.radio("¿Cómo quieres ingresar los datos?", ["Manual", "Archivo CSV"])

    datos = None # Aquí guardaremos los datos

    if tipo_entrada == "Manual":
        st.subheader("Ingreso Manual")
        # Pedimos X e Y separados por comas
        x_input = st.text_input("Ingresa los valores de X (separados por coma):", "1, 2, 3, 4, 5")
        y_input = st.text_input("Ingresa los valores de Y (separados por coma):", "2, 4, 5, 4, 5")
        
        if x_input and y_input:
            try:
                # Convertimos texto a listas de números
                x_lista = [float(i) for i in x_input.split(',')]
                y_lista = [float(i) for i in y_input.split(',')]
                
                # Creamos un DataFrame (tabla) simple
                datos = pd.DataFrame({'X': x_lista, 'Y': y_lista})
            except ValueError:
                st.error("Asegúrate de ingresar solo números separados por comas.")

    elif tipo_entrada == "Archivo CSV":
        st.subheader("Carga de Archivo")
        archivo = st.file_uploader("Sube tu archivo CSV (debe tener columnas llamadas X e Y)", type=["csv"])
        if archivo is not None:
            datos = pd.read_csv(archivo)

    # Mostrar los datos si ya existen
    if datos is not None:
        st.write("Vista previa de los datos cargados:")
        st.dataframe(datos)
        
        # Verificamos que tengamos suficientes datos
        if len(datos) > 1:
            st.success(f"¡Cargados {len(datos)} datos correctamente!")

            st.markdown("---") # Una línea separadora visual

        # --- SECCIÓN: CÁLCULOS MATEMÁTICOS MANUALES ---
        st.subheader("Parámetros del Modelo (Calculados Manualmente)")
        
        # 1. Extraemos los valores a arreglos de Numpy para poder operar vectores
        X = datos['X'].values
        Y = datos['Y'].values
        
        # 2. Calcular promedios (X barra y Y barra)
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        
        # 3. Calcular la pendiente (m) usando la fórmula de Mínimos Cuadrados
        # Fórmula: m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        numerador = np.sum((X - x_mean) * (Y - y_mean))
        denominador = np.sum((X - x_mean) ** 2)
        
        m = numerador / denominador
        
        # 4. Calcular el intercepto (b)
        # Fórmula: b = y_mean - (m * x_mean)
        b = y_mean - (m * x_mean)
        
        # Mostrar la ecuación resultante en formato matemático bonito
        st.latex(f"y = {m:.2f}x + {b:.2f}")
        
        # --- SECCIÓN: CÁLCULO DE ERROR ---
        # Calculamos las predicciones para todos los puntos que tenemos
        Y_pred = m * X + b
        
        mse = np.mean((Y - Y_pred) ** 2)
        st.write(f"**Error Cuadrático Medio (MSE):** {mse:.4f}")
        
        # --- SECCIÓN: VISUALIZACIÓN ---
        st.subheader("Gráfico de Regresión")
        
        # Creamos el gráfico con Matplotlib y Seaborn
        fig, ax = plt.subplots()
        sns.scatterplot(x=X, y=Y, ax=ax, label='Datos Reales', color='blue', s=100)
        sns.lineplot(x=X, y=Y_pred, ax=ax, label='Línea de Tendencia', color='red')
        ax.set_title("Regresión Lineal Simple")
        ax.set_xlabel("Variable X")
        ax.set_ylabel("Variable Y")
        st.pyplot(fig)
        
        # --- SECCIÓN: PREDICCIÓN ---
        st.subheader("Hacer una Predicción")
        st.write("Prueba el modelo con un valor nuevo:")
        
        val_x = st.number_input("Ingresa valor de X:", value=0.0)
        
        if st.button("Predecir Y"):
            pred_y = m * val_x + b
            st.success(f"Para X = {val_x}, la predicción es Y = {pred_y:.2f}")
        else:
            st.warning("Necesitas al menos 2 puntos de datos.")

elif opcion == "K-Nearest Neighbors":
    st.title("Clasificación K-Nearest Neighbors (K-NN)")

    # 1. Configuración de Datos
    st.subheader("1. Cargar Datos de Entrenamiento")
    
    # IGUAL QUE REGRESIÓN: Manual o Archivo CSV
    tipo_entrada = st.radio("¿Cómo quieres ingresar los datos?", ["Manual", "Archivo CSV"])

    df_knn = None

    if tipo_entrada == "Manual":
        st.subheader("Ingreso Manual")
        st.write("Ingresa los datos separados por comas. (Asegúrate de poner la misma cantidad de elementos en las tres cajas).")
        
        # Cajas de texto con los datos de ejemplo pre-llenados para no tener que escribir todo de cero
        x1_input = st.text_input("Ingresa los valores de X1 (separados por coma):", "1.0, 1.5, 2.0, 1.2, 5.0, 5.5, 6.0, 5.2, 2.5, 6.5")
        x2_input = st.text_input("Ingresa los valores de X2 (separados por coma):", "1.0, 1.5, 1.0, 2.0, 5.0, 5.5, 4.5, 6.0, 2.0, 5.0")
        clase_input = st.text_input("Ingresa las Clases correspondientes (separadas por coma, ej. 0, 1, 0...):", "0, 0, 0, 0, 1, 1, 1, 1, 0, 1")
        
        if x1_input and x2_input and clase_input:
            try:
                # Convertimos los textos en listas limpiando los espacios
                x1_lista = [float(i.strip()) for i in x1_input.split(',')]
                x2_lista = [float(i.strip()) for i in x2_input.split(',')]
                # La clase puede ser texto (A, B) o números (0, 1)
                clase_lista = [int(i.strip()) if i.strip().isdigit() else i.strip() for i in clase_input.split(',')]
                
                # Verificamos que el usuario no se haya equivocado y olvidado un número
                if len(x1_lista) == len(x2_lista) == len(clase_lista):
                    df_knn = pd.DataFrame({'X1': x1_lista, 'X2': x2_lista, 'Clase': clase_lista})
                else:
                    st.error(f"Error: Las listas tienen diferente tamaño. X1 tiene {len(x1_lista)}, X2 tiene {len(x2_lista)} y Clases tiene {len(clase_lista)}.")
            except ValueError:
                st.error("Error: Asegúrate de ingresar solo números separados por comas en las cajas de X1 y X2.")

    elif tipo_entrada == "Archivo CSV":
        st.subheader("Carga de Archivo")
        st.warning("El archivo CSV debe tener las columnas llamadas exactamente: X1, X2 y Clase")
        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo is not None:
            df_knn = pd.read_csv(archivo)

    # Si hay datos (ya sea manuales o por CSV), mostramos la tabla editable
    if df_knn is not None:
        st.write("Vista previa de los datos (¡Puedes editar las celdas directamente!):")
        df_knn = st.data_editor(df_knn)

        st.markdown("---")
        st.subheader("2. Configurar el Algoritmo")
        # ... AQUÍ CONTINÚA TU CÓDIGO ORIGINAL (col1, col2, col3, etc.) ...
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Seleccionar K (número de vecinos)
            k = st.slider("Valor de K (Vecinos):", min_value=1, max_value=len(df_knn), value=3)
        with col2:
            # Ingresar coordenadas del NUEVO punto a clasificar
            nuevo_x1 = st.number_input("Coordenada X1 del nuevo punto:", value=3.5)
        with col3:
            nuevo_x2 = st.number_input("Coordenada X2 del nuevo punto:", value=3.5)

        if st.button("Clasificar Nuevo Punto"):
            # 1. Convertir datos a arreglos numpy
            puntos = df_knn[['X1', 'X2']].values
            clases = df_knn['Clase'].values
            
            # Nuevo punto
            nuevo_punto = np.array([nuevo_x1, nuevo_x2])
            
            # 2. Calcular Distancia Euclidiana manualmente
            # Distancia = raiz((x2-x1)^2 + (y2-y1)^2)
            # Calculamos la distancia desde el nuevo punto a TODOS los puntos existentes
            distancias = np.sqrt(np.sum((puntos - nuevo_punto)**2, axis=1))
            
            # 3. Encontrar los K vecinos más cercanos
            # argsort nos da los ÍNDICES de los valores ordenados de menor a mayor
            indices_ordenados = np.argsort(distancias)
            indices_vecinos = indices_ordenados[:k] # Tomamos los primeros K
            
            # Obtenemos las clases de esos vecinos
            clases_vecinos = clases[indices_vecinos]
            
            # 4. Votación (Moda)
            # Contamos cuál clase aparece más veces
            clases_unicas, conteos = np.unique(clases_vecinos, return_counts=True)
            clase_ganadora = clases_unicas[np.argmax(conteos)] 
            
            st.success(f"El nuevo punto pertenece a la **Clase {clase_ganadora}**")
            st.write("Vecinos más cercanos encontrados (Índice, Distancia, Clase):")
            for i in range(k):
                idx = indices_vecinos[i]
                st.text(f"Vecino {i+1}: Distancia={distancias[idx]:.4f} -> Clase {clases[idx]}")

            # --- VISUALIZACIÓN ---
            st.subheader("Visualización Gráfica")
            fig, ax = plt.subplots()
            
            # Graficar puntos existentes (Clase 0 y Clase 1)
            sns.scatterplot(data=df_knn, x='X1', y='X2', hue='Clase', palette='deep', s=100, ax=ax)
            
            # Graficar el NUEVO punto (como una estrella grande)
            ax.scatter(nuevo_x1, nuevo_x2, color='red', marker='*', s=300, label='Nuevo Punto')
            
            # Dibujar círculos alrededor de los vecinos seleccionados
            vecinos_cercanos = puntos[indices_vecinos]
            ax.scatter(vecinos_cercanos[:, 0], vecinos_cercanos[:, 1], facecolors='none', edgecolors='black', s=200, linewidths=2, label='Vecinos elegidos')

            ax.legend()
            ax.set_title(f"Clasificación K-NN (K={k})")
            st.pyplot(fig)
    

elif opcion == "Naive Bayes":
    st.title("Clasificador Naive Bayes (Probabilístico)")

    # 1. Cargar Datos
    st.subheader("1. Cargar Datos de Entrenamiento")
    
    tipo_entrada = st.radio("Fuente de datos:", ["Dataset de Ejemplo", "Archivo CSV"])
    
    df_bayes = None

    if tipo_entrada == "Dataset de Ejemplo":
        # Usamos el dataset original de clima
        data_bayes = {
            'Clima': ['Sol', 'Sol', 'Nublado', 'Lluvia', 'Lluvia', 'Lluvia', 'Nublado', 'Sol', 'Sol', 'Lluvia', 'Sol', 'Nublado', 'Nublado', 'Lluvia'],
            'Temperatura': ['Calor', 'Calor', 'Calor', 'Templado', 'Frio', 'Frio', 'Frio', 'Templado', 'Frio', 'Templado', 'Templado', 'Templado', 'Calor', 'Templado'],
            'Jugar': ['No', 'No', 'Si', 'Si', 'Si', 'No', 'Si', 'No', 'Si', 'Si', 'Si', 'Si', 'Si', 'No']
        }
        df_bayes = pd.DataFrame(data_bayes)
        st.info("Se cargó el dataset de 'Jugar Tenis' automáticamente.")
        
    elif tipo_entrada == "Archivo CSV":
        st.warning("Nota: Sube un archivo donde la ÚLTIMA columna sea la clase a predecir.")
        archivo = st.file_uploader("Sube tu archivo CSV:", type=["csv"])
        if archivo is not None:
            df_bayes = pd.read_csv(archivo)

    # --- LÓGICA DINÁMICA CON DISCRETIZACIÓN ---
    # Fíjate cómo este "if" está alineado a la izquierda con los "if" de arriba
    if df_bayes is not None:
        st.write("Dataset de Entrenamiento Original (¡Puedes editar las celdas directamente!):")
        df_bayes = st.data_editor(df_bayes)

        # NUEVO: DISCRETIZACIÓN AUTOMÁTICA DE NÚMEROS
        df_transformado = df_bayes.copy()
        columnas_features = df_transformado.columns[:-1] # Todas menos la última
        columna_objetivo = df_transformado.columns[-1]   # La última columna
        columnas_numericas = []

        for col in columnas_features:
            # Si la columna es de tipo numérico (entero o decimal)
            if df_transformado[col].dtype in ['int64', 'float64']:
                columnas_numericas.append(col)
                # Convertimos los números en 3 rangos: Bajo, Medio, Alto
                df_transformado[col] = pd.cut(df_transformado[col], bins=3, labels=["Bajo", "Medio", "Alto"])

        if len(columnas_numericas) > 0:
            st.success(f"¡Magia de Datos! Se detectaron columnas numéricas ({', '.join(columnas_numericas)}). Fueron convertidas en categorías ('Bajo', 'Medio', 'Alto') para que el modelo probabilístico funcione correctamente.")
            st.write("Dataset Transformado:")
            st.dataframe(df_transformado.head())

        st.markdown("---")
        st.subheader("2. Tablas de Frecuencia y Probabilidad")

        # 1. Probabilidad Previa P(Clase)
        total_datos = len(df_transformado)
        conteos_clase = df_transformado[columna_objetivo].value_counts()
        prob_clase = conteos_clase / total_datos
        
        st.write(f"**Probabilidades Previas de '{columna_objetivo}':**")
        st.write(prob_clase)

        # 2. Calcular Verosimilitud dinámicamente con los datos transformados
        probs_condicionales = {}
        
        for columna in columnas_features:
            tabla = pd.crosstab(df_transformado[columna], df_transformado[columna_objetivo])
            tabla_prob = tabla.div(conteos_clase, axis=1)
            probs_condicionales[columna] = tabla_prob
            
            st.write(f"**Probabilidad P({columna} | {columna_objetivo}):**")
            st.dataframe(tabla_prob)

        st.markdown("---")
        st.subheader("3. Realizar Predicción")
        
        # Generar los cuadros de selección
        valores_usuario = {}
        for col in columnas_features:
            # Ahora los cuadros de selección mostrarán "Bajo", "Medio", "Alto" para números
            valores_usuario[col] = st.selectbox(f"Selecciona {col}:", df_transformado[col].unique())
        
        clases_posibles = df_transformado[columna_objetivo].unique()
        
        if st.button("Calcular Probabilidad"):
            resultados = {}
            st.write("### Paso a paso del Teorema de Bayes:")
            
            for clase in clases_posibles:
                p_final = prob_clase[clase]
                st.write(f"**Analizando Clase '{clase}':**")
                st.write(f"   * P({clase}) = {p_final:.4f}")
                
                for col in columnas_features:
                    val_seleccionado = valores_usuario[col]
                    try:
                        p_feature = probs_condicionales[col].loc[val_seleccionado, clase]
                    except KeyError:
                        p_feature = 0
                    
                    p_final *= p_feature
                    st.write(f"   * P({col}={val_seleccionado} | {clase}) = {p_feature:.4f}")
                
                resultados[clase] = p_final
                st.write(f"   **-> Probabilidad Posterior: {p_final:.4f}**")
                st.write("---")
                
            clase_ganadora = max(resultados, key=resultados.get)
            st.success(f"Predicción Final para '{columna_objetivo}': **{clase_ganadora}**")

        # --- CÁLCULO DE PRECISIÓN ---
        st.markdown("---")
        st.subheader("4. Validación del Modelo")
        
        if st.button("Calcular Precisión (Accuracy)"):
            aciertos = 0
            for i, row in df_transformado.iterrows():
                clase_real = row[columna_objetivo]
                
                probs = {}
                for clase in clases_posibles:
                    p = prob_clase[clase]
                    for col in columnas_features:
                        val = row[col]
                        try:
                            p *= probs_condicionales[col].loc[val, clase]
                        except KeyError:
                            p = 0
                    probs[clase] = p
                
                prediccion = max(probs, key=probs.get)
                if prediccion == clase_real:
                    aciertos += 1
            
            accuracy = (aciertos / total_datos) * 100
            st.info(f"El modelo clasificó correctamente {aciertos} de {total_datos} casos.")
            st.metric(label="Precisión del Modelo (Accuracy)", value=f"{accuracy:.2f}%")