# PlayNow

PlayNow recomienda qué videojuego de tu biblioteca abrir en una sesión concreta. La pregunta que busca resolver es: **«De los juegos que ya tengo disponibles, ¿qué debería jugar ahora?»**

Es un prototipo para el curso Introducción al diseño de sistemas recomendadores de la Maestría en Inteligencia Artificial Aplicada de Icesi. El núcleo usa contenido y similaridad, con apoyo del contexto de la sesión.

## Qué incluye

- Catálogo de 68 videojuegos con atributos de sesión, carga cognitiva, exigencia motora, narrativa, valencia y activación.
- Bibliotecas de 15 usuarios simulados, con 109 relaciones usuario-juego.
- 58 sesiones contextuales y 136 comentarios sintéticos.
- Motor compartido por un notebook y una demo en Streamlit.
- Informes de las tres partes del proyecto y diagrama de arquitectura.

## Datos reales y simulados

Los títulos de los videojuegos son reales. Sus atributos profundos son estimaciones de diseño para el prototipo, no mediciones obtenidas de jugadores.

Los usuarios, bibliotecas, historiales, señales de wearable y comentarios son simulados. Los comentarios se generan desde los mismos atributos del catálogo: muestran un posible pipeline de procesamiento de reseñas y no constituyen evidencia independiente.

Steam, PlayStation Network, Xbox, la agenda y el wearable no están conectados. La única interacción explícita prevista por sesión es **«¿Qué te provoca jugar hoy?»**, con cuatro intenciones: relajarme, algo ligero, historia y reto. Los controles adicionales de la demo representan las señales simuladas.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `playnow/engine.py` | Motor de recomendación y funciones auxiliares. |
| `data/` | CSV simulados y generador reproducible. |
| `notebooks/Prototipo_SR_Videojuegos.ipynb` | Explicación y ejecución del prototipo. |
| `demo/app_demo.py` | Demo interactiva en Streamlit. |
| `informe/` | Informes de Parte 1, Parte 2 y Parte 3, y arquitectura. |

## Instalación

Se recomienda Python 3.12. Desde la raíz del repositorio, crea un entorno virtual e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Generación de datos

Los CSV ya están incluidos. Para regenerarlos con la semilla fija 42:

```bash
python data/generador_dataset_profundo.py
```

Este comando reemplaza los cinco CSV de `data/` con los datos generados.

## Ejecución del notebook

```bash
jupyter notebook notebooks/Prototipo_SR_Videojuegos.ipynb
```

Selecciona el entorno en el que instalaste las dependencias y ejecuta las celdas en orden. El notebook compara la recomendación contextual con una línea base ordenada por horas jugadas.

## Ejecución de la demo

```bash
python -m streamlit run demo/app_demo.py
```

La demo permite elegir un perfil y explorar tres escenarios: una sesión corta para relajarse, una sesión larga de historia y una sesión de reto.

## Cómo recomienda

El motor parte de los juegos de la biblioteca del usuario. Compara el vector de la intención con los atributos de cada juego mediante similitud coseno y cercanía en el plano de valencia y activación de Russell. La lógica difusa combina minutos disponibles y fatiga para estimar la carga admisible. El ranking incorpora viabilidad, reenganche y, cuando hay evidencia, una señal de vecinos del mismo perfil ocupacional.

PCA y K-Means sirven como análisis auxiliar de familias de juegos. Los clústeres no determinan el orden de las recomendaciones.

## Limitaciones

- Los resultados ilustran el comportamiento del prototipo; no son una evaluación con jugadores reales.
- Los pesos del ranking son heurísticos de diseño, no parámetros aprendidos de datos.
- La fatiga es un proxy heurístico simulado. Solo el pulso usa una línea base personal; HRV y sueño usan referencias fijas.
- El juego refugio está en los perfiles, pero no interviene directamente en el ranking actual.
- Las integraciones reales y la caracterización con reseñas de jugadores quedan como trabajo futuro.
