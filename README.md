# PlayNow

PlayNow recomienda qué videojuego de tu biblioteca abrir en una sesión concreta. La pregunta que busca resolver es: **«De los juegos que ya tengo disponibles, ¿qué debería jugar ahora?»**

El núcleo del prototipo usa contenido y similaridad, con apoyo del contexto de la sesión.

## Contexto académico

- **Universidad:** Universidad Icesi.
- **Programa:** Maestría en Inteligencia Artificial Aplicada (MIAA).
- **Curso:** Introducción al diseño de sistemas recomendadores.
- **Periodo académico:** 2026-2.

## Equipo de trabajo

- Juan Camilo Macías.
- Jairo Andrés Valencia Gutiérrez.
- Ulises Caicedo.

## Qué incluye

- Catálogo de 68 videojuegos con atributos de sesión, carga cognitiva, exigencia motora, narrativa, valencia y activación.
- Bibliotecas de 15 usuarios simulados, con 109 relaciones usuario-juego.
- 58 sesiones contextuales y 136 comentarios sintéticos.
- Motor compartido por un notebook y una demo en Streamlit.

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
| `tests/` | Pruebas de límites de similitud y de los escenarios de referencia. |

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

La cercanía de Russell usa `1 - distancia / (2 * sqrt(2))`, acotada a `[0, 1]`. Después de combinarla con el coseno, el ajuste de reenganche se calcula como `clip(similitud_base - penalizacion, 0, 1)`. Un valor negativo de la penalización actúa como bono y el resultado sigue limitado a 1.

PCA y K-Means sirven como análisis auxiliar de familias de juegos. Los clústeres no determinan el orden de las recomendaciones.

## Pruebas

Con el entorno activado, ejecuta:

```bash
python -m unittest discover -s tests -v
```

Las pruebas comprueban los extremos del plano de Russell, los límites del reenganche, los Top 3 y puntajes de referencia, y las recomendaciones de los 15 perfiles en los tres contextos.

## Limitaciones

- Los resultados ilustran el comportamiento del prototipo; no son una evaluación con jugadores reales.
- Los pesos del ranking son heurísticos de diseño, no parámetros aprendidos de datos.
- La fatiga es un proxy heurístico simulado. Solo el pulso usa una línea base personal; HRV y sueño usan referencias fijas.
- El juego refugio está en los perfiles, pero no interviene directamente en el ranking actual.
- Las integraciones reales y la caracterización con reseñas de jugadores quedan como trabajo futuro.
