

## Contexto y Objetivo

### Visión General
- Nuestros clientes utilizan la aplicación de la empresa para comprar artículos en nuestra tienda de comestibles.
- Ocasionalmente, buscamos promocionar artículos específicos. Esto puede deberse a planes de discontinuación, proximidad de expiración o metas de expansión de cuota de mercado.

### Desafíos
- Se utilizan notificaciones push para impulsar las ventas e incentivar la participación del usuario con estos productos seleccionados.
- Sin embargo, un exceso de notificaciones push puede llevar a la insatisfacción del usuario y a la desinstalación de la aplicación, incurriendo en costos significativos.
- La tasa de apertura actual de notificaciones push es de aproximadamente el 5%.

### Objetivo
- Desarrollar un producto basado en un modelo predictivo.
- Este modelo identificará a los usuarios con mayor probabilidad de estar interesados en los artículos promocionados.
- Dirigir estas notificaciones push a estos usuarios para aumentar la efectividad y reducir la potencial molestia.

## Requisitos

### Segmentación de Usuarios
- Enfocarse en usuarios que compren el artículo promocionado junto con al menos otros 4 artículos (cesta mínima de 5 artículos).
- Esto se debe a los altos costos de envío para pedidos más pequeños que podrían superar el margen bruto.

### Funcionalidad del Sistema
- Los operadores de ventas deben poder seleccionar un artículo a través de un menú desplegable o barra de búsqueda.
- El sistema identificará el segmento de usuarios objetivo.
- Los operadores pueden activar notificaciones push personalizables para estos usuarios.

## Planificación

### Cronograma
- Esta herramienta es de alta prioridad, dado el dinamismo competitivo del mercado.
- Se espera un Concepto de Prueba (PoC) en una semana.
- El objetivo es estar en funcionamiento en 2 a 3 semanas.

## Impacto

### Resultados Esperados
- Módulo 3: TRD 2
- Objetivo de aumentar las ventas mensuales en un 2%.
- Meta de incrementar un 25% en las ventas de artículos seleccionados.
- Más detalles se pueden encontrar en el informe de análisis de empuje del departamento de ventas.





# Fase 1: Fase de Exploración

## Visión General

En esta fase, nos centraremos en construir el modelo predictivo con un claro entendimiento de nuestros datos. Nuestro enfoque involucra dos pasos clave:

### Filtrado de Datos
- **Objetivo**: Refinar el conjunto de datos para incluir solo aquellos pedidos con 5 artículos o más.
- **Razón**: Esto se alinea con nuestro requisito de enfocarnos en usuarios que realizan compras más grandes, optimizando costos de envío y margen bruto.

### Construcción y Evaluación del Modelo
- **Enfoque**: Utilizar modelos lineales por su simplicidad y rapidez, ya que apuntamos a un Concepto de Prueba (PoC) en un corto plazo de tiempo.
- **Proceso**:
  - Implementar una división de entrenamiento/validación/prueba para evaluar los modelos.
  - Enfocarse en modelos lineales debido a las limitaciones de tiempo y su eficiencia probada en escenarios similares.

### Resultado Esperado
- Un informe completo, cuaderno o documentación detallando:
  - El rendimiento y los hallazgos de los modelos implementados.
  - Ideas sobre lo que funcionó bien y lo que no, incluyendo análisis de por qué ciertos enfoques fueron más efectivos.
- Lo más importante, la selección de un modelo final para proceder a la Fase 2.

