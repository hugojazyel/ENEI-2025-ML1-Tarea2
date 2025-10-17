# Tarea 2: Regresión Logística y Extensiones Multiclase

**Fecha:** Jueves, 16 de octubre de 2025

**Integrantes:** Calle Ontaneda Hugo Jazyel, Chero Villegas Leidy Fabiola, Cueva Mendoza Jherson Aldair.

La regresión logística constituye uno de los pilares fundamentales de la modelación estadística y el aprendizaje supervisado, al permitir estimar probabilidades de pertenencia a una clase mediante una función de enlace logit. No obstante, su formulación se adapta a distintos escenarios de clasificación entre ellos el modelo binario, adecuado para problemas de dos categorías; el enfoque One-vs-All (OvA), que extiende la lógica binaria a entornos multiclase mediante la construcción de clasificadores independientes; y el modelo multinomial o Softmax, que optimiza de manera conjunta las probabilidades para todas las clases. Si bien las tres formulaciones comparten la base de la log-verosimilitud, difieren en la estructura del gradiente, la estabilidad numérica durante el entrenamiento y el tipo de coherencia que imponen sobre las predicciones.

En la forma binaria, la regresión logística modela la probabilidad de la clase positiva usando la función sigmoide $\sigma(z) = \frac{1}{1 + exp(-z)}$, donde $z = \theta^\top x$. La log-verosimilitud se define como

$$
\ell(\theta) = \sum_{i=1}^{n} \Big[ y^{(i)} \log \sigma(\theta^\top x^{(i)}) + (1 - y^{(i)}) \log (1 - \sigma(\theta^\top x^{(i)})) \Big],
$$

y su gradiente,

$$
\frac{\partial \ell(\theta)}{\partial \theta} = \sum_{i=1}^{n} (y^{(i)} - \sigma(\theta^\top x^{(i)})) x^{(i)},
$$

refleja directamente el error entre la etiqueta observada y la probabilidad predicha. Este gradiente es simple, lineal respecto al error, y garantiza convexidad en la función de pérdida. En la aplicación desarrollada con el conjunto de datos *Heart Disease* del repositorio UCI, el entrenamiento mediante descenso de gradiente desde cero alcanzó una convergencia estable con tasas de aprendizaje de 0.01 y 0.1, reproduciendo los mismos resultados que el solver `lbfgs` de *scikit-learn*. En este marco, la coincidencia exacta en la matriz de confusión y las métricas de desempeño confirmó la validez analítica del gradiente implementado y la estabilidad de la función logística bajo variables estandarizadas.

En tanto, el enfoque OvA amplía esta lógica al contexto multiclase entrenando $K$ clasificadores binarios independientes, uno por cada clase, donde cada modelo aprende a distinguir su propia clase frente a todas las demás. Para la clase $l$, la log-verosimilitud mantiene una forma análoga a la binaria, y su gradiente se expresa como

$$
\frac{\partial \ell(\theta_l)}{\partial \theta_l} = \sum_{i=1}^{n} \Big( 1\{y^{(i)} = l\} - \sigma(\theta_l^\top x^{(i)}) \Big) x^{(i)}.
$$

Dado que los clasificadores se ajustan por separado, los gradientes no presentan acoplamiento entre clases, lo que reduce la complejidad computacional pero puede generar inconsistencias probabilísticas (por ejemplo, la suma de probabilidades por observación no necesariamente es igual a uno). En particular, en la práctica aplicada con el conjunto de datos *Wine*, este modelo desde cero alcanzó una exactitud de 0.9815, idéntica a la lograda con `LogisticRegression(multi_class="ovr")` de *scikit-learn*. Además, la similitud coseno entre los vectores de coeficientes de ambas implementaciones superó 0.97 en las tres clases, demostrando una alta correspondencia direccional y exhibiendo la corrección del descenso por gradiente manual.

Por su parte, la formulación multinomial, también conocida como regresión logística Softmax, optimiza un modelo conjunto donde las probabilidades de todas las clases se ajustan simultáneamente. En este caso, la función Softmax se define como

$$
\sigma(z^{(i)})_l = \frac{\exp(\theta_l^\top x^{(i)})}{\sum_{j=1}^{K} \exp(\theta_j^\top x^{(i)})},
$$

asegurando que todas las probabilidades sean positivas y sumen a uno. La log-verosimilitud correspondiente es

$$
\ell(\theta) = \sum_{i=1}^{n} \theta_{y^{(i)}}^\top x^{(i)} - \sum_{i=1}^{n} \log \Big( \sum_{j=1}^{K} \exp(\theta_j^\top x^{(i)}) \Big),
$$

y el gradiente para cada parámetro $\theta_l$ adopta la forma

$$
\frac{\partial \ell(\theta)}{\partial \theta_l} = \sum_{i=1}^{n} \Big( 1\{y^{(i)} = l\} - \sigma(z^{(i)})_l \Big) x^{(i)}.
$$

A diferencia del esquema OvA, este gradiente está acoplado entre clases a través del denominador común, lo que garantiza coherencia probabilística y mejora la calibración de los resultados en fronteras ambiguas. En la aplicación práctica, este modelo multinomial alcanzó resultados comparables en precisión global (≈98 %), pero con una matriz de confusión más equilibrada y menor dispersión, especialmente en observaciones limítrofes entre variedades de vino.

El modelo Softmax, sin embargo, enfrenta desafíos de estabilidad numérica. Dado que involucra la exponenciación de los valores de activación, puede experimentar *overflow* o *underflow* cuando los valores de $z$ son muy grandes o muy pequeños, produciendo gradientes inestables o pérdidas no definidas. Para mitigar este problema, se aplica una normalización previa restando el máximo de cada fila antes de exponenciar:

$$
z^{(i)} \leftarrow z^{(i)} - \max_j z^{(i)}_j,
$$

lo cual conserva las proporciones entre clases pero evita exponentes extremos. En código, esta técnica se implementó como `Z -= np.max(Z, axis=1, keepdims=True)`, logrando un entrenamiento estable incluso en etapas iniciales con pesos aleatorios o datos de alta varianza.

En cuanto a las predicciones, los enfoques OvA y multinomial tienden a coincidir en datasets bien separados, pero divergen cuando las fronteras de decisión son difusas o las clases se encuentran desbalanceadas. En este contexto, OvA tiende a sobrerrepresentar las clases mayoritarias y a producir probabilidades no normalizadas, mientras que el modelo multinomial, al ajustar globalmente las probabilidades, genera fronteras más suaves y coherentes. En el experimento con *Wine*, ambas variantes alcanzaron accuracies casi idénticas, pero la multinomial mostró una calibración superior en regiones fronterizas, reduciendo errores de clasificación en observaciones ambiguas. En consecuencia, aunque el enfoque OvA resulta computacionalmente eficiente, el modelo multinomial con Softmax ofrece un marco más sólido y teóricamente consistente para problemas multiclase, combinando estabilidad numérica, coherencia probabilística y una mejor calibración general de las predicciones.

