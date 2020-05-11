## Problema

El problema que se desea resolver es tener un pronóstico de valores de la TRM para los primeros días hábiles de Enero del 2020.Las variables financieras son de las más difíciles de pronosticar dada la volatilidad delcorto plazo por la cantidad de fenómenos que generan grandes fluctuaciones. Dada lanaturaleza de éste problema: donde se tiene entradas multivariadas en el modelo, ruido en los datos, modelado de las relaciones más complejas en los datos; una muy buena opción para abordar éste problema sería mediante redes neuronales.Para lograrlo se plantean diferentes arquitecturas de redes neuronales aprendidas durante el curso: redes RNN (Recurrent Neural Network), Redes LSTM (Long ShortTerm Memory) y redes NARX (Nonlinear autoregressive models with exogenous inputs).La metodología utilizada para seleccionar el mejor modelo fue ejecutar un conjunto de experimentos con diferentes hiper-parámetros, para cada tipo de red neuronal. Luego comparamos resultados y se selecciona el modelo que tiene un menor error en la predicción.

## Modelos

A continuación la estructura de esta entrega:

![Image of Yaktocat](https://github.com/4JL/inteligencia_computacional/blob/master/image.png?raw=true)

Cada modelo tiene su propia base de datos con la cual se trabajan los programas.
