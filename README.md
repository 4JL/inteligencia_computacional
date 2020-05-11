## Problema

El problema que se desea resolver es tener un pronóstico de valores de la TRM paralos primeros días hábiles de Enero del 2020.Las variables financieras son de las más difíciles de pronosticar dada la volatilidad delcorto plazo por la cantidad de fenómenos que generan grandes fluctuaciones. Dada lanaturaleza de éste problema: donde se tiene entradas multivariadas en el modelo, ruidoen los datos, modelado de las relaciones más complejas en los datos; una muy buenaopción para abordar éste problema sería mediante redes neuronales.Para abordarlo se plantean diferentes arquitecturas de redes neuronales aprendidasdurante el curso: redes RNN (Recurrent Neural Network), Redes LSTM (Long ShortTerm Memory) y redes NARX (Nonlinear autoregressive models with exogenous inputs).La metodología utilizada para seleccionar el mejor modelo fue ejecutar un conjunto deexperimentos con diferentes hiper-parámetros, para cada tipo de red neuronal. Luegocomparamos resultados y se selecciona el modelo que tiene un menor error en la predicción.

## Modelos

A continuación la estructura de esta entrega:

Cada modelo tiene su propia base de datos con la cual se trabajan los programas.