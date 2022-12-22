# TODO : This is a TimeSeries Problem defenitely *
# Pipeline + Cross Validation of Ridge, Lasso, ElasticNet and Majority Vote
# k-fold with similar incrementally parameters: lambda, C, alpha and another ufsesul
# Wich and interesting dataset 'ai4i2020.csv'
# se debe de hacer regresion a cada uno de los targets, aqui lo interesante es lo predictivo NO la clasificacion.
# predictivo: que tan cerca esta uno de esos target al fallo? ver cuales son las variables mas adecuadas a cada target
# la prediccion por individual se puede 'mergear' y presentar una sola vista
# https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
# incluso se podria simular una maquina que envia (por rest) data y se ve evaluada en un monitoreo
#
# TODO: accelerometer.csv for classification task y:wconfid, X:[x, y, z]

# TODO: https://www.kaggle.com/datasets/spscientist/telecom-data
# crear historico por numeros para applicacion ralista y predictiva.
# Los que tiene Churn=True seria la observacion "mas grande" de su historia
# Los demas deberian tener valores decrementales simulando que van en aumento en cuanto a consumo, etc

# TODO: Estudio del Comportamiento Progresivo de una Poblacion
# Discurso:
# (a) Lo socialmente establecido, lo politicamente correcto en una poblacion o sociedad
# (b) las personas con comportamientos difetentes y destacados por su concentracion, tienen una clasificacion
#     desde el punto (a)
#   cada uno de nosotros tenemos alguna propiedad con % desarrollos y otros muy desarrolladas dejando de notar otras
#   Ocurre que una poblacion aislada de otra representa un comportamiento politicamente correcto en cada ciudad/poblacion
#   e inverso con elementos cruzados.
#   El comportamiento, acciones, y demas de una poblacion puede ser oscilatorio, estacionario, ciclico
#   hasta el punto que se pueden clusterizar teniendo en cuenta que pueden cambiar de A a B en t tiempo
#   Esto es una poblacion dinamica, de interez: ANTICIPAR cuando un (muchos) elemento A pasaria a B con t tiempo    .
#   suponiendo que A es el estatus quo
#   Se hace procedimiento:
#    1) Se observan clientes que se fueron (churn)
#    2) Se toman estos clientes con t tiempo historia atras (t tiempo (meses despoues) se fueron-churn)
#      2.1) Que tienen estos clientes en comun ademas del churn?
#    3) Cuales son los otros, los que permanecen?
#    El tiempo juega un papel muy importante, con este enfoque se hace "un filtro" a t tiempo para "capturar"
#    los que con alta probabilidad seran churn
#    TODO: Otro ennfoque, Como quedaron (la data X-features) los numeros en el momento que se fueron-churn?
#     1) hacer Clasificacion de churn, y una muestra aleatoria con longitud de historia similar que representara
#     los NO-churn. Pues ya se como clasificar de acuerdo a X-features
#     2) Pronostico-TimeSeries cada numero-elemento (bien cabron) con el resultado de una nueva base generada,
#        hago clasificacion y asi se quienes seran churn a partir de cualquier tiempo
#     Umm pensandolo bien, como queda la data al final (fecha del churn) no es determinante, no vale de nada la data
#     historica? es determinante como termina o es determinante el mean de como termina ?.
#    -> para formar la base de 3 meses por ej. se toma de diferentes epocas ya que hacerlo por ej. en diciembre puede tener bias
#    mas interesante: se puede hacer de varios grupos de meses, quien se va a los 4 meses, 3 meses en un mes
#     .