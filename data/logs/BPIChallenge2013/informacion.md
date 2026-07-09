BPI Challenge 2013, incidents (Volvo IT — VINST)
Contiene 7.554 trazas y 65.533 eventos del proceso de gestión de incidentes; cada evento registra atributos como el estado del problema, el equipo de soporte (grupo) involucrado y la persona que trabaja en el problema. Otra fuente detalla que los eventos incluyen información no obligatoria sobre lifecycle, grupo, empleado responsable, país del recurso, país de la organización, organizaciones involucradas, impacto y producto. figsharearxiv
Definición de participante. Es el más rico de los tres: hay al menos tres candidatos a Participant con granularidades distintas — la línea organizacional (1ª/2ª/3ª línea de soporte), el grupo de soporte, y el país de la organización. El comportamiento de "ping-pong" entre equipos de soporte es uno de los fenómenos centrales del log: idealmente un incidente debería resolverse sin involucrar demasiados equipos — esto garantiza empíricamente lo que tu mapeo necesita: múltiples unidades organizacionales interleaved dentro del mismo caso, con handovers frecuentes. Elegir cuál atributo es el pool sería una decisión D a registrar (mi lectura: la línea organizacional se acerca más a "pool BPMN"; el grupo se acerca más a "lane". No tengo certeza de la cardinalidad de valores de cada atributo — verificar en el archivo). figshare
Limitación estructural. No hay eventos de mensaje. Toda la interacción es implícita en el cambio de grupo entre eventos consecutivos.

3. Hallazgo principal: Queued como observación de comunicación
El patrón, verificado cuantitativamente sobre los 65.533 eventos:

De los 4.350 handovers entre líneas, el 93,1% entra a la nueva línea mediante un evento Queued/Awaiting Assignment.
De esos 4.051 eventos Queued con cambio de línea, el 98,2% es ejecutado por el recurso del handler anterior (el emisor encola el ticket en la línea destino), y el 96,7% es seguido por continuación efectiva en la línea nueva.

Es decir: el evento Queued con cambio de línea es una transferencia observada. Y encaja de manera natural con tu diseño vigente:

El evento está estampado con la línea receptora en organization involved. Bajo tu regla "el participante de un ReceiveTask es siempre el receptor", leer estos eventos como observaciones de recepción es consistente con la atribución que el propio log ya hace — no hay que reatribuir nada.
Bajo el modelo de mensaje independiente (D11) esto es directamente mapeable: S_L = ∅, R_L = {eventos Queued con cambio de línea}, un objeto Message por observación, sin ρ y sin necesidad de emparejar con un send que el sistema no registra. Con el modelo correlacionado anterior, este log habría sido inutilizable o habría exigido fabricar pares.
receiver = línea estampada; sender sería derivable como la línea del evento inmediatamente anterior — derivación determinista pero es una inferencia de preprocesamiento que habría que declarar (o dejar sender indefinido, decisión abierta).
Hay 1.156 eventos Queued en posición inicial de traza: transferencias desde un origen no observado. Es exactamente tu categoría de "unmatched receives": se flaggean y reportan, no se corrigen.

