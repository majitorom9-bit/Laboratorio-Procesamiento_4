# Laboratorio-Procesamiento_4

# OBJETIVOS DE LA PRACTICA

Analizar de manera integral señales electromiográficas (EMG) mediante la aplicación de técnicas de procesamiento digital y herramientas computacionales, incluyendo el filtrado, la segmentación y el análisis en el dominio de la frecuencia, con el propósito de identificar y caracterizar la aparición de la fatiga muscular. Esto se logrará a partir del cálculo e interpretación de parámetros espectrales como la frecuencia media y mediana, así como de la comparación entre señales emuladas y señales reales, permitiendo comprender el comportamiento fisiológico del músculo durante contracciones repetidas y su relación con los fenómenos de fatiga.

# PARTE A
**1. OBTENER LA SEÑAL**
En esta primera parte, se configuró el generador de señales biológicas en modo EMG, con el objetivo de simular el comportamiento de un músculo durante contracciones voluntarias. Se ajustaron los parámetros y obtuvimos aproximadamente cinco contracciones consecutivas, asegurando que la señal tuviera una forma representativa y clara para su análisis.
Una vez configurado el generador, adquirimos la señal. Durante este proceso, se verificó que la señal fuera capturada de manera continua y sin interrupciones, garantizando la integridad de la información. Luego de ello, la señal fue almacenada para su procesamiento.
Por ultimo se cargo la señal en un archivo txt con los parametros basicos y estos son los valores mostrados 

```python

import numpy as np
import matplotlib.pyplot as plt

with open('Senal_lab_4_partea_ver3.txt', 'r') as f:
    contenido = f.read()

datos = np.array(contenido.split(), dtype=float)

mitad = len(datos)//2
t = datos[:mitad]
senal = datos[mitad:]

print(f"Longitud tiempo: {len(t)} | Longitud señal: {len(senal)}")

```


**2. GRAFICA DE LA SEÑAL**
Con el código implementado se grafica la señal electromiográfica (EMG) en función del tiempo, utilizando los datos previamente cargados y un eje temporal definido a partir de la frecuencia de muestreo. De esta forma, se representa la amplitud de la señal, permitiendo observar las contracciones simuladas mediante picos en la gráfica.

Además, el código ajusta la visualización para enfocarse en un intervalo de 0 a 5 segundos, lo que facilita el análisis sin que la señal se vea comprimida. En la gráfica, el eje horizontal corresponde al tiempo (s) y el eje vertical a la amplitud (V), mostrando claramente la variación de la actividad muscular a lo largo del tiempo.

```python

plt.figure(figsize=(8,4))
plt.plot(t, senal, color='red')
plt.title(f"fs={1/(t[1]-t[0]):.0f}Hz, duración={t[-1]:.1f}s, muestras={len(senal)}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)
plt.show()

```

<img width="536" height="310" alt="image" src="https://github.com/user-attachments/assets/1fd530f0-5116-41f5-92c4-d87607243814" />

**3. SEGMENTACION DE LA SEÑAL**

Con la señal ya registrada, se realizó la segmentación, identificando los intervalos correspondientes a cada una de las cinco contracciones simuladas. Esta división permitió aislar cada evento y facilitar su análisis individual en etapas posteriores.

**3.1 Preprocesamiento de la Señal**

En esta parte se define la frecuencia de muestreo de la señal, lo que permite establecer la relación entre las muestras y el tiempo, lo cual es fundamental para cualquier análisis posterior, especialmente en frecuencia. El objetivo es centrar la señal y resaltar la intensidad de la actividad muscular, independientemente de su signo.

```python

fs = 2000  # Hz

senal_proc = np.abs(senal - np.mean(senal))

```
**3.2 Filtrado y Normalizacion** 

En esta parte, se implementó un filtro pasabajos de segundo orden con una frecuencia de corte de 2 Hz, utilizando la función butter y aplicándolo con filtfilt  Este proceso permite obtener la envolvente de la señal, es decir, una versión suavizada que facilita la identificación de las contracciones musculares al eliminar variaciones rápidas y ruido.
Luego, la señal filtrada fue normalizada, con el fin de llevar sus valores a un rango entre 0 y 1, lo que mejora la comparabilidad y la detección de eventos.
Finalmente, se realizó la detección de contracciones utilizando un umbral definido como la suma del promedio de la señal normalizada y la mitad de su desviación estándar,y por ultimo se segmento la señal en eventos individuales.

```python

#filtro pasabajos
b, a = butter(2, 2 / (fs / 2), btype='low')
filtrada = filtfilt(b, a, senal_proc)

#normalizar
filtrada_norm = filtrada / np.max(filtrada)

#deteccion de umbrales, cuando pasa el umbral inicia una contraccion y cuando vuelve a estar debajo del umbral se termina la contraccion
umbral = np.mean(filtrada_norm) + 0.5 * np.std(filtrada_norm)
en_contraccion = filtrada_norm > umbral
inicios = np.where(np.diff(en_contraccion.astype(int)) == 1)[0]
fines = np.where(np.diff(en_contraccion.astype(int)) == -1)[0]

```

**3.3 Ajuste y extraccion de segmentos**

Por ultimo, se realizó un ajuste de los segmentos con el fin de garantizar que cada contracción estuviera correctamente delimitada, asegurando coherencia entre los puntos de inicio y fin detectados. Este paso es importante para evitar errores en la segmentación y asegurar que cada evento muscular sea analizado de forma adecuada.

Luego, se procedió a la visualización de la señal, graficando tanto la señal original normalizada como la señal filtrada. Además, se resaltaron las regiones correspondientes a las contracciones detectadas mediante sombreado, lo que permitió una interpretación visual clara del proceso de segmentación.

Finalmente, se extrajeron los segmentos de la señal correspondientes a cada contracción utilizando segmentos. Estos segmentos constituyen la base para el análisis posterior, ya que sobre ellos se calcularán parámetros en el dominio de la frecuencia, como la frecuencia media y la frecuencia mediana, tal como lo establece la guía de la práctica.

```python
if len(fines) < len(inicios):
    fines = np.append(fines, len(senal) - 1)
elif len(fines) > len(inicios):
    fines = fines[:len(inicios)]

print(f"Contracciones detectadas: {len(inicios)}")

plt.figure(figsize=(12, 4))
plt.plot(t, senal / np.max(np.abs(senal)), label="Señal EMG", color='hotpink', alpha=0.7)
plt.plot(t, filtrada_norm, label="Señal filtrada", color='palevioletred', linewidth=2)

for ini, fin in zip(inicios, fines):
    plt.axvspan(t[ini], t[fin], color='skyblue', alpha=0.3)

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.title('Detección y segmentación de contracciones musculares simuladas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

segmentos = [senal[ini:fin] for ini, fin in zip(inicios, fines)]
print(f"Se extrajeron {len(segmentos)} segmentos.")

```
**3.4 Grafica de la señal**

Se generó la gráfica de la señal EMG en función del tiempo, donde se observa la señal original junto con la señal filtrada. Las contracciones musculares detectadas se resaltan con regiones sombreadas en color azul, lo que permite identificar claramente su ubicación y duración dentro del registro. En total, se detectan cinco contracciones, coherentes con el comportamiento de la señal.

Esta representación facilita la interpretación visual del proceso de segmentación, ya que los picos de la señal filtrada coinciden con las zonas sombreadas. A partir de esto, se extraen los segmentos correspondientes a cada contracción, los cuales serán utilizados para el análisis posterior de sus características.

<img width="906" height="329" alt="image" src="https://github.com/user-attachments/assets/10d2e53e-f616-42a1-b103-b59dfb048a80" />



**4. ANALISIS DE FRECUENCIA**

Para cada uno de los segmentos obtenidos, se aplicó la transformada de Fourier con el fin de obtener el espectro de frecuencias. A partir de este espectro, se calcularon dos parámetros fundamentales: la frecuencia media y la frecuencia mediana, los cuales describen la distribución de energía de la señal en el dominio frecuencial.

**4.1 Frecuencia de muestreo**
Se determino el periodo de muestreo y la frecuencia de muestreo a partir del vector de tiempo
Con esto se obtuvo un periodo de muestreo de dt = 0.0005 s y una frecuencia de muestreo de fs = 2000 Hz, valores adecuados para el análisis de señales EMG.

```python

# Frecuencia de muestreo
=
dt = np.mean(np.diff(t))   # periodo de muestreo
fs = 1 / dt                # frecuencia de muestreo

print("dt =", dt)
print("fs =", fs, "Hz")

```

**4.2 Definición de funciones**

Se definieron funciones para calcular la frecuencia media y la frecuencia mediana a partir del espectro de potencia en dos partes con igual energía.

```python

def calcular_frecuencia_media(freqs, potencia):
    return np.sum(freqs * potencia) / np.sum(potencia)

def calcular_frecuencia_mediana(freqs, potencia):
    potencia_acumulada = np.cumsum(potencia)
    mitad_potencia = potencia_acumulada[-1] / 2
    idx = np.where(potencia_acumulada >= mitad_potencia)[0][0]
    return freqs[idx]

```

**4.3 Aplicación de la FFT a cada contracción**

Para cada segmento de la señal, se eliminó el valor promedio y se aplicó la Transformada Rápida de Fourier (FFT), Este proceso permite transformar la señal del dominio del tiempo al dominio de la frecuencia.

```python

frecuencias_medias = []
frecuencias_medianas = []

for i, c in enumerate(segmentos):

    c = c - np.mean(c)

    N = len(c)

    # FFT
    fft_vals = np.fft.fft(c)
    freqs = np.fft.fftfreq(N, d=dt)

```

**4.4 Cálculo del espectro de potencia**

Se seleccionaron únicamente las frecuencias positivas y se calculó la potencia de la señal, esto permite analizar únicamente la información relevante del espectro.

```python

mask = freqs >= 0
freqs_pos = freqs[mask]
fft_pos = fft_vals[mask]

potencia = np.abs(fft_pos)**2

```

**4.5 Cálculo de parámetros frecuenciales**

A partir del espectro de potencia, se calcularon la frecuencia media y la frecuencia mediana para cada contracción, 

```python

f_media = calcular_frecuencia_media(freqs_pos, potencia)
f_mediana = calcular_frecuencia_mediana(freqs_pos, potencia)

```
**4.6 Almacenamiento y visualización de resultados**

Finalmente, los valores de frecuencia media y frecuencia mediana obtenidos para cada contracción se almacenan en listas y se imprimen en consola, en este fragmento, los resultados de cada contracción se guardan para su posterior uso, como la generación de gráficas o análisis comparativos. Además, se muestran en pantalla de forma organizada, indicando el número de contracción junto con sus respectivas frecuencias media y mediana.

```python

frecuencias_medias.append(f_media)
frecuencias_medianas.append(f_mediana)

print(f"Contracción {i+1}:")
print(f"  Frecuencia media   = {f_media:.2f} Hz")
print(f"  Frecuencia mediana = {f_mediana:.2f} Hz")
print("-----------------------------------")

```
Los resultados obtenidos fueron:

dt = 0.0005

fs = 2000.0 Hz


**5. RESULTADOS OBTENIDOS**

Finalmente, los valores calculados para cada contracción fueron organizados en una tabla. Además, se realizaron gráficas que mostraban la evolución de la frecuencia media y mediana a lo largo de las contracciones simuladas, permitiendo visualizar tendencias y comportamientos de la señal.

```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tabla_resultados = pd.DataFrame({
    "Contracción": np.arange(1, len(segmentos)+1),
    "Frecuencia media (Hz)": frecuencias_medias,
    "Frecuencia mediana (Hz)": frecuencias_medianas
})

print("Tabla de resultados:")
display(tabla_resultados)

```

<img width="373" height="159" alt="image" src="https://github.com/user-attachments/assets/0e99c8dd-5bf4-42b1-8522-e8897144ad56" />


**6. GRAFICA DE FRECUENCIAS**

Con el fin de analizar el comportamiento de la señal en el dominio de la frecuencia, se realizó una gráfica que muestra la evolución de la frecuencia media y la frecuencia mediana a lo largo de las contracciones, se crea una figura donde se representan dos curvas: una correspondiente a la frecuencia media y otra a la frecuencia mediana, ambas en función del número de contracción. 
Esta representación permite observar de manera clara la tendencia de las frecuencias a lo largo del tiempo. En general, se evidencia un comportamiento relativamente estable en las primeras contracciones, con una ligera disminución hacia la última. Este patrón es característico de la fatiga muscular, ya que el contenido frecuencial de la señal EMG tiende a desplazarse hacia frecuencias más bajas a medida que el músculo se fatiga.

```python

plt.figure(figsize=(10,5))

plt.plot(tabla_resultados["Contracción"],
         tabla_resultados["Frecuencia media (Hz)"],
         marker='o', linewidth=2, label='Frecuencia media')

plt.plot(tabla_resultados["Contracción"],
         tabla_resultados["Frecuencia mediana (Hz)"],
         marker='s', linewidth=2, label='Frecuencia mediana')

plt.xlabel("Número de contracción")
plt.ylabel("Frecuencia (Hz)")
plt.title("Evolución de la frecuencia media y mediana")
plt.xticks(tabla_resultados["Contracción"])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

<img width="747" height="373" alt="image" src="https://github.com/user-attachments/assets/5cb53497-fac5-4d8d-af78-20451df5b229" />




**Diagrama de Flujo**

<img width="915" height="1535" alt="image" src="https://github.com/user-attachments/assets/7530c8e1-9073-47f1-bfa3-00f4b2d4b209" />

# PARTE B 
**1. PREPARACION Y UBICACION DE LOS ELECTRODOS**

En esta etapa, se seleccionó el grupo muscular a analizar (como el bíceps o antebrazo) y se procedió a la colocación de los electrodos de superficie. Se aseguró que la piel estuviera limpia y seca.

**2. OBTENER LA SEÑAL **

Una vez colocados los electrodos, se registró la señal electromiográfica mientras el voluntario realizaba contracciones musculares repetidas hasta alcanzar un estado de fatiga. Durante la adquisición, se mantuvieron condiciones controladas para evitar interferencias externas.

3. Filtrado de la señal

La señal obtenida fue procesada mediante la aplicación de un filtro pasa banda entre 20 y 450 Hz. Este paso permitió eliminar componentes no deseados como ruido de baja frecuencia, interferencias eléctricas y artefactos de movimiento, conservando únicamente la información relevante de la actividad muscular.

4. Segmentación de la señal

Posteriormente, la señal filtrada fue dividida en segmentos correspondientes a cada contracción realizada por el voluntario. Esta segmentación permitió analizar de forma individual cada evento muscular.

5. Cálculo de parámetros frecuenciales

Para cada uno de los segmentos, se calcularon la frecuencia media y la frecuencia mediana a partir del espectro de la señal. Estos parámetros fueron utilizados para caracterizar el comportamiento del músculo durante las contracciones.

6. Análisis de resultados

Finalmente, los resultados obtenidos fueron graficados, permitiendo observar la evolución de las frecuencias a medida que avanzaba la fatiga muscular. A partir de estas gráficas, se analizaron las tendencias y se identificaron patrones asociados al cansancio muscular.

# PARTE C

**1. APLICACION DE LA TRANSFORMADA RAPIDO DE FOURIER (FFT)**

En esta etapa, se aplicó la Transformada Rápida de Fourier (FFT) a cada uno de los segmentos de la señal EMG real. Este proceso permitió transformar la señal del dominio del tiempo al dominio de la frecuencia. 

**1.1 Inicialización de variables**

Se crearon listas para almacenar las frecuencias y magnitudes obtenidas de cada contracción. Además, se define el tamaño de la figura, ajustándolo dinámicamente según el número de segmentos, para visualizar cada FFT de forma organizada.

```python

frecuencias_fft = []
magnitudes_fft = []

plt.figure(figsize=(12, 2.5*len(segmentos)))

```

**1.2 Procesamiento de cada contracción y aplicacion de la FFT**

Se recorre cada segmento de la señal (cada contracción). Primero se elimina el valor promedio para centrar la señal y evitar componentes DC, y se obtiene el número de muestras, necesario para calcular la FFT.
Se aplica la Transformada Rápida de Fourier (FFT), que permite pasar la señal del dominio del tiempo al dominio de la frecuencia. Además, se calcula el vector de frecuencias asociado a cada componente del espectro.

```python

for i, seg in enumerate(segmentos):

    seg = seg - np.mean(seg)

    N = len(seg)

    # FFT
    fft_vals = np.fft.fft(seg)
    freqs = np.fft.fftfreq(N, d=1/fs)

```

**1.3 Frecuencias positivas**

Se seleccionan únicamente las frecuencias positivas, ya que contienen la información relevante de la señal. Luego, se calcula la magnitud del espectro, que indica la intensidad de cada componente frecuencial.

```python

mask = freqs >= 0
freqs_pos = freqs[mask]
magnitud = np.abs(fft_vals[mask])

```
**1.4 Resultados y Representacion Grafica**

Se guardan las frecuencias y sus magnitudes correspondientes para cada contracción, lo que permite un análisis posterior o comparaciones entre segmentos, luego se grafica el espectro de cada contracción en subgráficas independientes. Se limita el eje de frecuencia entre 0 y 500 Hz (frecuencia de Nyquist), lo cual es coherente con la frecuencia de muestreo utilizada. Esto permite visualizar cómo se distribuye la energía de la señal en el dominio de la frecuencia.
Finalmente, se ajusta la distribución de las gráficas para evitar superposición y se muestran en pantalla.

```python

frecuencias_fft.append(freqs_pos)
magnitudes_fft.append(magnitud)

 plt.subplot(len(segmentos), 1, i+1)
    plt.plot(freqs_pos, magnitud, color='navy')
    plt.xlim(0, 500)  # como fs=1000, Nyquist=500 Hz
    plt.title(f'FFT - Contracción {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(True)

plt.tight_layout()
plt.show()

```

| PRIMERAS 5 CONTRACCIONES

<img width="605" height="629" alt="image" src="https://github.com/user-attachments/assets/b302a411-845a-4efa-ac8b-2ed7061fbfd1" />


| 5 CONTRACIONES DEL MEDIO

<img width="614" height="627" alt="image" src="https://github.com/user-attachments/assets/647cfae2-e74a-47e2-a44a-1caf52d279a2" />


| ULTIMAS 5 CONTRACCIONES 

<img width="614" height="630" alt="image" src="https://github.com/user-attachments/assets/8f7c5e65-36b9-444c-8f3b-bd4a93a1eed7" />


**2 COMPARACION ENTRE CONTRACCIONES**

Se realizó una comparación entre los espectros de las primeras contracciones y los de las últimas. Esto permitió evidenciar cambios en el contenido frecuencial de la señal a medida que el músculo se fatigaba.

**2.1 Gráfica de la primera contracción**

Aquí se grafica el espectro de la primera contracción. La magnitud se normaliza dividiéndola por su valor máximo, lo que permite que los valores estén entre 0 y 1. El objetivo es representar el contenido frecuencial inicial del músculo, cuando aún no presenta fatiga significativa.

```python

plt.figure(figsize=(10,5))

plt.plot(frecuencias_fft[0],  magnitudes_fft[0] / np.max(magnitudes_fft[0]), label='Primera contracción')

```

**2.2 Gráfica de la última contracción**

En este caso se grafica el espectro de la última contracción registrada. Al igual que en el caso anterior, se normaliza la magnitud para facilitar la comparación, El objetivo es observar cómo cambia el contenido frecuencial cuando el músculo ya está fatigado.

```python
plt.plot(frecuencias_fft[-1],
         magnitudes_fft[-1] / np.max(magnitudes_fft[-1]),
         label='Última contracción')
```

**2.3 Configuración de ejes Y grafica**

Se limita el eje de frecuencia entre 0 y 500 Hz, que corresponde a la frecuencia de Nyquist. Además, se etiquetan los ejes para indicar qué representa cada uno, finalmente se muestra la grafica

```python

plt.xlim(0, 500)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud normalizada')
plt.title('Comparación espectral: primera vs última contracción')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="641" height="361" alt="image" src="https://github.com/user-attachments/assets/e63f8f80-1189-4417-87d3-ed380d7f9481" />


**3. Identificación de patrones de fatiga**

A partir del análisis espectral, se identificó una disminución de las componentes de alta frecuencia y un desplazamiento del contenido espectral hacia frecuencias más bajas. Estos cambios son característicos de la fatiga muscular.

**3.1. Definición de frecuencia**

Aquí se definen condiciones (máscaras) para seleccionar únicamente las frecuencias mayores a 150 Hz, tanto para la primera como para la última contracción, el objetivo es enfocarse específicamente en las altas frecuencias, que son las más afectadas por la fatiga muscular.


```python

plt.figure(figsize=(10,5))

mask_i = frecuencias_fft[0] > 150
mask_f = frecuencias_fft[-1] > 150

```

**3.2 Gráfica de la primera contracción (altas frecuencias)**

Se grafica el espectro de la primera contracción, pero solo en el rango de altas frecuencias. La magnitud se normaliza para facilitar la comparación. El objetivo es mostrar el contenido de altas frecuencias cuando el músculo aún no está fatigado.

```python

plt.plot(frecuencias_fft[0][mask_i],
         magnitudes_fft[0][mask_i] / np.max(magnitudes_fft[0]),
         label='Primera contracción')
```


**3.3 Gráfica de la última contracción (altas frecuencias)**

Se grafica la última contracción bajo las mismas condiciones. Esto permite comparar directamente cómo han cambiado las componentes de alta frecuencia.

```python
plt.plot(frecuencias_fft[-1][mask_f],
         magnitudes_fft[-1][mask_f] / np.max(magnitudes_fft[-1]),
         label='Última contracción')
```


**3.4 Configuración de ejes**
Se etiquetan los ejes para indicar que se está representando la frecuencia en Hertz y la magnitud normalizada.

```python

plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud normalizada')
plt.title('Reducción del contenido de alta frecuencia')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="659" height="357" alt="image" src="https://github.com/user-attachments/assets/7330c024-b282-4c2e-9fd7-b6e75cdc1b24" />

**4.Análisis del pico espectral**

**4.1 Recorrido de los segmentos**

Se recorre cada una de las contracciones previamente segmentadas, con el objetivo es analizar cada segmento de manera independiente.

```python
picos_espectrales = []

```

**4.2 Identificación del índice del pico**

Se obtiene el índice donde la magnitud del espectro es máxima, se identifica la posición del pico espectral, es decir, el punto donde la señal tiene mayor energía. A partir del índice obtenido, se determina la frecuencia correspondiente. Se guarda la frecuencia del pico en la lista creada inicialmente, Esto permite construir un conjunto de datos para analizar la evolución de esta frecuencia a lo largo del tiempo.

```python
for i in range(len(segmentos)):
idx_pico = np.argmax(magnitudes_fft[i])
f_pico = frecuencias_fft[i][idx_pico]
picos_espectrales.append(f_pico)
```

6. Representación gráfica
Se grafica la frecuencia del pico espectral en función del número de contracción. El objetivo es visualizar cómo cambia la frecuencia dominante a lo largo de las contracciones.

```python
plt.plot(range(1, len(picos_espectrales)+1),
         picos_espectrales,
         marker='o')
plt.xlabel('Contracción')
plt.ylabel('Frecuencia del pico (Hz)')
plt.title('Desplazamiento del pico espectral')
plt.grid(True)
plt.show()
```


<img width="463" height="350" alt="image" src="https://github.com/user-attachments/assets/e6e9bbcc-6e99-4c7d-b08a-46631811f89d" />


# ANALISIS 

# PREGUNTAS A DISCUSIÓN 
# CONCLUSIONES


Parte A: señal simulada → entender comportamiento básico

Parte B: señal real → detectar fatiga

Parte C: FFT → analizar en frecuencia y confirmar fatiga
