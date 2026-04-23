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


# PARTE B 

# PARTE C

# ANALISIS 

# PREGUNTAS A DISCUSIÓN 
# CONCLUSIONES


Parte A: señal simulada → entender comportamiento básico

Parte B: señal real → detectar fatiga

Parte C: FFT → analizar en frecuencia y confirmar fatiga
