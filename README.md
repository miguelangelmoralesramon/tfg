# TFG
Este es el código de mi TFG, donde abarco las redes neuronales más importantes hasta terminar con el transformer. 

- ann_MNIST.ipynb contiene el código utilizado para entrenar las redes neuronales artificiales correspondientes al capítulo 2, así como las métricas calculadas para evaluar su rendimiento.
- cnn_MNIST.ipynb es el script utilizado para diseñar y entrenar las redes neuronales artificiales del capítulo 3.
- Analisis_de_reviews_IMDb.ipynb contiene los distintos tipos de redes neuronales recurrentes utilizadas en el capítulo 4 para clasificar reseñas de IMDb.
- En la carpeta 'transformers' se encuentran los siguientes archivos:
  - transformer.py : script con el desarrollo de la clase Transformer, que replica la arquitectura y funcionamiento de este tipo de redes.
  - transformer_training.ipynb : código utilizado para entrenar todos los modelos evaluados del traductor de inglés a español.
  - translation_arrays.py : contiene los arrays con las predicciones sobre el conjunto de prueba de cada uno de los modelos entrenados.
    original_sentence son las frases del conjunto de test traducidas por los modelos.
    spanish_translation son las traducciones en español del conjunto de prueba.
    prediction_{"nombre del transformer"} contiene las predicciones realizadas sobre el conjunto de prueba del transformer que se indica.
  - transformers_BLEU.upynb : cálculo de las métricas sacreBLEU para evaluar los transformers entrenados.

## Utilización de los transformers
Entrenar los modelos del archivo transformer_training.ipynb ha requerido de más de 40h utilizando una GPU NVIDIA A100. Para evitar este costoso proceso, los modelos están disponibles en esta [carpeta de OneDrive](https://ucomplutense-my.sharepoint.com/:f:/g/personal/mimora02_ucm_es/EpydMcjnsEJHkTRVycbAA8EBFRQMjzfQFhnJGbe5bDzXug?e=yXNT56), solo para miembros de la UCM. Los modelos tienen los mismos nombres que en la memoria del TFG: transformer_base, transformer1, transformer2 y transformer_final.pth.tar.

#### Para utilizar cualquiera de los modelos hay que seguir los siguientes pasos:

1. Clonar el repositorio:
```bash
cd 'path'
git clone git@github.com:miguelangelmoralesramon/tfg.git
```
2. Descargar el modelo que se desea utilizar:
<img width="1113" alt="Screenshot 2024-06-26 at 17 42 51" src="https://github.com/miguelangelmoralesramon/tfg/assets/30403390/9d658279-f419-4d62-8ba0-fce7d41aec84">
3. Abrir el archivo transformer_training.ipynb y crear el objeto del modelo que se quiere utilizar:

```python

d_model = 512
batch_size = 50
ffn_hidden = 2048
num_heads_1 = 16
num_heads_2 = 8
num_heads_3 = 8
drop_prob = 0.1
num_layers_1 = 1
num_layers_2 = 2
num_layers_3 = 4
max_sequence_length = 300
esp_vocab_size = len(spanish_vocab)

transformer_final = Transformer(d_model,
                          ffn_hidden,
                          num_heads_3,
                          drop_prob,
                          num_layers_3,
                          max_sequence_length,
                          esp_vocab_size,
                          eng_to_ind,
                          esp_to_ind,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)
```
4. Cargar el modelo pre-entrenado en el objeto creado. Para ello será necesario el path del modelo desacargado, en mi caso:
   
```python
transformer_final.load_state_dict(torch.load('./drive/MyDrive/ColabNotebooks/Models/transformer_final.pth.tar',map_location=torch.device('cuda')))
```
5. Enviar el transformer a la GPU:
```python
transformer_final.to(torch.device('cuda'))
```
6. Utilizar la función translate() para traducir:
```python
traduccion = translate(transformer_final, "my name is miguel")
```
```bash
>>> my nombre es miguel
```
:warning: Los vocabularios utilizados para entrenar los modelos no contienen letras mayúsculas, así que intentar traducir una frase que las contenga seguramente resulte en un error :warning:
