from mtranslate import translate
import pandas as pd
import random
from tqdm import tqdm

# Función para traducir texto a un idioma específico
def translate_text(text, target_language='en'):
    try:
        translated_text = translate(text, target_language).encode('utf-8')
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

def traduce(pathTest, pathTraducedTest, language):
    # Cargar el CSV
    df = pd.read_csv(pathTest)
    df = df[:10]

    # Crear un nuevo DataFrame para almacenar solo las instancias traducidas
    translated_df = pd.DataFrame(columns=['id', 'text', 'class'])

    # Iterar sobre cada fila del DataFrame
    for index, row in df.iterrows():
        # Obtener el ID, texto y clase
        instance_id = row['id']
        text = row['text']
        label = row['class']

        # Traducir el texto a cada idioma y agregar al DataFrame
        translated_text = translate_text(text, language)

        # Crear una nueva fila con el mismo ID y clase, pero con el texto traducido
        new_row = pd.DataFrame({
            'id': [instance_id],
            'text': [translated_text],
            'class': [label]
        })

        # Agregar la nueva fila al DataFrame de instancias traducidas
        translated_df = pd.concat([translated_df, new_row], ignore_index=True)

    # Guardar el DataFrame de instancias traducidas en el CSV
    translated_df.to_csv(pathTraducedTest, index=False)



def randomize(pathTest,pathRandomizedTest):
    # Lee el archivo CSV original
    df = pd.read_csv(pathTest)
    df = df[:10]

    # Función para mezclar las palabras en un texto
    def mezclar_palabras(texto):
        palabras = texto.split()
        random.shuffle(palabras)
        return ' '.join(palabras)

    # Aplica la función a la columna 'text'
    df['text'] = df['text'].apply(mezclar_palabras)

    # Guarda el DataFrame modificado en un nuevo archivo CSV
    df.to_csv(pathRandomizedTest, index=False)



if __name__ == '__main__':
    traduce('../Datasets/test.csv','../Datasets/traduced_test_es.csv','es')

