from deep_translator import GoogleTranslator
import pandas as pd
import random
import re 

def translate_text(text, language):
    try:
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        cleaned_text = cleaned_text[:4999]

        translated_text = GoogleTranslator(source='en', target=language).translate(cleaned_text)

        return translated_text
    
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

def traduce(pathTest, pathTraducedTest, language, numinstances):

    df = pd.read_csv(pathTest)
    df = df[:numinstances]

    translated_df = pd.DataFrame(columns=['id', 'text', 'class'])

    for index, row in df.iterrows():
        # Obtener el ID, texto y clase
        instance_id = row['id']
        text = row['text']
        label = row['class']

        translated_text = translate_text(text, language)

        new_row = pd.DataFrame({
            'id': [instance_id],
            'text': [translated_text],
            'class': [label]
        })

        translated_df = pd.concat([translated_df, new_row], ignore_index=True)

    translated_df.to_csv(pathTraducedTest, index=False)



def randomize(pathTest,pathRandomizedTest, numinstances):

    df = pd.read_csv(pathTest)
    df = df[:numinstances]

    def mezclar_palabras(texto):
        palabras = texto.split()
        random.shuffle(palabras)
        return ' '.join(palabras)

    df['text'] = df['text'].apply(mezclar_palabras)

    df.to_csv(pathRandomizedTest, index=False)

