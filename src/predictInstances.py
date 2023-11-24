from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

def chunks(lista, batch_size):
    for i in range(0, len(lista), batch_size):
        yield lista [i:i+batch_size]
def predict(pathTest, language, numinstances):

    MODEL_PATH = '../models/'
    TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCHSIZE = 16

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)

    test = pd.read_csv(pathTest)
    labels = test['class'][:numinstances].to_list()
    tiny_texts = test['text'][:numinstances].to_list()

    outputs = []
    for batch in tqdm(chunks(tiny_texts, BATCHSIZE), total=int(len(tiny_texts) / BATCHSIZE)):
        test_tokens = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=512,
                                add_special_tokens=True).to(DEVICE)
        with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not update weigths
            outputs.append(model(**test_tokens)[0].detach().cpu())

    outputs=torch.cat(outputs)

    predictions = []
    
    correct = 0
    for index, output in enumerate(outputs):
        if output[0] > output[1]: predicted = 'suicide'
        else: predicted = 'non-suicide'

        predictions = predictions + [predicted]

        # print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

        if test["class"][index] == predicted: correct += 1

    with open(f'../results/report_{language}.txt', 'w') as file:

        file.write('################################\n')
        file.write(f'############### {language} ############# \n')
        file.write('################################\n\n')

        file.write('------ REPORT ------\n')
        
        print(classification_report(labels, predictions), file=file)

        file.write('\n------ CORRECTLY CLASSIFIED INSTANCES ------\n')

        print(str(correct) + ' / ' + str(numinstances), file=file)