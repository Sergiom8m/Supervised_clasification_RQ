import predictInstances,traduceAndRandomize
import sys
from tqdm import tqdm

if __name__ == '__main__':

    numInstances = int(sys.argv[1])
    testpath = '../formated/test.csv'

    # TRADUCED INSTANCES
    languages = ['en','es','fr','it','ca','eu','bg','hy','ka','ug']
    for language in tqdm(languages, desc='Procesando idiomas'):

        traducedPath = f'../traduced/test_{language}.csv'

        traduceAndRandomize.traduce(testpath, traducedPath, language, numInstances)
        
        predictInstances.predict(traducedPath, language, numInstances)

    #RANDOMIZED INSTANCES
    traduceAndRandomize.randomize(testpath,'../traduced/randomized_test.csv', numInstances)
    predictInstances.predict('../traduced/randomized_test.csv', 'rndm', numInstances)

    