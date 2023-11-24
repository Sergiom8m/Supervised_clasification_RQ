import predictInstances,traduceAndRandomize
import sys

if __name__ == '__main__':

    numInstances = int(sys.argv[1])
    testpath = '../formated/test.csv'

    # TRADUCED INSTANCES
    languages = ['en','es','fr','it','ca','eu','bg','hy','ka','ug']
    for language in languages:

        traducedPath = f'../traduced/test_{language}.csv'

        traduceAndRandomize.traduce(testpath, traducedPath, language, numInstances)
        
        predictInstances.predict(traducedPath, language, numInstances)

    # RANDOMIZED INSTANCES
    traduceAndRandomize.randomize(testpath,'randomized_test.csv', numInstances)
    predictInstances.predict('randomized_test.csv', 'en', numInstances)