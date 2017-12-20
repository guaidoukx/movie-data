# %matplotlib inline
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate
import warnings; warnings.simplefilter('ignore')


# select some data and transform it to CSV=================
# md = pd. read_csv('movies_metadata.csv')
# data = md.iloc[:55,:]
# pd.DataFrame.to_csv(data,'test_1.csv', ',', '', None, None, True, False)


# test_data = pd.read_csv('test_1.csv')
# print(type(test_data['original_title'][30]))#select some item
# print(len(test_data.iloc[34,:]))#select some row
# print(test_data.iloc[:,3]) #select some column
#
# test_data = test_data.drop(1)#delete one row
# del test_data['adult']#delete one column



# delete some columns and are transferred to CSV
# del test_data['adult']
# del test_data['belongs_to_collection']
# del test_data['homepage']
# del test_data['poster_path']
# del test_data['video']
# pd.DataFrame.to_csv(test_data, 'form.csv', ',', '', None, None, True, False)


# data has been transferred to be cleaner
md = pd.read_csv('form.csv')
#
# md['genres'] = md['genres'].fillna('[]').apply(literal_eval).\
#     apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['production_companies'] = md['production_companies'].fillna('[]').apply(literal_eval).\
#     apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['production_countries'] = md['production_countries'].fillna('[]').apply(literal_eval).\
#     apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['spoken_languages'] = md['spoken_languages'].fillna('[]').apply(literal_eval).\
#     apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# del md['status']
#
# pd.DataFrame.to_csv(md, 'after.csv', ',', '', None, None, True, False)



# md = pd.read_csv('after.csv')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
# print(C)
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
m = vote_counts.quantile(0.95)
md['spoken_languages'] = md['spoken_languages'].fillna('[]').apply(literal_eval).\
    apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# print(md['year'])
# qualified = md[(md['vote_count'] >= 0) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())]\
#     [['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
# pd.DataFrame.to_csv(qualified, 'qualified.csv', ',', '', None, None, True, False)
print(md['spoken_languages'])