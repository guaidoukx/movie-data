import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,StandardScaler
import warnings
warnings.filterwarnings('ignore')

movies = pd.read_csv("../input/movie_metadata.csv")
# print (movies.shape)
# print (movies.columns)

#get numeric data for computation and correlation purposes
numerical_data = movies.select_dtypes(exclude=["object"])
# print(numerical_data.columns)


score_imdb= numerical_data["imdb_score"]
numerical_data = numerical_data.drop(["imdb_score"],axis=1)
year_category = numerical_data["title_year"]
numerical_data = numerical_data.drop(["title_year"],axis=1)
numerical_columns = numerical_data.columns
sns.distplot(score_imdb, rug=True,label="IMDB Scores").legend()
# plt.show()


#fill missing values and normalize the data
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)      #default values
numerical_data[numerical_columns] = imp.fit_transform(numerical_data[numerical_columns])
# print (numerical_data.describe())


#Without StandardScaler, SVR with poly kernel will throw error of large size.
#With standard scaling, models has seen improvement in predicting.
#knn model is the most beneficiary of standard scaling
scaler = StandardScaler()
numerical_data[numerical_columns] = scaler.fit_transform(numerical_data[numerical_columns])

# print (numerical_data.describe())
# print (numerical_data.shape)
# numerical_data = pd.DataFrame(numerical_data)
# print (numerical_data.describe())
# print(numerical_data[numerical_columns])

#get non_numeric informational content
information_data = movies.select_dtypes(include=["object"])
# print (information_data.columns)

