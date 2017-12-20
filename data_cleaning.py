# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer,StandardScaler



#count zero number=============================
def count_zero(object_name):
    objects = meta_df[object_name]
    x = 0
    for object in objects:
        if object == 0:
            x = x+1
    print(object_name+ ' zero_num: %d' % x)

def print_count_zero():
    for a in numerical_columns:
        count_zero(a)
# print_count_zero()


#print count nan=============================================
def count_nan(object_name):
    objects = meta_df[object_name].fillna('XXXX')
    x = 0
    for object in objects:
        if object != 'XXXX':
            x = x+1
    print(object_name + ': %d' %x)

def print_count_nan():
    for a in df_columns:
        count_nan(a)
# print_count_nan()


#fill all zeros in all numerical columns================================
def fill_zero(object_name):
    numerical_data[object_name] = numerical_data[object_name].apply(lambda x: numerical_data[object_name].mean() if x==0 else x)

def fill_all_zero():
    for a in numerical_columns:
        fill_zero(a)


meta_df = pd.read_csv("movie_metadata.csv"
                      # ,usecols=['director_name', 'director_facebook_likes', 'num_critic_for_reviews', 'duration',
                      #          'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes',
                      #          'actor_1_facebook_likes', 'actor_3_name', 'actor_2_name', 'actor_1_name', 'genres',
                      #          'gross', 'num_voted_users', 'num_user_for_reviews', 'cast_total_facebook_likes',
                      #          'country', 'content_rating', 'budget', 'plot_keywords', 'imdb_score']
                      )

df_columns = ['color', 'director_name', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
       'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
       'movie_title', 'num_voted_users', 'cast_total_facebook_likes',
       'actor_3_name', 'facenumber_in_poster', 'plot_keywords',
       'movie_imdb_link', 'num_user_for_reviews', 'language', 'country',
       'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',
       'imdb_score', 'aspect_ratio', 'movie_facebook_likes']

numerical_data = meta_df.select_dtypes(exclude=["object"])
numerical_columns = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
       'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
       'num_voted_users', 'cast_total_facebook_likes',
       'facenumber_in_poster', 'num_user_for_reviews', 'budget',
       'title_year', 'actor_2_facebook_likes', 'imdb_score',
       'aspect_ratio', 'movie_facebook_likes']

information_data = meta_df.select_dtypes(include=["object"])
information_columns = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name',
       'movie_title', 'actor_3_name', 'plot_keywords', 'movie_imdb_link',
       'language', 'country', 'content_rating']



# 将数值型空值的填充为平均值,并执行填充0值函数，转移到meta_df中============================================
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)  # default values
numerical_data[numerical_columns] = imp.fit_transform(numerical_data[numerical_columns])
fill_all_zero()#还在numerical_data
for C in numerical_columns:#将numerical_data中数据转到meta_df中
    meta_df[C] = numerical_data[C]


# 将文本型的空值填充为上一条数据，转移到meta_df中============================================
information_data = pd.DataFrame(information_data)
information_data = information_data.fillna(method='pad')
for C in information_columns:#将information_data转到meta_df中
    meta_df[C] = information_data[C]


pd.DataFrame.to_csv(meta_df, 'data_cleaning.csv')