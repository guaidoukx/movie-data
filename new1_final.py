# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np

# meta_df = pd.read_csv("movie_metadata.csv",#原始的
#                       usecols=['director_name', 'director_facebook_likes', 'num_critic_for_reviews', 'duration',
#                                'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes',
#                                'actor_1_facebook_likes', 'actor_3_name', 'actor_2_name', 'actor_1_name', 'genres',
#                                'gross', 'num_voted_users', 'num_user_for_reviews', 'cast_total_facebook_likes',
#                                'country', 'content_rating', 'budget', 'plot_keywords', 'imdb_score'])
meta_df = pd.read_csv("data_cleaning.csv",
                      usecols=['director_name',  'num_critic_for_reviews', 'duration',
                               'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes',
                               'actor_3_name', 'actor_2_name', 'actor_1_name', 'genres',
                               'gross', 'num_voted_users', 'num_user_for_reviews', 'cast_total_facebook_likes',
                               'country', 'content_rating', 'budget', 'plot_keywords', 'imdb_score'])

# def get_onehot():
#     directors = set(meta_df["director_name"])
#
#     director_num = len(directors)
#     director_dict = {}
#     for director_rank, director in enumerate(directors):
#         zero_vector = np.zeros(shape=director_num)
#         zero_vector[director_rank] = 1
#         director_dict[director] = zero_vector
#
#     meta_df["director_name"] = meta_df["director_name"].apply(lambda x: director_dict[x])


def string_2_vec(column_name="director_name"):
    objects = set(meta_df[column_name])
    object_num = len(objects)
    object_dict = {}
    for object_rank, object in enumerate(objects):
        object_vector = np.zeros(shape=object_num)
        object_vector[object_rank] = 1
        object_dict[object] = object_vector

    meta_df[column_name] = meta_df[column_name].apply(lambda x: object_dict[x])


# genres = []
# genre_dict = {}
# for movie_genres in meta_df["genres"]:
#     genres.extend(movie_genres.split("|"))
# genres = set(genres)
# genre_num = len(genres)
# for genre_rank, genre in enumerate(genres):
#     zero_vector = np.zeros(shape=genre_num)
#     zero_vector[genre_rank] = 1
#     genre_dict[genre] = zero_vector
#
# genres_to_vectors = []
# for movie_genres in meta_df["genres"]:
#     genres = movie_genres.split("|")
#     genre_vec = sum([genre_dict[genre] for genre in genres])
#     genres_to_vectors.append(genre_vec)
# meta_df["genres"] = genres_to_vectors


def word_to_vec(column_name='genres'):
    object_list = []
    object_dict = {}
    objects = meta_df[column_name]
    for object in objects:
        object_list.extend(str(object).split('|'))
    object_list = set(object_list)
    object_num = len(object_list)
    for object_rank, object in enumerate(object_list):
        zero_vector = np.zeros(shape=object_num)
        zero_vector[object_rank] = 1
        object_dict[object] = zero_vector

    object_to_vectors = []
    for object in objects:
        object_list = str(object).split('|')
        object_vec = sum(object_dict[object] for object in object_list)
        object_to_vectors.append(object_vec)
    meta_df[column_name] = object_to_vectors


# df = pd.DataFrame()
# for i_rank, i in enumerate(meta_df["genres"]):
#     df[i_rank] = i
# df = df.T
# df.columns = ["genre{}".format(i+1) for i in range(26)]
# df = pd.concat([meta_df, df], axis=1)
# print df.shape


def split_vector(column_name='genres'):
    objects = meta_df[column_name]
    df = pd.DataFrame()
    for object_rank, object in enumerate(objects):
        df[object_rank] = object
    lenth = len(df)
    df = df.T
    df.columns = [column_name+'{}'.format(i+1) for i in range(lenth)]
    df = pd.concat([meta_df, df], axis=1)
    return df

string_2_vec('director_name')
string_2_vec('actor_1_name')
string_2_vec('actor_2_name')
# string_2_vec('actor_3_name')
string_2_vec('country')
word_to_vec('genres')
# word_to_vec('plot_keywords')

meta_df = split_vector('director_name')
meta_df = split_vector('actor_1_name')
meta_df = split_vector('actor_2_name')
# meta_df = split_vector('actor_3_name')
meta_df = split_vector('country')
meta_df = split_vector('genres')
# meta_df = split_vector('plot_keywords')
del meta_df['director_name']
del meta_df['actor_1_name']
del meta_df['actor_2_name']
del meta_df['actor_3_name']
del meta_df['country']
del meta_df['genres']
del meta_df['plot_keywords']
print(meta_df.shape)
pd.DataFrame.to_csv(meta_df, 'one_hot_dcgaa.csv', index=None)



