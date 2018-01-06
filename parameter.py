from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn import neighbors
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def read_data(split):
    # imdb_score
    # df = pd.read_csv("one_hot.csv",
    #                   usecols=[ 'num_critic_for_reviews', 'duration',
    #                            'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes',
    #                            'gross', 'num_voted_users', 'num_user_for_reviews', 'cast_total_facebook_likes',
    #                            'budget',  'imdb_score'])
    # df = pd.read_csv("to.csv")
    # Y = df['vote_average']
    # X = df.drop(['vote_average'], axis=1)

    # df = pd.read_csv("input/one_hot_gc.csv")
    # df = df.drop(['content_rating'], axis=1)
    # Y = df['imdb_score']
    # X = df.drop(['imdb_score'], axis=1)

    df = pd.read_csv("2.csv")
    Y = df['score']
    X = df.drop(['score'], axis=1)
    print(X)


    x_columns = X.columns
    indexs = np.arange(0, len(X))
    np.random.shuffle(indexs)
    X = X.values[indexs]
    Y = Y.values[indexs]

    X_scaled = preprocessing.scale(X)

    train_ = [X_scaled[:split], Y[:split]]
    test_ = [X_scaled[split:], Y[split:]]

    print(X_scaled.shape, Y.shape)
    data = np.concatenate([X_scaled, Y.reshape([-1, 1])], axis=1)
    data = pd.DataFrame(data, columns=list(x_columns) + ['imdb_score'])
    # data.describe().T.to_csv('desc_scaled.csv')
    # exit()
    return train_, test_
def plot_(pred, real):
    yy_pred = pred
    yy_real = real
    def sample(y):
        import random
        index=random.sample(range(len(y)),30)
        return index

    index = sample(yy_pred)

    y_pred_sample = yy_pred[index]
    y_real_sample = yy_real[index]

    samples = zip(y_real_sample,y_pred_sample)
    samples = sorted(samples)
    # samples = sorted(samples,key=lambda samples : samples[1])

    y_real_sample_sorted = []
    y_pred_sample_sorted = []
    for sample in samples:
        y_real_sample_sorted.append(sample[0])
        y_pred_sample_sorted.append(sample[1])
    # print(y_real_sample_sorted)
    # print(y_pred_sample_sorted)

    plt.figure()
    plt.plot(range(len(y_real_sample_sorted)),y_pred_sample_sorted,'b',label="predict")
    plt.plot(range(len(y_real_sample_sorted)),y_real_sample_sorted,'r',label="real")
    plt.legend(loc="upper right")  #  显示图中的标签
    plt.xlabel("movies")
    plt.ylabel('vote average')
    plt.show()


def main():
    print("------------------------------")
    cmp_model_list = ['Linear Regression','SVR','GBDT','KNN','ETR']#, 'Lasso','Linear Regression', 'SVR', 'GBDT', 'KNN',
    print('start!')
    tran_test_split_ = 2200
    train_, test_ = read_data(tran_test_split_)

    print('read finish!')

    model_lr = LinearRegression()
    model_lr.fit(train_[0], train_[1])
    print('LinearRegression Training Finish')

    model_svr = SVR()
    model_svr.fit(train_[0], train_[1])
    print('SVR Training Finish')

    model_gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model_gbdt.fit(train_[0], train_[1])
    print('GBDT Training Finish')
    #
    model_knn = neighbors.KNeighborsRegressor()
    model_knn.fit(train_[0], train_[1])
    print('KNN Training Finish')

    # model_lasso = Lasso()
    # model_lasso.fit(train_[0], train_[1])
    # print('Lasso Training Finish')

    model_etr = ExtraTreesRegressor()
    model_etr.fit(train_[0], train_[1])
    print('ETR Training Finish')

    lr_train_pred = model_lr.predict(train_[0])
    lr_test_pred = model_lr.predict(test_[0])
    svr_train_pred = model_svr.predict(train_[0])
    svr_test_pred = model_svr.predict(test_[0])
    gbdt_train_pred = model_gbdt.predict(train_[0])
    gbdt_test_pred = model_gbdt.predict(test_[0])
    knn_train_pred = model_knn.predict(train_[0])
    knn_test_pred = model_knn.predict(test_[0])
    # lasso_train_pred = model_lasso.predict(train_[0])
    # lasso_test_pred = model_lasso.predict(test_[0])
    etr_train_pred = model_etr.predict(train_[0])
    etr_test_pred = model_etr.predict(test_[0])


    for i, pred in enumerate([
        [lr_train_pred, lr_test_pred],
        [svr_train_pred, svr_test_pred],
        [gbdt_train_pred, gbdt_test_pred],
        [knn_train_pred, knn_test_pred],
        # [lasso_train_pred, lasso_test_pred],
        [etr_train_pred, etr_test_pred]
    ]):
        print("------------------------------")
        print("-Train-")
        print("Selected Model: %s" % cmp_model_list[i])
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(train_[1], pred[0]))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f'
              % r2_score(train_[1], pred[0]))
        print("-Test-")
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(test_[1], pred[1]))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f'
              % r2_score(test_[1], pred[1]))
    print("------------------------------")
    # plot_(knn_test_pred, test_[1])




if __name__ == '__main__':
    main()
