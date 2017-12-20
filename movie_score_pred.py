from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn import neighbors
import numpy as np
import pandas as pd


def read_data(split):
    # imdb_score
    # df = pd.read_csv('4.csv')#original

    #changed===============
    df = pd.read_csv('one_hot_g.csv')
    df = df.drop(['content_rating'], axis=1)
    # df = df.drop(['num_critic_for_reviews'], axis=1)

    # df.describe().T.to_csv('desc.csv')
    # exit()

    Y = df['imdb_score']
    X = df.drop(['imdb_score'], axis=1)
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


def main():
    print("------------------------------")
    cmp_model_list = ['ETR']#, 'Lasso','Linear Regression', 'SVR', 'GBDT', 'KNN',
    print('start!')
    tran_test_split_ = 4000
    train_, test_ = read_data(tran_test_split_)

    print('read finish!')

    # model_lr = LinearRegression()
    # model_lr.fit(train_[0], train_[1])
    # print('LinearRegression Training Finish')
    #
    # model_svr = SVR()
    # model_svr.fit(train_[0], train_[1])
    # print('SVR Training Finish')
    #
    # model_gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    # model_gbdt.fit(train_[0], train_[1])
    # print('GBDT Training Finish')
    #
    # model_knn = neighbors.KNeighborsRegressor()
    # model_knn.fit(train_[0], train_[1])
    # print('KNN Training Finish')

    # model_lasso = Lasso()
    # model_lasso.fit(train_[0], train_[1])
    # print('Lasso Training Finish')

    model_etr = ExtraTreesRegressor()
    model_etr.fit(train_[0], train_[1])
    print('ETR Training Finish')

    # lr_train_pred = model_lr.predict(train_[0])
    # lr_test_pred = model_lr.predict(test_[0])
    # svr_train_pred = model_svr.predict(train_[0])
    # svr_test_pred = model_svr.predict(test_[0])
    # gbdt_train_pred = model_gbdt.predict(train_[0])
    # gbdt_test_pred = model_gbdt.predict(test_[0])
    # knn_train_pred = model_knn.predict(train_[0])
    # knn_test_pred = model_knn.predict(test_[0])
    # lasso_train_pred = model_lasso.predict(train_[0])
    # lasso_test_pred = model_lasso.predict(test_[0])
    etr_train_pred = model_etr.predict(train_[0])
    etr_test_pred = model_etr.predict(test_[0])


    for i, pred in enumerate([
        # [lr_train_pred, lr_test_pred],
        # [svr_train_pred, svr_test_pred],
        # [gbdt_train_pred, gbdt_test_pred],
        # [knn_train_pred, knn_test_pred],
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


if __name__ == '__main__':
    main()
