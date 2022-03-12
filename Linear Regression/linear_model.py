#########################
#      Exercise 2       #
# Name: Adam Shtrasner  #
# ID: 208837260         #
#########################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######
# Q9 #
######


def fit_linear_regression(X, y):
    """
    :param X: a design matrix
    :param y: a response vector
    :return: 2 values:
               * numpy array of the coefficients vector w
               * numpy array of the singular values of X
    """
    w = np.linalg.pinv(X) @ y
    S = np.linalg.svd(X, compute_uv=False)
    return w, S

#######
# Q10 #
#######


def predict(X, w):
    """
    :param X: a design matrix
    :param w: coefficients vector
    :return: a numpy array with the predicted values by the model
    """
    return X @ w

#######
# Q11 #
#######


def mse(y, y_hat):
    """
    :param y: a response vector
    :param y_hat: a prediction vector
    :return: the MSE over the received samples
    """
    return np.mean((y-y_hat)**2)

#############
# Q12 + Q13 #
#############


def load_data(csv_path):
    """
    The function loads the dataset given and performs preprocessing
    :param csv_path: path to the data set
    :return: a tuple:
              * a design matrix
              * a response vector
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().dropna()

    # Adding the intercept
    df.insert(0, "intercept", 1, True)

    # Making sure feature's values are valid
    for feature in ["sqft_living", "sqft_lot", "floors",
                    "sqft_above", "sqft_basement"]:
        df = df[df[feature] >= 0]

    for feature in ["price", "bedrooms", "bathrooms", "yr_built", "yr_renovated"]:
        df = df[df[feature] > 0]

    # Making sure there are no outliers
    df = df[df["view"].isin(range(5)) & df["condition"].isin(range(1, 6)) & df["grade"].isin(range(1, 14))]

    # Dealing with categorical features:
    # (1) Changing "yr_built" column into decade and century dummy column
    # (2) Changing "yr_renovated" to a new column - 1 for renovated, 0 for not renovated
    # (3) Removing id, date, let, long columns
    df["dec_cent_built"] = (df["yr_built"] / 10).astype(int)
    df = pd.get_dummies(df, prefix="decade_century_built", columns=["dec_cent_built"])

    df["is_renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)
    df = df.drop("yr_renovated", 1)
    df = pd.get_dummies(df, prefix="zip_code", columns=["zipcode"])

    for feature in ["id", "date", "lat", "long"]:
        df = df.drop(feature, 1)

    return df.drop("price", 1), df.price

#######
# Q14 #
#######


def plot_singular_values(sv):
    """
    The function plots the singular values of a design matrix
    :param sv: a numpy array of singular values of some design matrix
    """
    plt.plot(range(len(sv)), sv, "b.-", linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("")
    plt.ylabel("Singular Values")
    plt.legend(["Singular Values"], loc='best', borderpad=0.3,
               shadow=False, markerscale=0.5)
    plt.show()

##########
# Q15-17 #
##########


def feature_evaluation(X, y):
    """
    The function plots for every non-categorical feature, a scatter plot of the feature
    values and the response values, and computes and shows on the graph the pearson
    correlation between the feature and the response.
    :param X: The design matrix
    :param y: The response vector
    """

    for feature in X:
        sigma_feature = np.std(X[feature])
        sigma_y = np.std(y)
        corr = np.cov(X[feature], y)[0, 1] / (sigma_feature * sigma_y)

        # Scatter plot of the feature values and the response values
        plt.scatter(X[feature], y, marker=".")
        plt.title("Correlation between " + feature + " values and the response values\n"
                                                    "Pearson Correlation = " + "{:.4f}".format(corr))
        plt.xlabel(feature + "Values")
        plt.ylabel("Response Values")
        m, b = np.polyfit(X[feature], y, 1)
        plt.plot(X[feature], m * X[feature] + b, 'r--', linewidth=2, markersize=12)
        plt.show()


def fit_model(train_x, train_y, test_x, test_y):
    """
    The function fits a linear regression model over a training set,
    and tests the performance of the fitted model on the test-set.
    :param train_x: training sample
    :param train_y: training response
    :param test_x: test sample
    :param test_y: test response
    :return: list of the MSE's over the test sets according to p for every p in 1,...,100.
    """
    test_set = list()
    for p in range(1, 101):
        n = max(round(len(train_y) * (p / 100)), 1)
        w, S = fit_linear_regression(train_x[:n], train_y[:n])
        test_set.append(mse(test_y, predict(test_x, w)))
    return test_set


def train_and_test(X, y):
    """
    The function splits the data into train and test sets randomly.
    :param X: Design matrix
    :param y: Response vector
    :return: the fitted model using the fit_model function
    """
    X = X.sample(frac=0.25, random_state=1)
    test_x = X
    test_y = y.reindex_like(test_x)

    X = X.sample(frac=1)
    train_x = X
    train_y = y.reindex_like(train_x)

    return fit_model(train_x, train_y, test_x, test_y)


def plot_mse_test_set(results):
    """
    The function plots The MSE over the test set as a function of p, according tio the results.
    :param results: The list of MSE's over the test sets given by the fit_model function
    """
    plt.plot(range(1, 101), results, "b.-", linewidth=2)
    plt.title("MSE over the test set as a function of p%")
    plt.xlabel("First percentage of the training set")
    plt.ylabel("MSE over the test set")
    plt.show()


if __name__ == "__main__":
    X, y = load_data("kc_house_data.csv")

    X = X.loc[:, ~X.columns.str.startswith("zip_code") & ~X.columns.str.startswith("decade_century_built")]
    X = X.drop("intercept", 1)

    w, S = fit_linear_regression(X, y)
    plot_singular_values(S)
    feature_evaluation(X, y)

    results = train_and_test(X, y)
    plot_mse_test_set(results)
