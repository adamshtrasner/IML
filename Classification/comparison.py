#########################
#      Exercise 3       #
# Name: Adam Shtrasner  #
# ID: 208837260         #
#########################


from models import *
import matplotlib.pyplot as plt


def draw_points(m):
    """
    :param m: an integer
    :return: a pair X, y where X is m x 2 matrix where each column represents an i.i.d sample from the
    standard normal distribution, and y in {-1,1} is its corresponding label, according to f.
    """
    X = np.random.randn(m, 2)
    w = np.array([0.3, -0.5])
    while np.all(np.sign(X @ w + 0.1) > 0) or np.all(np.sign(X @ w + 0.1) < 0):
        X = np.random.randn(m, 2)
    y = np.sign(X @ w + 0.1)
    return X, y


def plot_m_points():
    f, ax = plt.subplots(2, 3, figsize=(15, 20))

    for i, m in enumerate([5, 10, 15, 25, 70]):
        X, y = draw_points(m)

        perceptron = Perceptron()
        perceptron.fit(X, y)

        svm = SVM()
        svm.fit(X, y)

        ax.flatten()[i].set_title("m = {}".format(m))

        ax.flatten()[i].scatter(X[y > 0, 0], X[y > 0, 1])
        ax.flatten()[i].scatter(X[y < 0, 0], X[y < 0, 1])

        ax.flatten()[i].plot([-2, 2], [-1, 1.4], label="f", color="green")
        ax.flatten()[i].plot([-2, 2], [
            -2 * perceptron.w[0] / -perceptron.w[1] + perceptron.w[2] / -perceptron.w[1],
            2 * perceptron.w[0] / -perceptron.w[1] + perceptron.w[2] / -perceptron.w[1]],
                             label="Perceptron", color="blue")
        ax.flatten()[i].plot([-2, 2], [
            -2 * svm.coef_[0, 0] / -svm.coef_[0, 1] + svm.intercept_ / -svm.coef_[0, 1],
            2 * svm.coef_[0, 0] / -svm.coef_[0, 1] + svm.intercept_ / -svm.coef_[0, 1]],
                             label="SVM", color="red")

        ax.flatten()[i].legend()

    plt.show()


def test_train_and_compare():
    acc_perc = np.zeros(5)
    acc_svm = np.zeros(5)
    acc_lda = np.zeros(5)
    k = 10000
    for n in range(500):
        for i, m in enumerate([5, 10, 15, 25, 70]):
            X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(k)

            perc = Perceptron()
            perc.fit(X_train, y_train)

            svm = SVM()
            svm.fit(X_train, y_train)

            lda = LDA()
            lda.fit(X_train, y_train)

            acc_perc[i] += perc.score(X_test, y_test)["accuracy"]
            acc_svm[i] += svm.score(X_test, y_test)["accuracy"]
            acc_lda[i] += lda.score(X_test, y_test)["accuracy"]

    plt.plot([5, 10, 15, 25, 70], acc_perc, label="Perceptron", color="blue",)
    plt.plot([5, 10, 15, 25, 70], acc_svm, label="SVM", color="red")
    plt.plot([5, 10, 15, 25, 70], acc_lda, label="LDA", color="green")
    
    plt.xlabel("Number of samples - m")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()


# Question 9
plot_m_points()

# Question 10
test_train_and_compare()
