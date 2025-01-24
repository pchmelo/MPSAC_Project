import urllib.request
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from sklearn.neighbors import KNeighborsClassifier

t_end = 100  # Set this to the appropriate value for your use case

### Control part
def my_controller(c, b, x, xd, xd_dot):
    k = 1  # Tuning parameter
    if x > -0.7:
        u = -k*(x - xd) + xd_dot + c*abs(x)*x + b
    else:
        u = max(-k * (x - xd), 0)

    u = max(min(u, 1), -1)
    return u

def set_signals(t_signal, xd_signal, xd_dot_signal, X, Y):
    for t in range(0, len(t_signal)):
        if (t_signal[t] >= 0 and t_signal[t] < X) or (t_signal[t] >= 50 and t_signal[t] < Y):
            xd_signal[t] = 0.75
            xd_dot_signal[t] = 0.0
        elif (t_signal[t] >= X and t_signal[t] < 50) or (t_signal[t] >= Y):
            xd_signal[t] = -0.75
            xd_dot_signal[t] = 0.0
    return xd_signal, xd_dot_signal

### Machine Learning part
def myKNN(K, trainX_data, trainY_data, new_pt):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(trainX_data, trainY_data)
    new_pt_reshaped = new_pt.reshape(1, -1)
    majority_label = knn.predict(new_pt_reshaped)
    return majority_label[0]

h = 0.01  # Sample time
c = 0.1
b = 0.65

# Discrete system
def disc_sys(x, u, h, c, b):
    if x > -0.7:
        return x + (-c*abs(x)*x - b + u + np.random.normal(-0.05, 0.2)) * h
    else:
        return x + max(0, u) * h

def main_loop(X_tested, K, Y_tested):
    # Signals
    t_signal = np.arange(0, t_end, h)  # time samples
    x_signal = np.zeros_like(t_signal)
    u_signal = np.zeros_like(t_signal)
    xd_signal, xd_dot_signal = set_signals(t_signal, np.zeros_like(t_signal), np.zeros_like(t_signal), X_tested, Y_tested)

    # Initial condition of our system
    x_signal[0] = 0

    np.random.seed(10)
    # Iteration
    for t in range(0, len(t_signal) - 1):
        u_signal[t] = my_controller(c, b, x_signal[t], xd_signal[t], xd_dot_signal[t])
        u_signal[t] = np.clip(u_signal[t], -1, 1)
        x_signal[t + 1] = disc_sys(x_signal[t], u_signal[t], h, c, b)

    dataTrain = pd.read_csv('data-set1.csv', index_col=0)

    X_1 = dataTrain.iloc[0:100, np.arange(0, 2, 1)].values
    X_2 = dataTrain.iloc[100:200, np.arange(0, 2, 1)].values
    X = dataTrain.iloc[:, np.arange(0, 2, 1)].values
    Y = dataTrain["label"].values

    dataTest = pd.read_csv('data-set2.csv', index_col=0)
    dataYTest = pd.read_csv('data-set3.csv', index_col=0)

    testX = dataTest.iloc[:, np.arange(0, 2, 1)].values

    np.random.seed(123)
    energy_R1 = 0
    count_class1_R1, count_class2_R1 = 0, 0
    for t in np.arange(0, 5000):
        if t_signal[t] < 50 or (t_signal[t] >= 60 and t_signal[t] < 80):
            if x_signal[t] <= 0.8 and x_signal[t] >= 0.6:
                count_class2_R1 += 1
            if x_signal[t] <= -0.6 and x_signal[t] >= -0.8:
                count_class1_R1 += 1
            energy_R1 = energy_R1 + u_signal[t]**2
    count_class1_R1 = int(count_class1_R1 * h)
    count_class2_R1 = int(count_class2_R1 * h)

    energy_R2 = 0
    count_class1_R2, count_class2_R2 = 0, 0
    for t in np.arange(6000, 8000):
        if t_signal[t] < 50 or (t_signal[t] >= 60 and t_signal[t] < 80):
            if x_signal[t] <= 0.8 and x_signal[t] >= 0.6:
                count_class2_R2 += 1
            if x_signal[t] <= -0.6 and x_signal[t] >= -0.8:
                count_class1_R2 += 1
            energy_R2 = energy_R2 + u_signal[t]**2
    count_class1_R2 = int(count_class1_R2 * h)
    count_class2_R2 = int(count_class2_R2 * h)

    iter = 5
    error_clas_R2 = np.zeros(iter)
    error_clas_R1 = np.zeros(iter)
    for j in range(iter):
        testY = dataYTest["label"].values
        random_indices_X_1 = np.random.randint(0, X_1.shape[0], count_class1_R1)
        random_indices_X_2 = np.random.randint(0, X_2.shape[0], count_class2_R1)
        X_train = np.vstack((X_1[random_indices_X_1], X_2[random_indices_X_2]))
        Y_train = np.concatenate((np.ones(count_class1_R1), 2 * np.ones(count_class2_R1)))

        for i in np.arange(0, 1000):
            result = myKNN(K, X_train, Y_train, testX[i])
            if result != testY[i]:
                error_clas_R1[j] += 1

        testY = dataYTest["label"].values
        random_indices_X_1 = np.random.randint(0, X_1.shape[0], count_class1_R2 + count_class1_R1)
        random_indices_X_2 = np.random.randint(0, X_2.shape[0], count_class2_R2 + count_class2_R1)
        X_train = np.vstack((X_1[random_indices_X_1], X_2[random_indices_X_2]))
        Y_train = np.concatenate((np.ones(count_class1_R1 + count_class1_R2), 2 * np.ones(count_class2_R1 + count_class2_R2)))

        for i in np.arange(1000, testX.shape[0]):
            result = myKNN(K, X_train, Y_train, testX[i])
            if result != testY[i]:
                error_clas_R2[j] += 1

    # Calculate and print the performance index J
    energy_R1 /= 100
    energy_R2 /= 100
    average_error_R1 = np.mean(error_clas_R1)
    average_error_R2 = np.mean(error_clas_R2)
    performance_index_J = 0.3 * average_error_R1 + 0.7 * average_error_R2 + 0.5 * (energy_R1 + energy_R2)
    print("Average Performance index J =", performance_index_J)
    print("X_tested =", X_tested)
    print("K =", K)
    print("Y_tested =", Y_tested)

    # Save the results to a CSV file
    results_df = pd.DataFrame({'X': [X_tested], 'Y': [Y_tested], 'K': [K], 'J': [performance_index_J]})
    results_df.to_csv('results.csv', mode='a', header=not os.path.exists('results.csv'), index=False)

    return performance_index_J

List_X = np.arange(14, 20, 0.1).tolist()
List_K = np.arange(1, 9, 2).tolist()
List_Y = np.arange(61, 71, 0.5).tolist()

List_X_teste =  [5, 10, 15, 20, 25]
k = 1


#teste mais pequeno
'''
for X_tested in List_X:
    for K in List_K:
        main_loop(X_tested, K, 70)
'''

#teste mais grande XD

for X_tested in List_X:
    for Y_tested in List_Y:
        main_loop(X_tested, 1, Y_tested)
