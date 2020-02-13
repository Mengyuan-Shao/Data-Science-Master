#coding=utf-8
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(12345)

# Function for loading the iris data
# load_data returns a 2D numpy array where each row is an example
#  and each column is a given feature.
def load_data():
    iris = datasets.load_iris()
    return iris.data

# calculate the distance between two points that they are multi-dimensions.
def Distance(p1,p2):
    tmp = 0
    for q in range(len(p1)):
        tmp += np.sum((p1[q]-p2[q]) ** 2)
    return np.sqrt(tmp)


# Assign labels to each example given the center of each cluster
def assign_labels(X, centers):
    # create a array that contain len(centers) of empty elements.
    labels = [[] for a in range(len(centers))]

# calculate the minimam distance over each point in X to every centoid.
    for i in X:
        dis = []
        for j in centers:
        #for each i, it calculates the distance between i and center points.
        # store disrance to dis. 
            distance = Distance(i, j)
            # distance = np.sqrt(sum(np.power((i - j), 2))
            dis.append(distance)
        #choose the point which is smallest distance to store in lables[n].
        for n in range(len(dis)):
# label the point which the distacen is minimam to 1, otherwise label the point to 0.
            if (dis[n] == min(dis)):
                labels[n].append(1)
            elif (dis[n] != min(dis)):
                labels[n].append(0)
    
    a = np.array(labels)
    position = np.argmax(a, axis=1)
    return a
center = [[4.6, 3.2, 1.4, 0.2], [7.7, 3.8, 6.7, 2.2], [5.8, 2.7, 4.1, 1. ], [6.3, 2.9, 5.6, 1.8], [4.9, 3.1, 1.5, 0.1], [5.1, 3.7, 1.5, 0.4]]
x = load_data()
# lab = assign_labels(x, center)


# Calculate the center of each cluster given the label of each example
def calculate_centers(X, labels):
#calculating distance within each 4-D point to centroids.
    # X[lables == 2].mean() #calculate the mean value.
   
    # according to different value in labels to know which cluster each point 
    # belong to, store these points ot new_label, calculate the mean value of 
    # each cluster. The new_label will be resetted in each loop.
    centers = []
    for i in labels:
        new_label = []
        for biary in range(len(i)):
            if (i[biary] == 1):
                new_label.append(X[biary])
        new_cent = np.mean(new_label, axis= 0)
        # print(new_cent)
        # store new centriod to centers.
        centers.append(new_cent)
    return centers
# calculate_centers(x, lab)


# Test if the algorithm has converged
# Should return a bool stating if the algorithm has converged or not.
def test_convergence(old_centers, new_centers):
# one assign completed. 
    for i in range(len(old_centers)):
        # If old centers and new centers are same, return true.
        for j in range(len(new_centers[0])):
            if (old_centers[i][j] == new_centers[i][j]):
                return True
            else:
                return False

# Evaluate the preformance of the current clusters
# This function should return the total mean squared error of the given clusters
def evaluate_performance(X, labels, centers):
    Sum = 0
    # calculate the distance among each point in X to their centriod.
    for i in range(len(labels)):
        new_contain = []
        label = labels[i]
        for value in range(len(label)):
            if (label[value] == 1):
                new_contain.append(X[value])
        for every_element in new_contain:
            Sum += Distance(every_element, centers[i])
    # print(Sum)
    error = Sum / len(X)
    print(error)
    return error
# evaluate_performance(x, lab, new_cent)
# evaluate_performance(x, lab, center)




# Algorithm for preforming K-means clustering on the give dataset
def k_means(X, K):

#initinal centriods from 1 to 10 randomly.
    init_centers = []
    np.random.seed(0)
    for n in range(K):
        init_center = X[np.random.randint(0, len(X))]
        init_centers.append(init_center)
    old_centers = init_centers

    loop = False
    #It won't stop updating centriods until function test_convergence is true.
    while (loop == False):
        new_labels = assign_labels(X, old_centers)
        new_centers = calculate_centers(X, new_labels)
        loop = test_convergence(old_centers, new_centers)
        old_centers = new_centers

    result = evaluate_performance(X, new_labels, old_centers)
    # plt.scatter(np.asarray(new_centers))
    # plt.show()
    # print(result)
    return result


def main():
    dataset = load_data()
    y = []
    x = range(1, 10)
    # perform function k_means over different k.
    for i in x:
        performance = k_means(dataset, i)
        y.append(performance)
    # plt.plot(i, performance)
    plt.plot(x, y)
    plt.show()
    
if __name__ == "__main__":
    main()


# get path
# data = pd.read_csv('/Users/shaomengyuan/Desktop/balance-scale.data.numbers')
# print(data)
# np.argmin(dis) get minimum value from matrix
# X[lables == 2].mean() calculate the mean value.