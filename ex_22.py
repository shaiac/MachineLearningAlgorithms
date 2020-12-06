# Shai Acoca 315314278
import numpy as np
import sys
from scipy.spatial import distance
import heapq
from sklearn.utils import shuffle

from sklearn.model_selection import KFold


# Function for All
def checkPerceptron(array_x, array_y):
        cv = KFold(n_splits=5, random_state=89, shuffle=True)
        trainX = []
        trainY = []
        testX = []
        testY = []
        Bias = 0.3
        sum_preceptron = 0
        sum_pa = 0
        for k in range(0, 5):
            for train_index, test_index in cv.split(array_x, array_y):
                for i in train_index:
                    trainX.append(array_x[i])
                    trainY.append(array_y[i])
                for j in test_index:
                    testX.append(array_x[j])
                    testY.append(array_y[j])
            perceptron_after_test = perceptron_pa_train(trainX, trainY, True)
            pa_after_test = perceptron_pa_train(trainX, trainY, False)
            length = len(testX)
            count_preceptron = 0
            count_pa = 0
            for i in range(0, length):
                if perceptron_pa_test_iteration(perceptron_after_test, testX[i]) == testY[i]:
                    count_preceptron += 1
                if perceptron_pa_test_iteration(pa_after_test, testX[i]) == testY[i]:
                    count_pa += 1
            sum_preceptron += (count_preceptron / length) * 100
            sum_pa += (count_pa / length) * 100

        print("Bias : {0}".format(Bias))
        print("Preceptron: {0} %".format(sum_preceptron / 5))
        print("PA: {0} %".format(sum_pa / 5))
        print()
# Class the has all the information about the wines that we get as a training set
class TrainingValues:
    training_set = None
    data_set = None
    examples_labels = None
    wines_labels_dic = {}

    def __init__(self):
        self.wines, self.labels = self.read_training_set()
        self.maximum_vector, self.minimum_vector = np.max(self.wines, axis=0), np.min(self.wines, axis=0)
        self.normalize_features_values()
        self.split_to_examples_and_training_sets(30)
        self.init_wines_labels_dic()
        self.wines_num, self.features_num = self.wines.shape

    def init_wines_labels_dic(self):
        index = 0
        for wine in self.wines:
            self.wines_labels_dic[wine.tobytes()] = self.labels[index]
            index += 1

    # Creating a single wine features vector.
    @staticmethod
    def create_vector(line):
        array = []
        split_line = line.strip('\n').split(',')
        for x in split_line:
            if x == 'W':
                array.append(1.0)
            elif x == 'R':
                array.append(0.0)
            else:
                array.append(x)
        return np.array(array)

    # Doing min max normalization to a float number.
    def min_max_normalization(self, feature_value, feature_index):
        min_value = self.minimum_vector[feature_index]
        max_value = self.maximum_vector[feature_index]
        return (feature_value - min_value) / (max_value - min_value) * (1 - 0) + 0

    # Iterating all the values of each feature to normalize them.
    def normalize_features_values(self):
        i = 0
        for wine in self.wines:
            j = 0
            for feature in wine:
                self.wines[i][j] = self.min_max_normalization(feature_value=feature, feature_index=j)
                j += 1
            i += 1

    # Reading the wines values from the input files.
    def read_training_set(self):
        train_x, train_y = [], []
        train_x_file_name, train_y_file_name = sys.argv[1], sys.argv[2]
        x_file, y_file = open(train_x_file_name, "r"), open(train_y_file_name, "r")
        for line in x_file:
            array = self.create_vector(line)
            train_x.append(array)
        for line in y_file:
            train_y.append(float(line.strip('\n')))
        return np.array(train_x).astype(np.float), np.array(train_y).astype(np.int)

    # Splitting the wines to training set (classification unknown) and test examples (classification known).
    def split_to_examples_and_training_sets(self, examples_num):
        # ex = np.random.choice(a=wines, size=examples_num, replace=False)
        num_rows, num_cols = self.wines.shape
        idx = np.random.randint(num_rows - 1, size=examples_num)
        self.data_set = self.wines[idx, :]
        self.examples_labels = self.labels[idx]
        self.training_set = np.delete(self.wines, idx, axis=0)


class KNN:
    def __init__(self, training_val, k):
        self.k = k
        self.training_values = training_val
        self.wines_labels_dic = {}

    # Finding the k nearest neighbours of a wine (the test examples that are
    # most similar to a wine from the training set)
    def find_k_nearest_neighbours(self, wine):
        d_example_dic = {}
        d_priority_queue = []
        max_d_in_queue = float('inf')
        example_index = 0
        for data in self.training_values.data_set:
            d = distance.euclidean(wine, data)
            if len(d_priority_queue) == self.k:
                if d > max_d_in_queue:
                    continue
                d_pop = -1 * heapq.heappop(d_priority_queue)
                try:
                    del d_example_dic[d_pop]
                except KeyError:
                    error = "distance already checked, ignore"
            heapq.heappush(d_priority_queue, -1 * d)
            d_example_dic[d] = example_index
            max_d_in_queue = -1 * heapq.heappop(d_priority_queue)
            heapq.heappush(d_priority_queue, -1 * max_d_in_queue)
            example_index += 1
        return d_example_dic

    def label_wine(self, wine, nearest_neighbours):
        labels_count_dic = {}
        for neighbour_index in nearest_neighbours.values():
            label = self.training_values.examples_labels[neighbour_index]
            if label in labels_count_dic:
                labels_count_dic[label] += 1
            else:
                labels_count_dic[label] = 1
        max_label = max(labels_count_dic, key=lambda k: labels_count_dic[k])
        self.wines_labels_dic[wine.tobytes()] = max_label

    def train(self):
        for wine in self.training_values.training_set:
            nearest_neighbours = self.find_k_nearest_neighbours(wine=wine)
            self.label_wine(wine=wine, nearest_neighbours=nearest_neighbours)


class Perceptron:

    def __init__(self, training_val, learning_rate, epochs):
        self.training_values = training_val
        self.rate = learning_rate
        self.epochs = epochs
        self.weights = self.init_weights([0, 1, 2])

    def init_weights(self, labels_num):
        weights = []
        for _ in labels_num:
            weights.append(np.zeros(self.training_values.features_num))
        return np.array(weights)

    def train(self):
        for _ in range(self.epochs):
            x_train, y_train = shuffle(self.training_values.wines, self.training_values.labels, random_state=1)
            for wine, label in zip(x_train, y_train):
                y_hat = np.argmax(np.dot(self.weights, wine))
                if label != y_hat:
                    self.weights[label] = self.weights[label] + self.rate * wine
                    self.weights[y_hat] = self.weights[y_hat] - self.rate * wine


class PassiveAggressive:
    def __init__(self, training_val, learning_rate, epochs):
        self.training_values = training_val
        self.rate = learning_rate
        self.epochs = epochs
        self.weights = self.init_weights([0, 1, 2])

    def init_weights(self, labels_num):
        weights = []
        for _ in labels_num:
            weights.append(np.zeros(self.training_values.features_num))
        return np.array(weights)

    def cal_teo(self, y, y_hat, x):
        loss = np.argmax(0, 1 - np.dot(self.weights[y], x) + np.dot(self.weights[y_hat, x]))
        print(loss)

    def train(self):
        for _ in range(self.epochs):
            x_train, y_train = shuffle(self.training_values.wines, self.training_values.labels, random_state=1)
            for wine, label in zip(x_train, y_train):
                y_hat = np.argmax(np.dot(self.weights, wine))
                self.cal_teo(label, y_hat, wine)
                if label != y_hat:
                    self.weights[label] = self.weights[label] + self.rate * wine
                    self.weights[y_hat] = self.weights[y_hat] - self.rate * wine


def check_training(wines_labels_dic, training_output):
    true_count = 0
    false_count = 0
    for wine in training_output:
        w1 = training_output[wine]
        w2 = wines_labels_dic[wine]
        if w1 == w2:
            true_count += 1
        else:
            false_count += 1
    print("True:" + str(true_count) + "     False:" + str(false_count))


# The main function
if __name__ == "__main__":
    training_values = TrainingValues()
    # knn = KNN(training_val=training_values, k=10)
    # knn.run()
    # check_training(training_values.wines_labels_dic, knn.wines_labels_dic)
    # perceptron = Perceptron(training_val=training_values, learning_rate=0.1, epochs=100)
    # perceptron.train()
    # print(perceptron.weights)
    pa = PassiveAggressive(training_val=training_values, learning_rate=0.1, epochs=100)
    pa.train()
