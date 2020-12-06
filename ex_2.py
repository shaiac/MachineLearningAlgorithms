# Shai Acoca 315314278
import numpy as np
import sys
from scipy.spatial import distance
import heapq


# Class the has all the information about the wines that we get as a training set
class TrainingValues:
    training_set = None
    data_set = None
    examples_labels = None

    def __init__(self, normalization_type):
        self.train_x, self.train_y, self.test_x = self.read_training_set()
        self.maximum_vector, self.minimum_vector = np.max(self.train_x, axis=0), np.min(self.train_x, axis=0)
        self.normalize(normalization_type)
        self.split_to_examples_and_training_sets(30)
        # self.init_wines_labels_dic()
        self.x_num, self.features_num = self.train_x.shape

    # Normalizing the data according to the input type
    def normalize(self, type):
        if type == "zscore":
            self.train_x = self.zscore_normalization(self.train_x)
            self.test_x = self.zscore_normalization(self.test_x)
        elif type == "minmax":
            self.normalize_features_values(self.train_x)
            self.normalize_features_values(self.test_x)

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

    # Doing min max normalization to a float number in that case to the range of [0,1].
    def min_max_normalization(self, feature_value, feature_index):
        min_value = self.minimum_vector[feature_index]
        max_value = self.maximum_vector[feature_index]
        return (feature_value - min_value) / (max_value - min_value) * (1 - 0) + 0

    # Doing zscore normalization to a float number in that case to the range of [0,1].
    @staticmethod
    def zscore_normalization(x_set):
        avg = np.mean(x_set, axis=0)
        std = np.std(x_set, axis=0)
        trans_train = np.copy(x_set.T)
        for i in range(len(trans_train)):
            if std[i] == 0:
                trans_train[i] /= trans_train[i]
                continue
            trans_train[i] = (trans_train[i] - avg[i]) / std[i]
        return trans_train.T

    # Iterating all the values of each feature to normalize them.
    def normalize_features_values(self, x_set):
        i = 0
        for x in x_set:
            j = 0
            for feature in x:
                x_set[i][j] = self.min_max_normalization(feature_value=feature, feature_index=j)
                j += 1
            i += 1

    # Reading the wines values from the input files.
    def read_training_set(self):
        train_x, train_y, test_x = [], [], []
        train_x_file_name, train_y_file_name, test_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
        x_file, y_file = open(train_x_file_name, "r"), open(train_y_file_name, "r")
        test_file = open(test_file_name, "r")
        for line in x_file:
            array = self.create_vector(line)
            train_x.append(array)
        for line in test_file:
            array = self.create_vector(line)
            test_x.append(array)
        for line in y_file:
            train_y.append(float(line.strip('\n')))
        return np.array(train_x).astype(np.float), np.array(train_y).astype(np.int), np.array(test_x).astype(np.float)

    # Splitting the wines to training set (classification unknown) and test examples (classification known).
    def split_to_examples_and_training_sets(self, examples_num):
        num_rows, num_cols = self.train_x.shape
        idx = np.random.randint(num_rows - 1, size=examples_num)
        self.data_set = self.train_x[idx, :]
        self.examples_labels = self.train_y[idx]
        self.training_set = np.delete(self.train_x, idx, axis=0)


class KNN:
    def __init__(self, k):
        self.k = k
        self.training_values = TrainingValues("zscore")
        self.x_y_dic = {}

    # Finding the k nearest neighbours of a wine (the test examples that are
    # most similar to a wine from the training set)
    def find_k_nearest_neighbours(self, x):
        d_example_dic = {}
        d_priority_queue = []
        max_d_in_queue = float('inf')
        example_index = 0
        for data in self.training_values.data_set:
            d = distance.euclidean(x, data)
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

    # Finding the label that most of the watch neighbours has.
    def label_x(self, x, nearest_neighbours):
        labels_count_dic = {}
        for neighbour_index in nearest_neighbours.values():
            label = self.training_values.examples_labels[neighbour_index]
            if label in labels_count_dic:
                labels_count_dic[label] += 1
            else:
                labels_count_dic[label] = 1
        max_label = max(labels_count_dic, key=lambda k: labels_count_dic[k])
        return max_label

    def train(self):
        labeled_test = []
        # Running the algorithm on the train examples
        for x in self.training_values.training_set:
            nearest_neighbours = self.find_k_nearest_neighbours(x=x)
            self.x_y_dic[x.tobytes()] = self.label_x(x=x, nearest_neighbours=nearest_neighbours)
        # Running the algorithm on the test
        for x in self.training_values.test_x:
            nearest_neighbours = self.find_k_nearest_neighbours(x=x)
            labeled_test.append(self.label_x(x=x, nearest_neighbours=nearest_neighbours))
        return labeled_test


# The Perceptron algorithm for machine learning.
class Perceptron:

    def __init__(self, learning_rate, epochs):
        self.training_values = TrainingValues("minmax")
        self.rate = learning_rate
        self.epochs = epochs
        self.weights = self.init_weights([0, 1, 2])

    # init weights vectors to 0 (as the number of the labels).
    def init_weights(self, labels_num):
        weights = []
        for _ in labels_num:
            weights.append(np.zeros(self.training_values.features_num))
        return np.array(weights)

    # Label all the test x according to the weight vectors the algorithm learned.
    def label_test(self):
        labeled_test = []
        for x in self.training_values.test_x:
            labeled_test.append(np.argmax(np.dot(self.weights, x)))
        return labeled_test

    # Perceptron training, finding the best weights vectors according to the train x and y values.
    def train(self):
        for _ in range(self.epochs):
            shuffle_indexes = np.random.permutation(len(self.training_values.train_y))
            train_x = self.training_values.train_x[shuffle_indexes]
            train_y = self.training_values.train_y[shuffle_indexes]
            for x, label in zip(train_x, train_y):
                y_hat = np.argmax(np.dot(self.weights, x))
                if label != y_hat:
                    self.weights[label] = self.weights[label] + self.rate * x
                    self.weights[y_hat] = self.weights[y_hat] - self.rate * x
        return self.label_test()


# The PassiveAggressive algorithm for machine learning.
class PassiveAggressive:
    def __init__(self, learning_rate, epochs):
        self.training_values = TrainingValues("minmax")
        self.rate = learning_rate
        self.epochs = epochs
        self.weights = self.init_weights([0, 1, 2])

    # init weights vectors to 0 (as the number of the labels).
    def init_weights(self, labels_num):
        weights = []
        for _ in labels_num:
            weights.append(np.zeros(self.training_values.features_num))
        return np.array(weights)

    # Calculating the tau
    def cal_tau(self, y, y_hat, x):
        loss = max(0, 1 - np.dot(self.weights[y], x) + np.dot(self.weights[y_hat], x))
        return loss / (2 * np.linalg.norm(x) ** 2)

    # Label all the test x according to the weight vectors the algorithm learned.
    def label_test(self):
        labeled_test = []
        for x in self.training_values.test_x:
            labeled_test.append(np.argmax(np.dot(self.weights, x)))
        return labeled_test

    # PA training, finding the best weights vectors according to the train x and y values and tau.
    def train(self):
        for _ in range(self.epochs):
            shuffle_indexes = np.random.permutation(len(self.training_values.train_y))
            train_x = self.training_values.train_x[shuffle_indexes]
            train_y = self.training_values.train_y[shuffle_indexes]
            for x, label in zip(train_x, train_y):
                y_hat = np.argmax(np.dot(self.weights, x))
                self.cal_tau(label, y_hat, x)
                if label != y_hat:
                    self.weights[label] = self.weights[label] + self.rate * x
                    self.weights[y_hat] = self.weights[y_hat] - self.rate * x
        return self.label_test()


def print_output(size):
    for i in range(size):
        print(f"knn: {knn_test_labels[i]}, perceptron: {perceptron_test_labels[i]}, pa: {pa_test_labels[i]}")


# The main function
if __name__ == "__main__":
    knn = KNN(k=7)
    knn_test_labels = knn.train()
    print(knn_test_labels)
    perceptron = Perceptron(learning_rate=0.33, epochs=20)
    perceptron_test_labels = perceptron.train()
    print(perceptron_test_labels)
    pa = PassiveAggressive(learning_rate=0.33, epochs=17)
    pa_test_labels = pa.train()
    print(pa_test_labels)
    print_output(len(pa_test_labels))
