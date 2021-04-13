import pandas as pd
import numpy as np

df = pd.read_csv('b_depressed.csv')
df = df.drop('Survey_id', axis='columns')
df = df.drop('Ville_id', axis='columns')


# print(df.values[0])
# print(df.columns)


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth


class Question:
    """
    Represents a question asked at each node of a tree.
    :param column:  number of column which stores the value
    :param value:   value at that given column
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    """
    Responds with a boolean value to determine if the supplied row's value 
    at the Question's column is greater or equal to the Question's value.
    """

    def match(self, row):
        val = row[self.column]
        return val >= self.value

    def __repr__(self):
        return "Is %s %s %s?" % (
            df.columns[self.column], ">=", str(self.value))


# Question:
# quest = Question(0, 3)
# print(quest)
# print(quest.match(df.values[0]))


def partition(rows, question):
    """
    Partitions the dataset in accordance with the Question being asked. Outputs two lists:
    one with rows that meet the condition asked in the question and the other that holds
    the rows that do not meet that condition.
    :param rows : Dataframe values as list.
    :param question : Question object with a given condition.
    """
    true_rows = []
    false_rows = []

    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


#
# true_ones, false_ones = partition(df.values, Question(0,1))


def get_gini(rows):
    """
    Calculates a gini index to determine impurity of given rows of data.
    :param rows : rows of data as 2+ dimensional python list
    :return : gini index float
    """
    count_zero = 0  # number of rows labelled healthy
    count_one = 0  # number of rows labelled depressed
    for row in rows:
        if row[len(row) - 1] == 0:
            count_zero = count_zero + 1
        else:
            count_one = count_one + 1
    return 1 - (count_zero / float(len(rows))) ** 2 - (count_one / float(len(rows))) ** 2


# impurity of the whole dataset
print(get_gini(df.values))
# data sample with 0 impurity
print(get_gini([[0, 0, 0], [0, 0, 0]]))


def get_info_gain(true_rows, false_rows, current_impurity):
    """
    Finds information gain for a partition produced by a question.
    :return: information gain (weighted average
    """
    avg_impurity = len(true_rows) * get_gini(true_rows) + len(false_rows) * get_gini(false_rows)
    return current_impurity - avg_impurity


def get_best_split(rows):
    """
    Finds best question to ask.
    :return: a Question object that produces maximum information gain and the value of that gain.
    """
    best_gain = 0
    best_question = None
    current_impurity = get_gini(rows)
    n_features = len(rows[0])

    for col in range(n_features):
        for row in rows:
            question = Question(col, row[col])
            true_rows, false_rows = partition(rows, question)
            question_gain = get_info_gain(true_rows, false_rows, current_impurity)
            if question_gain >= best_gain:
                best_gain = question_gain
                best_question = question_gain
    return best_gain, best_question
