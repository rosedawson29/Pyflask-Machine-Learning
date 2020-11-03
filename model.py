import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# dataset = pd.read_csv('sales.csv')

# dataset['rate'].fillna(0, inplace=True)

# dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# regressor.fit(X, y)

# pickle.dump(regressor, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[4, 300, 500]]))


# Phần bản thân

import random

from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions



df = pd.read_csv("/Users/rose/Workspace/Random Forest_1/datasets_216167_477177_heart.csv")

df=df.rename(columns = {'target':'label'})

random.seed(0)

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices,k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

train_df, test_df = train_test_split(df, test_size=0.2)

# Create Bootraping
def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped
# Create Forest
def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    return forest
# Make Prediction
def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions

# # Check the result
# forest = random_forest_algorithm(train_df, n_trees=100, n_bootstrap=300, n_features=2, dt_max_depth=4)
# predictions = random_forest_predictions(test_df, forest)

# predictions_correct = predictions == test_df.label
# predictions_correct.mean()

forest = random_forest_algorithm(train_df, n_trees=100, n_bootstrap=300, n_features=2, dt_max_depth=4)
pickle.dump(forest, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[4, 300, 500]]))
