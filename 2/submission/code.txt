#!/usr/bin/env python
# coding: utf-8

# Datasets Description
# 
# The UCI Machine Learning Repository is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.
# 
# Datasets are available on http://archive.ics.uci.edu/ml/datasets.html For this homework assignment, you need to download the datasets “glass” and “Tic-Tac-Toe Endgame” from the above link. The “glass” dataset is categorical and the “Tic-Tac-Toe” dataset is continuous.

# # **Question 1**
# 
# Design a C4.5 decision tree classifier to classify each dataset mentioned above. Report the accuracy based on the 10-times-10-fold cross validation approach (20% of training set as the validation set for every experiment). Report the mean accuracy and the variance of the accuracy for each experiment.

# In[1]:


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import random
from pprint import pprint
import matplotlib.pyplot as plt


# In[2]:


def if_node_is_label(data):
    if len(np.unique(data[:, -1])) == 1:
        return True
    return False


# In[3]:


def get_label(data):
    my_classes, class_count = np.unique( data[:, -1], return_counts=True)
    return my_classes[class_count.argmax()]


# In[4]:


def get_all_splits(data):
    all_splits = {}
    _, cols = data.shape
    for i in range(cols - 1): 
        all_splits[i] = np.unique(data[:, i])
    return all_splits


# In[5]:


def break_data(data, column, value, feature_types):
    
    split_column = data[:, column]
    feature_type = feature_types[column]
    if feature_type == "continuous":
        down = data[split_column <= value]
        up = data[split_column >  value]

    else:
        down = data[split_column_values == value]
        up = data[split_column_values != value]
    
    return down, up


# In[6]:


def calig(data):
    _, count = np.unique(data[:, -1], return_counts=True)
    p = count / count.sum()
    return sum(p * -np.log2(p))


# In[7]:


def calig_all(down, up):
    n = len(down) + len(up)
    p_down = len(down) / n
    p_up = len(up) / n
    overall_entropy =  (p_down * calig(down) 
                      + p_up * calig(up))
    return overall_entropy


# In[8]:


def get_best_split(data, all_splits, feature_types):
    max_entropy = 10000000
    for i in all_splits:
        for v in all_splits[i]:
            down, up = break_data(data, i, v, feature_types)
            current_max_entropy = calig_all(down, up)

            if current_max_entropy <= max_entropy:
                max_entropy = current_max_entropy
                best_col = i
                best_val = v
    
    return best_col, best_val


# In[9]:


def get_types_of_features(df):
    types = []
    treshold = 15
    for f in df.columns:
        if f != "label":
            vals = df[f].unique()
            example = vals[0]
            if (isinstance(example, str)) or (len(vals) <= treshold):
                types.append("symbol")
            else:
                types.append("continuous")
    
    return types


# In[10]:


class Node: 
    def __init__(self, data): 
        self.data = data 
        self.left = None
        self.right = None
        
class DecisionTree:
    
    def __init__(self): 
        self.root_main = None
        self.rules = []
        self.feature_types = None
        self.columns = None
    
    def build_tree(self, df, root=None, pointer="S", counter=0):


        if counter == 0:
            self.feature_types = get_types_of_features(df)
            data = df.values
            self.columns = df.columns
        else:
            data = df           

        if (if_node_is_label(data)):
            classification = get_label(data)
            node = Node(classification)
            if pointer is "L":
                root.left=node
                root=node
            if pointer is "R":
                root.right=node
                root=node
            return 
        # recursive part
        else:    
            counter += 1

            # computing C4.5 functions 
            all_splits = get_all_splits(data)
            my_split_column, my_split_value = get_best_split(data, all_splits, self.feature_types)
            down, up = break_data(data, my_split_column, my_split_value, self.feature_types)

            # check for empty data
            if len(down) == 0 or len(up) == 0:
                classification = get_label(data)
                node = Node(classification)
                if pointer is "L":
                    root.left=node
                    root=node
                if pointer is "R":
                    root.right=node
                    root=node
                return

            # determine question
            feature_name = self.columns[my_split_column]
            type_of_feature = self.feature_types[my_split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, my_split_value)
            else:
                question = "{} == {}".format(feature_name, my_split_value)

            if pointer is "S":
                root=Node(question)
                self.root_main = root
            else:
                node = Node(question)
                if pointer is "L":
                    root.left=node   
                    root=node

                if pointer is "R":
                    root.right=node
                    root=node

            self.build_tree(down, root, "L", counter)
            self.build_tree(up, root, "R", counter)

            return self.root_main
         
    def convert_tree_to_rules(self, path=[], pathLen=0, status=""):
        root = self.root_main
        self.get_paths(root,path,pathLen,status)
        ar = np.array(self.rules)
        for i in range(len(self.rules)):
            for j in range(len(self.rules[i])):
                if "R" in self.rules[i][j]:
                    feature_index, comparison_operator, value = self.rules[i][j-1].split(" ")
                    if self.feature_types[int(feature_index)] == "continuous":
                        self.rules[i][j-1] = "{} > {}".format(feature_index, value) 
                    else:
                        self.rules[i][j-1] = "{} != {}".format(feature_index, value) 
                self.rules[i][j]=self.rules[i][j].replace('R', '')
                self.rules[i][j]=self.rules[i][j].replace('L', '')
        return self.rules
        
    def save_paths(self, ints, len): 
        array = []
        for i in ints[0 : len]:
            array.append(str(i))
        self.rules.append(array)
    
    def get_paths(self, my_node, tree_path, p_len, status):

        if my_node is None: 
            return

        if(len(tree_path) > p_len):  
            tree_path[p_len] = status+str(my_node.data) 
        else: 
            tree_path.append(status+str(my_node.data))

        p_len = p_len + 1

        if my_node.left is None and my_node.right is None: 
            self.save_paths(tree_path, p_len) 
        else: 
            self.get_paths(my_node.left, tree_path, p_len, "L") 
            self.get_paths(my_node.right, tree_path, p_len, "R") 

    def get_class_of_sample_from_rule(self, rule, sample):
        sample_class = None
        for i in range(len(rule)):
            if i < len(rule) - 1:
                feature_index, comparison_operator, value = rule[i].split(" ")
                if self.feature_types[int(feature_index)] == "continuous":
                    if comparison_operator == "<=":
                        if not float(sample[int(feature_index)]) <= float(value):
                            break
                    elif comparison_operator == ">":
                        if not float(sample[int(feature_index)]) > float(value):
                            break
                else:
                    if comparison_operator == "==":
                        if not sample[int(feature_index)] == value:
                            break
                    elif comparison_operator == "!=":
                        if not sample[int(feature_index)] is not value:
                            break
            else:
                sample_class = rule[len(rule)-1]         
        return sample_class
     
    def prune_rule(self, rule, validation_data):
        
        base_rule = rule
        base_accuracy = 0.0
        new_rule = rule
        new_accuracy = 0.0
        old_accuracy = 0.0
        
        accuracy = 0
        total_for_accuracy = 0
        for i in range(len(validation_data)):
            predict_class = self.get_class_of_sample_from_rule(rule, validation_data[i])
            if predict_class is not None:
                total_for_accuracy+=1
                if predict_class == str(validation_data[i][9]):
                    accuracy +=1
                
        if total_for_accuracy is not 0: 
            base_accuracy = accuracy/total_for_accuracy
            old_accuracy = base_accuracy
            new_accuracy = base_accuracy
        
        flag=1
        while flag==1:
            flag=0 
            for i in range(len(rule)-1):
                my_rule = rule.copy()
                my_rule.pop(i)          
                accuracy = 0
                total_for_accuracy = 0
                for i in range(len(validation_data)):
                    predict_class = self.get_class_of_sample_from_rule(my_rule, validation_data[i])
                    if predict_class is not None:
                        total_for_accuracy+=1
                        if predict_class == str(validation_data[i][9]):
                            accuracy +=1
                
                tmp_rule = my_rule.copy()
                if total_for_accuracy is not 0: 
                    tmp_accuracy = accuracy/total_for_accuracy
                    if tmp_accuracy > old_accuracy:
                        old_accuracy = tmp_accuracy
                        new_accuracy = tmp_accuracy
                        new_rule = tmp_rule
                        flag=1   
        return base_rule, base_accuracy, new_rule, new_accuracy
      
    def prune_tree(self, validation_data):
        
        self.convert_tree_to_rules()
        validation_data=validation_data.values
        new_rules = []
    
        for i, rule in enumerate(self.rules):
            rule_label = rule[len(rule)-1]    
            old_rule, old_accurcay, new_rule, new_accuracy = self.prune_rule(rule, validation_data)
            new_rule = [new_accuracy] + new_rule
            new_rules.append(new_rule)
        
        self.rules = new_rules
        self.rules.sort(key=lambda x: x[0])
        self.rules = self.rules[::-1]
    
    def get_rules(self): 
        self.rules = [val[1:] for val in self.rules]
        return self.rules
    
    def predict_class(self, sample):
        sample_class = None
        for rule in self.rules:
            sample_class = self.get_class_of_sample_from_rule(rule,sample)
            if sample_class is not None:
                break
        return sample_class


# In[11]:


def train_test_k_fold_split(df, fold):
    indices = df.index.tolist()
    low = int((fold/10)*len(indices))
    high = int(((fold+1)/10)*len(indices))
    test_indices=indices[low:high]
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


# In[12]:


def split_training_and_validation(df, division):
    indices = df.index.tolist()
    validation_indices=indices[int((1-division)*len(indices)):]
    validation_df = df.loc[validation_indices]
    train_df = df.drop(validation_indices)
    return train_df, validation_df


# ## Glass Dataset

# In[13]:


from sklearn.utils import shuffle
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
df=df.drop(df.columns[0], axis=1)
df.columns = [i for i in range(len(df.columns))]

my_accuracies_main = []
for _ in range(10):
    my_accuracies = []
    df=shuffle(df)
    for i in range(10):
        decision_tree = DecisionTree()
        train_df, test_df = train_test_k_fold_split(df, i)
        train_data, validation_data=split_training_and_validation(train_df, 0.2)
        decision_tree.build_tree(train_data)
        decision_tree.prune_tree(validation_data)
        decision_tree.get_rules()

        correct = 0
        for j in range(len(test_df)):
            example = test_df.iloc[j]
            if decision_tree.predict_class(example) is not None and float(decision_tree.predict_class(example))==example[9]: correct+=1
        my_accuracies.append(correct/len(test_df)) 
    my_accuracies_main.append(my_accuracies)


# In[14]:


glass_mean_accuracy = np.mean(my_accuracies_main)
glass_var = np.var(my_accuracies_main)
print("Glass Accuracy Mean :",glass_mean_accuracy)
print("Glass Accuracy Variance :",glass_var)


# In[15]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data')
df.columns = [i for i in range(len(df.columns))]

my_accuracies_main = []
for _ in range(10):
    df=shuffle(df)
    my_accuracies = []
    for i in range(10):
        decision_tree = DecisionTree()
        train_df, test_df = train_test_k_fold_split(df, i)
        train_data, validation_data=split_training_and_validation(train_df, 0.2)
        decision_tree.build_tree(train_data)
        decision_tree.prune_tree(validation_data)
        decision_tree.get_rules()

        correct = 0
        for j in range(len(test_df)):
            example = test_df.iloc[j]
            if decision_tree.predict_class(example) is not None and decision_tree.predict_class(example)==example[9]: correct+=1
        my_accuracies.append(correct/len(test_df))
    my_accuracies_main.append(my_accuracies)


# In[16]:


ttt_mean_accuracy = np.mean(my_accuracies_main)
ttt_var = np.var(my_accuracies_main)
print("Tic-Tac-Toe Accuracy Mean :",ttt_mean_accuracy)
print("Tic-Tac-Toe Accuracy Variance :",ttt_var)


# # **Question 2**
# 
# There are two possible sources for class label noise:
# 
# a) Contradictory examples. The same sample appears more than once and is labeled with a different classification.
# 
# b) Misclassified examples. A sample is labeled with the wrong class. This type of error is common in situations where different classes of data have similar symptoms.
# 
# To evaluate the impact of class label noise, you should execute your experiments on both datasets, while various levels of noise are added. Then utilize the designed C4.5 learning algorithm from Question 1 to learn from the noisy datasets and evaluate the impact of class label noise (both Contradictory examples & Misclassified examples).
# 
# ● Note: when creating the noisy datasets, select L% of training data randomly and change them. (Try 10-times-10-fold cross validation to calculate the accuracy/error for each experiment.)

# In[17]:


def generate_misclassified_noise(data, val):    
    label_column = data.iloc[:,-1]
    unique_classes = np.unique(label_column)
    for _ in range(int(val*len(data))):
        random_index_for_data = np.random.randint(low=0, high=len(data))
        random_index_for_unique_labels = np.random.randint(low=0, high=len(unique_classes))
        data.iloc[random_index_for_data,-1]=unique_classes[random_index_for_unique_labels]
    return data


# In[62]:


def generate_contradictory_noise(data, val, unique_classes):    
    d=data
    for _ in range(int(val*len(data))):
        random_index_for_data = np.random.randint(low=0, high=len(data))
        new_row=data.iloc[random_index_for_data]
        tmp = np.delete(unique_classes,np.where(unique_classes==new_row[9]))
        random_index_for_unique_labels = np.random.randint(low=0, high=len(tmp))
        new_row[9]=tmp[random_index_for_unique_labels]
        d=d.append(new_row)
    return d


# a) Plot one figure for each dataset that shows the noise free classification accuracy along with the classification accuracy for the following noise levels: 5%, 10%, and 15%. Plot the two types of noise on one figure.

# ## Glass Dataset

# In[66]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
df=df.drop(df.columns[0], axis=1)
df.columns = [i for i in range(len(df.columns))]

accuracies_under_misclassified_noise = []
noises = [0.05,0.10,0.15]
for noise in noises:
    my_accuracies_main = []
    for _ in range(10):
        df=shuffle(df)
        my_accuracies = []
        for i in range(10):

            decision_tree = DecisionTree()
            train_df, test_df = train_test_k_fold_split(df, i)
            train_df = generate_misclassified_noise(train_df,noise)
            train_data, validation_data=split_training_and_validation(train_df, 0.2)
            decision_tree.build_tree(train_data)
            decision_tree.prune_tree(validation_data)
            decision_tree.get_rules()

            correct = 0
            for j in range(len(test_df)):
                example = test_df.iloc[j]
                if decision_tree.predict_class(example) is not None and float(decision_tree.predict_class(example))==example[9]: correct+=1
            my_accuracies.append(correct/len(test_df))
        my_accuracies_main.append(my_accuracies)
    accuracies_under_misclassified_noise.append([np.mean(my_accuracies_main),np.var(my_accuracies_main)])


# In[67]:


print("\nFor Glass Dataset, Misclassified Noise...\n")
print("For No Noise 0% : Mean is",glass_mean_accuracy," and Variance is", glass_var)
print("For No Noise 5% : Mean is",accuracies_under_misclassified_noise[0][0]," and Variance is", accuracies_under_misclassified_noise[1][1])
print("For No Noise 10% : Mean is",accuracies_under_misclassified_noise[1][0]," and Variance is", accuracies_under_misclassified_noise[1][1])
print("For No Noise 15% : Mean is",accuracies_under_misclassified_noise[2][0]," and Variance is", accuracies_under_misclassified_noise[2][1])
y_misclassified = [glass_mean_accuracy,accuracies_under_misclassified_noise[0][0]
     ,accuracies_under_misclassified_noise[1][0],accuracies_under_misclassified_noise[2][0]]


# In[69]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
df=df.drop(df.columns[0], axis=1)
df.columns = [i for i in range(len(df.columns))]

accuracies_under_contrary_noise = []
noises = [0.05,0.10,0.15]
for noise in noises:
    my_accuracies_main = []
    for _ in range(10):
        df=shuffle(df)
        my_accuracies = []
        for i in range(10):

            decision_tree = DecisionTree()
            train_df, test_df = train_test_k_fold_split(df, i)
            train_df = generate_contradictory_noise(train_df,noise, np.unique(df.iloc[:,9]))
            train_data, validation_data=split_training_and_validation(train_df, 0.2)
            decision_tree.build_tree(train_data)
            decision_tree.prune_tree(validation_data)
            decision_tree.get_rules()

            correct = 0
            for j in range(len(test_df)):
                example = test_df.iloc[j]
                if decision_tree.predict_class(example) is not None and float(decision_tree.predict_class(example))==example[9]: correct+=1
            my_accuracies.append(correct/len(test_df))
        my_accuracies_main.append(my_accuracies)
    accuracies_under_contrary_noise.append([np.mean(my_accuracies_main),np.var(my_accuracies_main)])


# In[70]:


print("\nFor Glass Dataset, Contradictoy Noise...\n")
print("For No Noise 0% : Mean is",glass_mean_accuracy," and Variance is", glass_var)
print("For No Noise 5% : Mean is",accuracies_under_contrary_noise[0][0]," and Variance is", accuracies_under_contrary_noise[1][1])
print("For No Noise 10% : Mean is",accuracies_under_contrary_noise[1][0]," and Variance is", accuracies_under_contrary_noise[1][1])
print("For No Noise 15% : Mean is",accuracies_under_contrary_noise[2][0]," and Variance is", accuracies_under_contrary_noise[2][1])
y_contradictory = [glass_mean_accuracy,accuracies_under_contrary_noise[0][0]
     ,accuracies_under_contrary_noise[1][0],accuracies_under_contrary_noise[2][0]]


# In[71]:


plt.plot(["No Noise","5%","10%","15%"], y_misclassified, linestyle="--",label="Misclassified Noise")
plt.plot(["No Noise","5%","10%","15%"], y_contradictory, linestyle="--",label="Contradictory Noise")
plt.title("Percentage of Misclassified Noise vs Accuracy")
plt.xlabel("Percentage of Noise",)
plt.ylabel("Accuracy")
plt.legend()
plt.show();


# ## Tic-Tac-Toe Glass Dataset

# In[25]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data')
df.columns = [i for i in range(len(df.columns))]

accuracies_under_misclassified_noise = []
noises = [0.05,0.10,0.15]
for noise in noises:
    my_accuracies_main = []
    for _ in range(10):
        df=shuffle(df)
        my_accuracies = []
        for i in range(10):
            decision_tree = DecisionTree()
            train_df, test_df = train_test_k_fold_split(df, i)
            train_df = generate_misclassified_noise(train_df,noise)
            train_data, validation_data=split_training_and_validation(train_df, 0.2)
            decision_tree.build_tree(train_data)
            decision_tree.prune_tree(validation_data)
            decision_tree.get_rules()

            correct = 0
            for j in range(len(test_df)):
                example = test_df.iloc[j]
                if decision_tree.predict_class(example) is not None and decision_tree.predict_class(example)==example[9]: correct+=1
            my_accuracies.append(correct/len(test_df))
        my_accuracies_main.append(my_accuracies)
    accuracies_under_misclassified_noise.append([np.mean(my_accuracies_main),np.var(my_accuracies_main)])


# In[31]:


print("\nFor Tic-Tac-Toe Dataset, Misclassified Noise...\n")
print("For No Noise 0% : Mean is",ttt_mean_accuracy," and Variance is",ttt_var)

print(accuracies_under_misclassified_noise)
print("For No Noise 5% : Mean is",accuracies_under_misclassified_noise[0][0]," and Variance is", accuracies_under_misclassified_noise[1][1])
print("For No Noise 10% : Mean is",accuracies_under_misclassified_noise[1][0]," and Variance is", accuracies_under_misclassified_noise[1][1])
print("For No Noise 15% : Mean is",accuracies_under_misclassified_noise[2][0]," and Variance is", accuracies_under_misclassified_noise[2][1])
y_misclassified = [ttt_mean_accuracy,accuracies_under_misclassified_noise[0][0]
     ,accuracies_under_misclassified_noise[1][0],accuracies_under_misclassified_noise[2][0]]


# In[63]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data')
df.columns = [i for i in range(len(df.columns))]

accuracies_under_contrary_noise = []
noises = [0.05,0.10,0.15]
for noise in noises:
    my_accuracies_main = []
    for _ in range(10):
        df=shuffle(df)
        my_accuracies = []
        for i in range(10):
            decision_tree = DecisionTree()
            train_df, test_df = train_test_k_fold_split(df, i)
            train_df = generate_contradictory_noise(train_df, noise, np.unique(df.iloc[:,9]))
            train_data, validation_data=split_training_and_validation(train_df, 0.2)
            decision_tree.build_tree(train_data)
            decision_tree.prune_tree(validation_data)
            decision_tree.get_rules()

            correct = 0
            for j in range(len(test_df)):
                example = test_df.iloc[j]
                if decision_tree.predict_class(example) is not None and decision_tree.predict_class(example)==example[9]: correct+=1
            my_accuracies.append(correct/len(test_df))
        my_accuracies_main.append(my_accuracies)
    accuracies_under_contrary_noise.append([np.mean(my_accuracies_main),np.var(my_accuracies_main)])


# In[64]:


print("\nFor Tic-Tac-Toe Dataset, Contradictoy Noise...\n")
print("For No Noise 0% : Mean is",ttt_mean_accuracy," and Variance is", ttt_var)
print("For No Noise 5% : Mean is",accuracies_under_contrary_noise[0][0]," and Variance is", accuracies_under_contrary_noise[1][1])
print("For No Noise 10% : Mean is",accuracies_under_contrary_noise[1][0]," and Variance is", accuracies_under_contrary_noise[1][1])
print("For No Noise 15% : Mean is",accuracies_under_contrary_noise[2][0]," and Variance is", accuracies_under_contrary_noise[2][1])
y_contradictory = [ttt_mean_accuracy,accuracies_under_contrary_noise[0][0]
     ,accuracies_under_contrary_noise[1][0],accuracies_under_contrary_noise[2][0]]


# In[65]:


plt.plot(["No Noise","5%","10%","15%"], y_misclassified, linestyle="--",label="Misclassified Noise")
plt.plot(["No Noise","5%","10%","15%"], y_contradictory, linestyle="--",label="Contradictory Noise")
plt.title("Percentage of Misclassified Noise vs Accuracy")
plt.xlabel("Percentage of Noise",)
plt.ylabel("Accuracy")
plt.legend()
plt.show();


# b) How do you explain the effect of noise on the C4.5 method?

# Answer in the report.





# ## Question 3
# 
# Design a feature selection algorithm to find the best features for classifying the Mnist dataset. Implement a bidirectional search algorithm using the provided objective function as the measure for your search algorithm.
# 
# Use the first 10000 samples of training set in the Mnist dataset for feature selection and training set for kNN approach. Use Euclidean distance to calculate Inter-class.
# 
# The objective function should be based on this equestion:
# 
# ### J = Inter Class distance

# In[242]:


import gzip
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

image_size = 28
training_samples = 60000

# Importing Train Data
f_train = gzip.open('train-images-idx3-ubyte.gz','r')
f_train.read(16)
buf = f_train.read(image_size * image_size * training_samples)
train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_data = train_data.reshape(training_samples, image_size* image_size)


# Importing Train Labels
f_train_label = gzip.open('train-labels-idx1-ubyte.gz','r')
f_train_label.read(8)
buf = f_train_label.read(training_samples)
train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)


# Importing Test Data
testing_images = 10000
f_test = gzip.open('t10k-images-idx3-ubyte.gz','r')
f_test.read(16)
buf = f_test.read(image_size * image_size * testing_images)
test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(testing_images, image_size * image_size)

# Importing Test Labels
f_test_label = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f_test_label.read(8)
buf = f_test_label.read(testing_images)
test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)


train_data = train_data[0:10000]
train_labels = train_labels[0:10000]


# In[228]:


from sklearn.neighbors import KNeighborsClassifier
def apply_knn(train_data, train_labels, k):
    
    train_d =train_data[:8000]
    train_l =train_labels[:8000]
    
    test_d = train_data[8000:]
    test_l =train_labels[8000:]
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_d,train_l)
    
    correct = 0
    for i in range(len(test_d)):
        if model.predict([test_d[i]]) == test_l[i]: correct+=1
    
    return correct/len(test_d)


# a) Select the set of {10, 50, 150, 392} features based on the implemented feature selection approach and report the accuracy on the test set of MNIST based on kNN with k = 3. Note: you can take advantage of data structure tricks to speed up the efficiency of kNN algorithm.

# In[285]:


dimensions = [10,50,150,392]
my_train_data = train_data
new_train_data = None
deleted_j_max_indices = []
deleted_j_min_indices = []
for _ in range(392):
    
    # SFS
    main_mean = np.mean(my_train_data, axis=0)
    mean_i = []
    for i in np.unique(train_labels):
        indices_of_i = np.where(train_labels == i)[0]
        c_i = np.take(my_train_data, indices_of_i, axis=0)
        mean_i.append(np.mean(c_i, axis=0))    

    J_max = 0
    j_max_index = 0
    for i in range(784):
        if i not in deleted_j_max_indices and i not in deleted_j_min_indices:
            distance = 0
            for mean in mean_i:
                distance += euclidean(main_mean[i],mean[i])
            if distance >= J_max:
                J_max = distance
                j_max_index = i
   
    deleted_j_max_indices.append(j_max_index)
    
    if new_train_data is None:
        new_train_data=my_train_data[:,j_max_index]
    else:  
        new_train_data = np.c_[new_train_data, my_train_data[:,j_max_index]]

    
    # SBS
    main_mean = np.mean(my_train_data, axis=0)
    mean_i = []
    for i in np.unique(train_labels):
        indices_of_i = np.where(train_labels == i)[0]
        c_i = np.take(my_train_data, indices_of_i, axis=0)
        mean_i.append(np.mean(c_i, axis=0))  
        
    J_min = J_max
    j_min_index = 0
    for i in range(784):
        if i not in deleted_j_max_indices and i not in deleted_j_min_indices:
            distance = 0
            for mean in mean_i:
                distance += euclidean(main_mean[i],mean[i])
            if distance <= J_min:
                J_min = distance
                j_min_index = i
    
    deleted_j_min_indices.append(j_min_index)

print(new_train_data.shape)


# In[328]:


accuracies = []
for d in dimensions:
    print("For d =",d, "Accuracy is",apply_knn(new_train_data[:,:d],train_labels, 3))


# b) Visualize the selected features for each set in {10, 50, 150, 392} by a zero 2-D plane where the selected features are pixels set to a value of 1. Compare the 4 different planes.

# In[292]:


for d in dimensions:
    data = np.zeros(784)
    for val in deleted_j_max_indices[:d]:
        data[val] = 1.0
    data = data.reshape((28, 28))
    plt.imshow(data);
    plt.title("First best %i features"%d)
    plt.show()


# c) Apply LDA on the dataset and report the accuracy based on kNN with k = 3. Compare the achieved accuracy by the reported accuracies in part (a). Note: you need to implement LDA method by yourself.

# In[323]:


my_train_data = train_data
n_comp=9

overall_mean = np.mean(my_train_data, axis=0)

S_W=np.zeros((784,784))
for i in np.unique(train_labels):
    S=np.zeros((784,784))
    indices_of_i = np.where(train_labels == i)[0]
    c_i = np.take(my_train_data, indices_of_i, axis=0)
    mean = np.mean(c_i, axis=0)
    mean= mean.reshape(784,1)
    for sample in c_i:
        sample= sample.reshape(784,1)
        S=S+np.dot(sample-mean,(sample-mean).T)
    S_W += S      
    
S_B=np.zeros((784,784))
for i in np.unique(train_labels):
    indices_of_i = np.where(train_labels == i)[0]
    c_i = np.take(my_train_data, indices_of_i, axis=0)
    mean = np.mean(c_i, axis=0)
    mean= mean.reshape(784,1)
    overall_mean= overall_mean.reshape(784,1)
    S_B = S_B + len(c_i)*(mean-overall_mean).dot((mean-overall_mean).T)
    
eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_b))

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W = np.hstack([eig_pairs[i][1].reshape(784, 1) for i in range(0, n_comp)])
new_data= my_train_data.dot(W)
print("Accuracy :",apply_knn(new_data.real, train_labels, 3))






