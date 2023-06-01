from itertools import chain, combinations, filterfalse
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori as apriori2
from mlxtend.frequent_patterns import association_rules as association
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from time import time


#######################################################################################################
#                                           Excercise 3                                               #
#######################################################################################################


##########################
####  Working Dataset ####
##########################

# read retail.dat to a list of sets
datContent = [i.strip().split() for i in open("retail.dat").readlines()]
datContent =  [[int(i) for i in datContent[j]] for j in range(len(datContent))]

# Convert to one hot encoded array
mlb = MultiLabelBinarizer()
one_hot_encoded = mlb.fit_transform(datContent)
one_hot_encoded = one_hot_encoded.astype(bool)
# Check for one-item baskets
basket_sizes = np.apply_along_axis(np.sum, axis=1, arr=one_hot_encoded)
# Get indices
indx = np.where(basket_sizes==1)[0]
# Remove one-item baskets
one_hot_encoded = np.delete(arr=one_hot_encoded, obj=indx, axis=0)
train, test = train_test_split(one_hot_encoded, test_size=0.1, random_state=23)

# Create test data
def create_test(test_data):
    test_output = {}
    counter = 0
    for i in range(len(test_data)):
        indx = np.random.choice(np.where(test_data[i])[0])
        test_data[i][indx] = 0
        test_output[counter]={indx}
        counter += 1
    return test_data, test_output

def create_complex_test(test_data):
    test_output = {}
    counter = 0
    for i in range(len(test_data)):
        items = np.where(test_data[i])[0]
        nr = np.random.choice(a=np.arange(1, len(items)))
        indx = np.random.choice(items, nr)
        test_data[i][indx] = 0
        test_output[counter]=set(indx)
        counter += 1
    return test_data, test_output


test, test_output = create_complex_test(test)

train_dataframe = pd.DataFrame(train)

def get_sets(data, unique_items):
    transactions = {}
    for i in range(len(data)):
        indx = np.where(data[i]==1)[0]
        transactions[i] = set(np.array(unique_items)[indx])
    return transactions

user_items = get_sets(test, np.arange(len(test[0])))

###################################
####  Reccomendation Functions ####
###################################

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def join_set(itemsets, k):
    return set(
        [itemset1.union(itemset2) for itemset1 in itemsets for itemset2 in itemsets if len(itemset1.union(itemset2)) == k]
    )

def itemsets_support(transactions, itemsets, min_support):
    support_count = {itemset: 0 for itemset in itemsets}
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                support_count[itemset] += 1
    n_transactions = len(transactions)
    return {itemset: support / n_transactions for itemset, support in support_count.items() if support/ n_transactions >= min_support}


def recommend_items(input_items, rules, top_n=5, metric="confidence"):
    recommendations = {}
    for antecedent, consequent, support, confidence, lift in rules:
        if antecedent.issubset(input_items) and not consequent.issubset(input_items):
            for item in consequent:
                if item not in input_items:
                    if item not in recommendations:
                        recommendations[item] = []
                    recommendations[item].append((confidence, support, lift))
    recommendations = {
        item: (sum(conf for conf, _, _ in item_rules) / len(item_rules), sum(sup for _, sup, _ in item_rules) / len(item_rules), sum(lift for _, _, lift in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    # Added part to evaluate based on different metrices
    if metric == "confidence":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][0], -x[1][1], -x[1][2]))
    elif metric == "support":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][1], -x[1][0], -x[1][2]))
    elif metric == "lift":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][2], -x[1][0], -x[1][1]))
    return [item for item, val in sorted_recommendations[:top_n]]

def evaluate_recommendations(user_items, test_data, rules, metric, top_n=5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for user, true_items in test_data.items():
        # Assuming user_items is a dictionary with user IDs as keys and their associated items as values
        input_items = user_items[user]
        # Get recommendations for the user
        recommended_items = set(recommend_items(input_items, rules, top_n=top_n, metric=metric))
        true_items = set(true_items)
        true_positives += len(recommended_items.intersection(true_items))
        false_positives += len(recommended_items - true_items)
        false_negatives += len(true_items - recommended_items)
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score



###################################################
#### Creating dataset without one-item baskets ####
###################################################

# read retail.dat to a list of sets
datContent = [i.strip().split() for i in open("retail.dat").readlines()]

# remove one-item baskets
indices = []
for i in range(len(datContent)):
    if len(datContent[i]) == 1:
        indices.append(i)

datContent_reduced = [datContent[i] for i in range(len(datContent)) if i not in indices]


# filename = 'reduced_retail.txt'

# with open(filename, 'w') as file:
#     for transaction in datContent_reduced:
#         line = ' '.join(transaction)
#         file.write(line + '\n')




################################################
#### Load apriori and run Association Rules ####
################################################


#### APRIORI RESULTS WITH MIN SUPPORT 85 TRANSACTIONS ####


# read apriori results
frq = [i.strip().split() for i in open("ndi 2/df/output1.txt").readlines()]

# transform output to df
frq_dict = {"support":[],
            "itemsets":[]}

for i in tqdm(range(len(frq))):
    itemsets_list = frq[i][0:len(frq[i])-2]
    frq_dict["itemsets"].append(frozenset([int(x) for x in itemsets_list]))
    supp_str = frq[i][len(frq[i])-2]
    supp_int = int(supp_str[1:len(supp_str)-1])
    frq_dict["support"].append(supp_int)

frq_df = pd.DataFrame.from_dict(frq_dict)
frq_df = frq_df.drop(index=[0])


# Run association
metric = "confidence"
threshold = 0.5

rules_new = association(frq_df, min_threshold = threshold)

# Change format of the associatotion rules
final_rules = []
for i in tqdm(range(len(rules_new))):
    antecedents, consequents, support, confidence, lift = rules_new.iloc[i]["antecedents"], rules_new.iloc[i]["consequents"], rules_new.iloc[i]["support"], rules_new.iloc[i]["confidence"], rules_new.iloc[i]["lift"]
    final_rules.append((antecedents, consequents, support, confidence, lift))


# Get evaluation
top_n = 5
precision, recall, f1_score = evaluate_recommendations(user_items, test_output, final_rules, metric, top_n=top_n)




#### MIN SUPPORT 125 TRANSACTIONS ####

# read apriori results
frq = [i.strip().split() for i in open("ndi 2/df/output2.txt").readlines()]

# transform output to df
frq_dict = {"support":[],
            "itemsets":[]}

for i in tqdm(range(len(frq))):
    itemsets_list = frq[i][0:len(frq[i])-2]
    frq_dict["itemsets"].append(frozenset([int(x) for x in itemsets_list]))
    supp_str = frq[i][len(frq[i])-2]
    supp_int = int(supp_str[1:len(supp_str)-1])
    frq_dict["support"].append(supp_int)

frq_df = pd.DataFrame.from_dict(frq_dict)
frq_df = frq_df.drop(index=[0])


# Run association
metric = "confidence"
threshold = 0.5
rules_new = association(frq_df, min_threshold = threshold)

# Change format of the associatotion rules
final_rules = []
for i in tqdm(range(len(rules_new))):
    antecedents, consequents, support, confidence, lift = rules_new.iloc[i]["antecedents"], rules_new.iloc[i]["consequents"], rules_new.iloc[i]["support"], rules_new.iloc[i]["confidence"], rules_new.iloc[i]["lift"]
    final_rules.append((antecedents, consequents, support, confidence, lift))


# Get evaluation
top_n = 5
precision, recall, f1_score = evaluate_recommendations(user_items, test_output, final_rules, metric, top_n=top_n)




#### MIN SUPPORT 250 TRANSACTIONS ####

# read apriori results
frq = [i.strip().split() for i in open("ndi 2/df/output3.txt").readlines()]

# transform output to df
frq_dict = {"support":[],
            "itemsets":[]}

for i in tqdm(range(len(frq))):
    itemsets_list = frq[i][0:len(frq[i])-2]
    frq_dict["itemsets"].append(frozenset([int(x) for x in itemsets_list]))
    supp_str = frq[i][len(frq[i])-2]
    supp_int = int(supp_str[1:len(supp_str)-1])
    frq_dict["support"].append(supp_int)

frq_df = pd.DataFrame.from_dict(frq_dict)
frq_df = frq_df.drop(index=[0])


# Run association
metric = "confidence"
threshold = 0.5
rules_new = association(frq_df, min_threshold = threshold)

# Change format of the associatotion rules
final_rules = []
for i in tqdm(range(len(rules_new))):
    antecedents, consequents, support, confidence, lift = rules_new.iloc[i]["antecedents"], rules_new.iloc[i]["consequents"], rules_new.iloc[i]["support"], rules_new.iloc[i]["confidence"], rules_new.iloc[i]["lift"]
    final_rules.append((antecedents, consequents, support, confidence, lift))


# Get evaluation
top_n = 5
precision, recall, f1_score = evaluate_recommendations(user_items, test_output, final_rules, metric, top_n=top_n)



#### MIN SUPPORT 425 TRANSACTIONS ####

# read apriori results
frq = [i.strip().split() for i in open("ndi 2/df/output4.txt").readlines()]

# transform output to df
frq_dict = {"support":[],
            "itemsets":[]}

for i in tqdm(range(len(frq))):
    itemsets_list = frq[i][0:len(frq[i])-2]
    frq_dict["itemsets"].append(frozenset([int(x) for x in itemsets_list]))
    supp_str = frq[i][len(frq[i])-2]
    supp_int = int(supp_str[1:len(supp_str)-1])
    frq_dict["support"].append(supp_int)

frq_df = pd.DataFrame.from_dict(frq_dict)
frq_df = frq_df.drop(index=[0])

# Run association
metric = "confidence"
threshold = 0.5
rules_new = association(frq_df, min_threshold = threshold)

# Change format of the associatotion rules
final_rules = []
for i in tqdm(range(len(rules_new))):
    antecedents, consequents, support, confidence, lift = rules_new.iloc[i]["antecedents"], rules_new.iloc[i]["consequents"], rules_new.iloc[i]["support"], rules_new.iloc[i]["confidence"], rules_new.iloc[i]["lift"]
    final_rules.append((antecedents, consequents, support, confidence, lift))

# Get evaluation
top_n = 5
precision, recall, f1_score = evaluate_recommendations(user_items, test_output, final_rules, metric, top_n=top_n)


from IPython.display import display

set(frq_items["itemsets"]).intersection(set(frq_df["itemsets"]))


np.where(frq_items["itemsets"]==frozenset({9693}))
np.where(frq_df["itemsets"]==frozenset({9693}))

frq_items.iloc[2,:]
frq_df.iloc[237,:]
set(["2"]).issubset(set(datContent_reduced[0]))
cntr = 0
for i in range(len(datContent_reduced)):
    if frozenset(["1146"]).issubset(set(datContent_reduced[i])):
        cntr += 1

cntr/len(datContent_reduced)



# read retail.dat to a list of sets
datContent = [i.strip().split() for i in open("retail.dat").readlines()]

# Convert to one hot encoded array
mlb = MultiLabelBinarizer()
one_hot_encoded = mlb.fit_transform(datContent)
one_hot_encoded = one_hot_encoded.astype(bool)
# Check for one-item baskets
basket_sizes = np.apply_along_axis(np.sum, axis=1, arr=one_hot_encoded)
# Get indices
indx = np.where(basket_sizes==1)[0]
# Remove one-item baskets
one_hot_encoded = np.delete(arr=one_hot_encoded, obj=indx, axis=0)

train_dataframe = pd.DataFrame(one_hot_encoded)

# Run apriori
min_sup = 0.005
frq_items = apriori2(train_dataframe, min_support=min_sup, use_colnames=True)
rules_new2= association(frq_items, min_threshold = 0.6)

# Change format of the associatotion rules
final_rules2 = []
for i in tqdm(range(len(rules_new2))):
    antecedents, consequents, support, confidence, lift = rules_new2.iloc[i]["antecedents"], rules_new2.iloc[i]["consequents"], rules_new2.iloc[i]["support"], rules_new2.iloc[i]["confidence"], rules_new2.iloc[i]["lift"]
    final_rules2.append((antecedents, consequents, support, confidence, lift))

# Get evaluation
top_n = 5
precision, recall, f1_score = evaluate_recommendations(user_items, test_output, final_rules2, metric, top_n=top_n)

np.sum(one_hot_encoded[:,2])/len(one_hot_encoded)



