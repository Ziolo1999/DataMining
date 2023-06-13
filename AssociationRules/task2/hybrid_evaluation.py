from itertools import chain, combinations

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
    for antecedent, consequent, support, confidence, lift, hybrid in rules:
        if antecedent.issubset(input_items) and not consequent.issubset(input_items):
            for item in consequent:
                if item not in input_items:
                    if item not in recommendations:
                        recommendations[item] = []
                    recommendations[item].append((confidence, support, lift, hybrid))
    recommendations = {
        item: (sum(conf for conf, _, _, _ in item_rules) / len(item_rules), 
               sum(sup for _, sup, _, _ in item_rules) / len(item_rules), 
               sum(lift for _, _, lift, _ in item_rules) / len(item_rules), 
               sum(hybrid for _, _, _, hybrid in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    # Added part to evaluate based on different metrices
    if metric == "confidence":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][0], -x[1][1], -x[1][2], -x[1][3]))
    elif metric == "support":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][1], -x[1][0], -x[1][2], -x[1][3]))
    elif metric == "lift":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][2], -x[1][0], -x[1][1], -x[1][3]))
    elif metric == "hybrid":
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][3], -x[1][2], -x[1][0], -x[1][1]))
    return [item for item, _ in sorted_recommendations[:top_n]]

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