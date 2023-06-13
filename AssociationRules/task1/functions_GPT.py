from itertools import chain, combinations, filterfalse

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

def apriori(transactions, min_support):
    items = set(chain(*transactions))
    itemsets = [frozenset([item]) for item in items]
    itemsets_by_length = [set()]
    k = 1
    while itemsets:
        support_count = itemsets_support(transactions, itemsets, min_support)
        itemsets_by_length.append(set(support_count.keys()))
        k += 1
        itemsets = join_set(itemsets, k)
    frequent_itemsets = set(chain(*itemsets_by_length))
    return frequent_itemsets, itemsets_by_length



def association_rules(transactions, min_support, min_confidence):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, confidence))
    return rules

def recommend_items(input_items, rules, top_n=5):
    recommendations = {}
    for antecedent, consequent, support, confidence in rules:
        if antecedent.issubset(input_items) and not consequent.issubset(input_items):
            for item in consequent:
                if item not in input_items:
                    if item not in recommendations:
                        recommendations[item] = []
                    recommendations[item].append((confidence, support))
    recommendations = {
        item: (sum(conf for conf, _ in item_rules) / len(item_rules), sum(sup for _, sup in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][0], -x[1][1]))
    return [item for item, _ in sorted_recommendations[:top_n]]

def evaluate_recommendations(test_data, rules, user_items, top_n=5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for user, true_items in test_data.items():
        # Assuming user_items is a dictionary with user IDs as keys and their associated items as values
        input_items = user_items[user]
        # Get recommendations for the user
        recommended_items = set(recommend_items(input_items, rules, top_n=top_n))
        true_items = set(true_items)
        true_positives += len(recommended_items.intersection(true_items))
        false_positives += len(recommended_items - true_items)
        false_negatives += len(true_items - recommended_items)
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

