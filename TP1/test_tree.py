from decision_trees import CART, read_dataset, train_cart


def test_cart_gini():
    """
    Test a previously built CART
    """
    dataset = read_dataset()
    tree = train_cart(dataset, gini_entropy_function="GINI")
    hits = 0
    fails = 0
    for observation in dataset:
        expected = observation[-1]
        result, leaf = tree.evaluate_input(observation)
        tree.update_confusion_matrix(result, expected)
        if expected == result:
            leaf.hits += 1
            leaf.update_accuracy()
            hits += 1
        else:
            leaf.fails += 1
            leaf.update_accuracy()
            fails += 1
    accuracy = (hits / (hits + fails)) * 100
    assert accuracy == 95.25


def test_cart_entropy():
    dataset = read_dataset()
    tree = train_cart(dataset, gini_entropy_function="ENTROPY")
    hits = 0
    fails = 0
    for observation in dataset:
        expected = observation[-1]
        result, leaf = tree.evaluate_input(observation)
        tree.update_confusion_matrix(result, expected)
        if expected == result:
            leaf.hits += 1
            leaf.update_accuracy()
            hits += 1
        else:
            leaf.fails += 1
            leaf.update_accuracy()
            fails += 1
    accuracy = (hits / (hits + fails)) * 100
    assert accuracy == 94.85
