
arange = range(10, 21, 1)


def initialize_recall_precision():
    sum_recall = {}
    sum_precision = {}
    arange = range(10, 21, 1)
    for i in arange:
        sum_recall[i] = float(0)
        sum_precision[i] = float(0)

    return sum_recall, sum_precision


def evaluate_precision_recall(sorted_list, user_preferences, sum_recall, sum_precision):

    global arange

    # sum_recall = {}
    # sum_precision = {}
    # for i in arange:
    #     sum_recall[i] = float(0)
    #     sum_precision[i] = float(0)

    for N in arange:
        relevant_selected_items = sum([1 for key, value in enumerate(sorted_list) if key <= N and value[0] in
                                       user_preferences])

        recall = relevant_selected_items / float(len(user_preferences))
        precision = relevant_selected_items / float(N)

        sum_recall[N] += recall
        sum_precision[N] += precision

    return sum_recall, sum_precision


def show_precision_recall(sum_recall, sum_precision, user_preferences_lenght):

    global arange

    for N in arange:
        print "Recall at", N, "is", sum_recall[N] / user_preferences_lenght

    for N in arange:
        print "Precision at", N, "is", sum_precision[N] / user_preferences_lenght

