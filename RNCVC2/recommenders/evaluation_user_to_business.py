import pkl_handler
import evaluation
import time

user_similarities = pkl_handler.load_features('../profiles/user_business_similarities_all.pkl')
user_preferences = pkl_handler.load_features('../profiles/user_preferences.pkl')

start = time.time()

sum_recall, sum_precision = evaluation.initialize_recall_precision()

for user_id, sims in user_similarities.iteritems():

    prefs = user_preferences[user_id]

    # for N in arange:
    #     relevant_selected_items = sum([1 for key, value in enumerate(sims) if key <= N and value[0] in prefs])
    #     # print [(key, value) for key, value in enumerate(sims) if value[0] in prefs]
    #     recall = relevant_selected_items / float(len(prefs))
    #
    #     sum_recall[N] += recall

    sum_recall, sum_precision = evaluation.evaluate_precision_recall(sims, prefs, sum_recall, sum_precision)
    # sum_recall += recall

    # print relevant_selected_items
    # break

evaluation.show_precision_recall(sum_recall, sum_precision, len(user_preferences))

end = time.time()

print "it tok", end - start, "seconds"
