import pkl_handler
import operator
import time
import evaluation

user_similarities = pkl_handler.load_features('../profiles/user_user_euclidean_similarities_all.pkl')
user_preferences = pkl_handler.load_features('../profiles/user_preferences.pkl')

sum_recall, sum_precision = evaluation.initialize_recall_precision()

start = time.time()

for user_id, sims in user_similarities.iteritems():

    # items that the current user liked
    prefs = user_preferences[user_id]

    # get most similar users ids
    similar_users_ids = [x[0] for x in sims]

    business_selection = [i for similar_user_id in similar_users_ids for i in user_preferences[similar_user_id]]
    # business_selecion2 = list(set([i for similar_user_id in similar_users_ids for i in user_preferences[similar_user_id]]))

    business_frequency = {i: business_selection.count(i) for i in business_selection}
    sorted_businesses = sorted(business_frequency.items(), key=operator.itemgetter(1), reverse=True)

    sum_recall, sum_precision = evaluation.evaluate_precision_recall(sorted_businesses, prefs, sum_recall, sum_precision)

    # for N in arange:
    #     relevant_selected_items = sum([1 for key, value in enumerate(sorted_businesses) if key <= N and value[0] in prefs])
    #     # print [(key, value) for key, value in enumerate(sims) if value[0] in prefs]
    #     recall = relevant_selected_items / float(len(prefs))
    #     precision = relevant_selected_items / float(N)
    #
    #     sum_recall[N] += recall
    #     sum_precision[N] += precision

    # print relevant_selected_items
    # break

evaluation.show_precision_recall(sum_recall, sum_precision, len(user_preferences))

end = time.time()

print "it tok", end - start, "seconds"
