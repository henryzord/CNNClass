
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))


# load_src("hausdorff", "../hausdorff.py")
load_src("pkl_handler", "../pkl_handler.py")

import pkl_handler
import operator
import evaluation
import time

user_preferences = pkl_handler.load_features('../profiles/user_preferences.pkl')
business_similarities = pkl_handler.load_features('../profiles/business_business_euclidean_similarities_all.pkl')

sum_recall, sum_precision = evaluation.initialize_recall_precision()

start = time.time()

count = 0

# for key, value in business_similarities.items():
#     if key in ['0z60l8uFoLa0qq_h5x5zhw', 'tXeXtTOvVLm_xGgaMEuoTA', 'VZUbvO-m-ceuoMPRM3-NrQ', 'R9Tll6IKpJtYcsbEv7edCA',
#                '0pGLXuFpUNTgArmOgbBpwg', 'phJiywlkjztj9jQETBO', 'fx4coO0OyW7Qe8vdLnlLiA', 't3rAWJwdiBfXPery6nDZHw',
#                'dfkmSx2CkO_ORwujV5p5Ag', 'mGqNm8kEq4ZVB1fKLMzImg']:
#         print key, value[2:5]
#     count += 1
#     if count == 1000:
#         exit()

for user_id, prefs in user_preferences.iteritems():

    business_selection = [similar_business_id for business_id in prefs for similar_business_id, similarity in
                          business_similarities[business_id]]

    business_frequency = {i: business_selection.count(i) for i in business_selection}
    sorted_businesses = sorted(business_frequency.items(), key=operator.itemgetter(1), reverse=True)

    # print len(business_selection), business_selection
    # print
    print len(sorted_businesses), sorted_businesses[:10]
    exit()
    #
    # if count == 50:
    #     exit()
    # else:
    #     count += 1
    #     continue

    sum_recall, sum_precision = evaluation.evaluate_precision_recall(sorted_businesses, prefs, sum_recall, sum_precision)

    # for N in arange:
    #     relevant_selected_items = sum([1 for key, value in enumerate(sorted_businesses) if key <= N and value[0] in prefs])
    #     recall = relevant_selected_items / float(len(prefs))
    #     precision = relevant_selected_items / N
    #     sum_recall[N] += recall
    #     sum_precision[N] += precision
    # print [(key, value) for key, value in enumerate(sims) if value[0] in prefs]

    # print relevant_selected_items
    # break

evaluation.show_precision_recall(sum_recall, sum_precision, len(user_preferences))

end = time.time()

print "it tok", end - start, "seconds"
