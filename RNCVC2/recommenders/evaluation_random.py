import sqlite3

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("hausdorff", "../hausdorff.py")
load_src("pkl_handler", "../pkl_handler.py")
import pkl_handler
import evaluation
import time
import random
from sets import Set

conn = sqlite3.connect('../buff2')
c = conn.cursor()
c.execute('select distinct business_id from photo_business_features')
businesses = [b[0] for b in c.fetchall()]

user_preferences = pkl_handler.load_features('../profiles/user_preferences.pkl')

all_prefs = []

for userid, prefs in user_preferences.items():
    for b in prefs:
        all_prefs.append(b)

business_set = Set(all_prefs)

sum_recall, sum_precision = evaluation.initialize_recall_precision()

start = time.time()

for user_id, prefs in user_preferences.iteritems():
    # add 150 randomly selected businesses to the user list

    # sorted_businesses = random.sample(business_set, 21)
    sorted_businesses = [(x, 1) for x in random.sample(business_set, 21)]

    sum_recall, sum_precision = evaluation.evaluate_precision_recall(sorted_businesses, prefs, sum_recall,
                                                                     sum_precision)

evaluation.show_precision_recall(sum_recall, sum_precision, len(user_preferences))

end = time.time()

print "it tok", end - start, "seconds"