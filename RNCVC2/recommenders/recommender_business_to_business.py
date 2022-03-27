import time

import numpy as np

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

# import os
# os.path.join('..')

load_src("hausdorff", "../hausdorff.py")
import hausdorff
load_src("pkl_handler", "../pkl_handler.py")
import pkl_handler

start = time.time()

item_profiles = pkl_handler.load_features('../profiles/business_profiles_all.pkl')

sim_list = {}

limit = 1
count_info = 0

for business_id, business_profile in item_profiles.iteritems():

    count_info += 1
    if count_info % 150 == 0:
        print count_info, "businesses readed"

    bp = np.array(business_profile).reshape(len(business_profile), -1)

    sims = []

    for business_id2, business_profile2 in item_profiles.iteritems():

        bp2 = np.array(business_profile2).reshape(len(business_profile2), -1)
        sims.append((business_id2, hausdorff.hausdorff(bp, bp2)))

    sim_list[business_id] = sorted(sims, key=lambda t: t[1])[:100]

    # if limit % 100 == 0:
    #     break
    # limit += 1

pkl_handler.save_obj(sim_list, 'profiles/business_business_euclidean_similarities_100items_all')

end = time.time()

print end-start, "seconds"
