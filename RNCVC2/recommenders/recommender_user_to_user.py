import os
import time

import numpy as np

os.path.join('..')
from hausdorff import hausdorff
import pkl_handler

#print hausdorff(np.array([[0]]), np.array([[1]]))

#exit(0)

start = time.time()

user_profiles = pkl_handler.load_features('../profiles/user_profiles_1k.pkl')
# item_profiles = pkl_handler.load_features('profiles/business_profiles_all.pkl')

sim_list = {}

limit = 1
count_info = 0

for user_id, user_profile in user_profiles.iteritems():

    count_info += 1
    if count_info % 10 == 0:
        print count_info, "users readed"

    up = np.array(user_profile).reshape(len(user_profile), -1)

    sims = []

    for user_id2, user_profile2 in user_profiles.iteritems():

        up2 = np.array(user_profile2).reshape(len(user_profile2), -1)
        sims.append((user_id2, hausdorff(up, up2)))

    # keep only the top 100 most similar users (10%)
    sim_list[user_id] = sorted(sims, key=lambda t: t[1])[:100]

    # if limit % 10 == 0:
    #     break
    # limit += 1

pkl_handler.save_obj(sim_list, '../profiles/user_user_euclidean_similarities_all')

end = time.time()

print end-start, "seconds"
