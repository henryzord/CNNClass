import time

import numpy as np

import pkl_handler
from hausdorff import hausdorff

start = time.time()

# user_sims = pkl_handler.load_features('profiles/user_business_similarities.pkl')
#
# for user_id, sims in user_sims.iteritems():
#     print user_id, sims[:10]
#
# exit()

user_profiles = pkl_handler.load_features('profiles/user_profiles_1k.pkl')
item_profiles = pkl_handler.load_features('profiles/business_profiles_all.pkl')

sim_list = {}

limit = 1

# for user_id, user_profile in user_profiles.iteritems():
#     print user_id
#     if limit % 10 == 0:
#         break
#     limit += 1
# exit()
# for business_id, business_profile in item_profiles.iteritems():
#     bp = business_profile
#     break

# print hausdorff(np.array(bp).reshape(len(bp), -1), np.array(up).reshape(len(up), -1))

# exit()

for user_id, user_profile in user_profiles.iteritems():

    up = np.array(user_profile).reshape(len(user_profile), -1)

    # sim_list[user_id] = []
    sims = []

    for business_id, business_profile in item_profiles.iteritems():

        bp = np.array(business_profile).reshape(len(business_profile), -1)
        # sim_list[user_id].append((business_id, hausdorff(up, bp)))
        sims.append((business_id, hausdorff(up, bp)))

    sim_list[user_id] = sorted(sims, key=lambda t: t[1])

    # if limit % 100 == 0:
    #     break
    # limit += 1

pkl_handler.save_obj(sim_list, 'profiles/user_business_euclidean_similarities_all')

# sorted = sorted(sim_list['Ynt9QATr3tur66_OVQNCng'], key=lambda t: t[1])
# sorted = sorted(sim_list['x1qmBwKEOFCaGiYxs6ppRw'], key=lambda t: t[1])
# sorted = sorted(sim_list['LjGpKQ7BePzV2JKjUyGaCA'], key=lambda t: t[1])

# print len(sorted)

# print [(key, value) for key, value in enumerate(sorted) if value[0] == 'waeCYkwULm9GKhFFkO3WdA']
# print [(key, value) for key, value in enumerate(sorted) if value[0] == 'p1Wb4mEH04qrUMfmPepFzw']
# print [(key, value) for key, value in enumerate(sorted) if value[0] == '8zdeaK3PWDsr5Kq4UOuNSw']

# Ynt9QATr3tur66_OVQNCng
# rated = ['3KCL_Bu8mOk-h7Bp-IXL8Q', 'ufIN_oA285Zivd-4szJKxQ', 'rcCGdKxMPJk4lCzZZ1i_bA', 'rfy_bJ5ad-gfVmMeqKtsyw',
#          'KH6stvNdq_aV0HvKw8P5yA',  '8zdeaK3PWDsr5Kq4UOuNSw', 'JdAyXD2lsYtOGUc7DTsVEQ', '2X5G4Ujq0s4Wfn4TC7gX0g',
#          '--UE_y6auTgq3FXlvUMkbw', 'p1Wb4mEH04qrUMfmPepFzw', 'waeCYkwULm9GKhFFkO3WdA', 'Ca-003BAqWW-IEsFvjlY9g',
#          'IxQ1ATP_Wg_QujO9nywzcQ', 'hryVOl_txhl9n14d224Ssw', 'hrN2jHYG5BYRrLUkav1uRQ', 'ehmtix6CHtneuYXzbrSwYw',
#          'nPvwk6NW7j-nbvUzCh94lw', 'i4O6QQe_gEr4aDSY5f4imA']

# x1qmBwKEOFCaGiYxs6ppRw
# rated = ['rfy_bJ5ad-gfVmMeqKtsyw', 'lgRLyicXByG1-W0QdsTopw', '2X5G4Ujq0s4Wfn4TC7gX0g', 'RjavzMXChoFWdsIZnfs5YQ',
#          'r9rub_ajcb-3yYsYYD5apw', '-a22bjfNCVL0gvKpd6RakA', 'rPwuIBv_mz1O_yI5R-VJ7Q']

# LjGpKQ7BePzV2JKjUyGaCA
# rated = ['vtcEXiQukuSzOMxyVxlh-Q', 'kg4AP5am_IUczgp1neOvMw', 'b-VLJb2e35qdhJmmbbZ36g']

# print [(key, value) for key, value in enumerate(sorted) if value[0] in rated]

# print sorted[:10]

end = time.time()

print end-start, "seconds"
