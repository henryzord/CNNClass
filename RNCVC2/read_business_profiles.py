import sqlite3
import time
import load_file
import save_file

feature_list = load_file.load_features('list_one_dim.pkl')

conn = sqlite3.connect('buff2')

sql = "select distinct business_id " \
      "from photo_business_features u "
      # "join photo_business_features pb on u.business_id = pb.business_id " \
      # "order by user_id, u.business_id"

sql_features = "select photo_id " \
               "from photo_business_features p " \
               "where business_id = ?"

c = conn.cursor()
# c.execute(sql)
# res = c.fetchall()

# limit = 1000

start = time.time()

business_profiles = {}

for business in c.execute(sql):
    c = conn.cursor()
    c.execute(sql_features, (business[0],))
    photo_ids = [f[0] for f in c.fetchall()]

    business_profiles[business[0]] = [x[1] for x in feature_list if x[0] in photo_ids]

    # if limit % 1000 == 0:
    #     break

end = time.time()

conn.close()

# for user, features in user_profiles.iteritems():
#     print user
#     print len(features)

save_file.save_obj(business_profiles, 'business_profiles')

print end - start, "seconds"
