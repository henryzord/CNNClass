import sqlite3
import time
import numpy as np
import save_file

conn = sqlite3.connect('buff2')

sql = "select distinct user_id " \
      "from user_to_business u " \
      "limit 1000"
      # "join photo_business_features pb on u.business_id = pb.business_id " \
      # "order by user_id, u.business_id"

sql_features = "select photo_id " \
               "from photo_business_features p " \
               "join user_to_business u on u.business_id = p.business_id " \
               "where user_id = ?"

sql_f = "select features from photo_business_features p where photo_id = ?"

c = conn.cursor()

user_profiles = {}
start = time.time()

for user in c.execute(sql):
    c = conn.cursor()
    c.execute(sql_features, (user[0],))
    photo_ids = [f[0] for f in c.fetchall()]

    user_profiles[user[0]] = []

    for p in photo_ids:
        x = c.execute(sql_f, (p,))
        v = x.fetchone()[0]
        user_profiles[user[0]].append(np.fromstring(v, dtype=np.float_, sep=","))

end = time.time()
conn.close()

print end - start, "seconds"

save_file.save_obj(user_profiles, 'user_profiles_1k')
