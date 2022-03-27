import sqlite3
import time
import numpy as np
import save_file

conn = sqlite3.connect('buff2')

sql = "select distinct business_id " \
      "from photo_business_features u"

sql_features = "select photo_id " \
               "from photo_business_features p " \
               "where business_id = ?"

sql_f = "select features from photo_business_features p where photo_id = ?"

c = conn.cursor()

start = time.time()

business_profiles = {}

for business in c.execute(sql):
    c = conn.cursor()
    c.execute(sql_features, (business[0],))
    photo_ids = [f[0] for f in c.fetchall()]

    business_profiles[business[0]] = []

    for p in photo_ids:
        x = c.execute(sql_f, (p,))
        v = x.fetchone()[0]
        business_profiles[business[0]].append(np.fromstring(v, dtype=np.float_, sep=","))

end = time.time()
conn.close()

save_file.save_obj(business_profiles, 'business_profiles_all')

print end - start, "seconds"
