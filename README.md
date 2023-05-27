# pandas-dp

pandas-dp is a differential privacy extension for Pandas. Use it simply by importing the module and instantiating a mechanism. Use the `.private` accessor on any Dataframe or Series to access differentially private versions of algorithms, or obtain noise directly.

```python
import numpy as np
import pandas as pd

# pandas-dp
import pandas_dp as pr

df = pd.DataFrame({"longitude": np.linspace(0, 10, num=5), "latitude": np.linspace(0, 20, num=5)})

#    longitude  latitude
# 0        0.0       0.0
# 1        2.5       5.0
# 2        5.0      10.0
# 3        7.5      15.0
# 4       10.0      20.0

df.longitude.mean()
# 5.0

mechanism = pr.LaplaceMechanism(epsilon=1)
df.longitude.private.mean(mechanism=mechanism)
# 4.861801498972586

# Each call is a new sample
df.longitude.private.mean(mechanism=mechanism)
# 4.927190360876554

# Noise an entire dataframe, without need to specify columns!
df_priv = df.private.noise(mechanism=mechanism)
#    longitude   latitude
# 0   2.284351   2.026169
# 1   3.645227   3.253937
# 2   6.068457  10.474715
# 3   8.633906  15.564795
# 4  10.120258  21.111389
```
