# %%
# UCI dataset UKM "user knowledge modeling"

# need: pip install ucimlrepo
# ( no registration )
#
# or:
# https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling

# %% [markdown]
# Information on variables see also on web page:
# https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling

# %% [markdown]
# - STG (The degree of study time for goal object materails), (input value)
# - SCG (The degree of repetition number of user for goal object materails) (input value)
# - STR (The degree of study time of user for related objects with goal object) (input value)
# - LPR (The exam performance of user for related objects with goal object) (input value)
# - PEG (The exam performance of user for goal objects) (input value)
# - UNS (The knowledge level of user) (target value)

# %% [markdown]
# Class Labels (frow website)
#
# - Very Low: 50
# - Low:129
# - Middle: 122
# - High 130

# %%
from ucimlrepo import fetch_ucirepo

import pandas as pd

# %%
# fetch dataset
ukm = fetch_ucirepo(id=257)

# %%
# data (as pandas dataframes)
X = ukm.data.features
y = ukm.data.targets


# %% [markdown]
# ## Optional: write Excel file

# %%
#
df = X.join(y)
df

# %%
df.to_excel('uci_ukm.xlsx', index=False)

# %% [markdown]
# ## Optional: Show column info

# %%
# show additional_info
ukm.metadata['additional_info']['variable_info'].split('\n')


