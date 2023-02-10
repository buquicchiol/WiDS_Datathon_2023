import pandas as pd
import numpy as np
from datetime import date, datetime
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

# Script Parameters
CLAMP_OR_SCALE = 'SCALE'

# Load the data
train_data = pd.read_csv('./Data/train_data.csv', index_col='index')
test_data = pd.read_csv('./Data/test_data.csv', index_col='index')
# The name of the column we are predicting
label_name = "contest-tmp2m-14d__tmp2m"
# Separate the data from the labels
label = train_data[label_name]
train_data = train_data.drop(labels=label_name, axis=1)

# Print out a summary of the data
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_data.describe())

# Explore data, lets see the weird types of columns in this dataset
for c in train_data.columns:
    if train_data[c].dtype != float:
        print(f'{c}: type={train_data[c].dtype}')

# 'climateregions__climateregion' is a categorical variable
# Let's one-hot encode both dataframes in unison with the following steps:
# Combine regions from training and testing
tr_regions = train_data['climateregions__climateregion'].values
te_regions = test_data['climateregions__climateregion'].values
all_regions = np.concatenate([tr_regions, te_regions]).reshape(-1, 1)
#   Declare and fit the one-hot encoder
ohe = OneHotEncoder(sparse=False)
ohe.fit(all_regions)
#   Transform the data
all_regions_oh = ohe.transform(all_regions)
tr_regions_ = all_regions_oh[:len(tr_regions)]
te_regions_ = all_regions_oh[len(tr_regions):]
#   Name our new columns
new_region_col_names = ['climate_region_'+r for r in list(ohe.categories_[0])]
print(f'Training Data Shape: {train_data.shape}')
print(f'One Hot Training Region Shape: {tr_regions_.shape}')
print('New Region Names shape: ', len(new_region_col_names))
print(new_region_col_names)
# Add the new columns to the dataframes
for i in range(len(new_region_col_names)):
    train_data[new_region_col_names[i]] = tr_regions_[:, i]
    test_data[new_region_col_names[i]] = te_regions_[:, i]
# Delete the old region columns
train_data.drop('climateregions__climateregion', axis=1, inplace=True)
test_data.drop('climateregions__climateregion', axis=1, inplace=True)

# Startdate categorization
Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
seasons = [('early winter', 0, (date(Y, 1, 1),  date(Y, 1, 20))),
           ('mid winter', 1, (date(Y, 1, 21),  date(Y, 2, 20))),
           ('late winter', 2, (date(Y, 2, 21),  date(Y, 3, 20))),

           ('early spring', 3, (date(Y,  3, 21),  date(Y,  4, 20))),
           ('mid spring', 4, (date(Y,  4, 21),  date(Y,  5, 20))),
           ('late spring', 5, (date(Y, 5, 21), date(Y, 6, 20))),

           ('early summer', 6, (date(Y,  6, 21),  date(Y,  7, 22))),
           ('mid summer', 7, (date(Y, 7, 21), date(Y, 8, 22))),
           ('late summer', 8, (date(Y, 8, 21), date(Y, 9, 22))),

           ('early autumn', 9, (date(Y,  9, 23),  date(Y, 10, 22))),
           ('mid autumn', 10, (date(Y, 10, 23), date(Y, 11, 22))),
           ('late autumn', 11, (date(Y, 11, 23), date(Y, 12, 20))),

           ('early winter', 0, (date(Y, 12, 21),  date(Y, 12, 31)))]


def get_season(date):
    now = datetime.strptime(date, '%m/%d/%y')
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(idx for season, idx, (start, end) in seasons
                if start <= now <= end)


train_data['startdate'] = train_data['startdate'].apply(get_season)
test_data['startdate'] = test_data['startdate'].apply(get_season)

print(np.unique(train_data['startdate'], return_counts=True))

# Iterate through all the floating point columns and clamp their values to mean +/- 2 standard deviations
col_scaling_dict = {}
for c in train_data.columns:
    if c != 'startdate' and \
       c != 'lat' and \
       c != 'lon' and \
       'climate_region' not in c:

        col_vec = train_data[c].values
        col_vec_mean = np.mean(col_vec)
        col_vec_std = np.std(col_vec)
        print(f'Column: {c} | mean={col_vec_mean}, std={col_vec_std}')

        if CLAMP_OR_SCALE == 'CLAMP':
            # Clamp the values below and above 2 stds
            new_min = col_vec_mean - 2*col_vec_std
            new_max = col_vec_mean + 2*col_vec_std
            col_vec[col_vec < new_min] = new_min
            col_vec[col_vec > new_max] = new_max
            col_scaling_dict[c] = (new_min, new_max)
            # Put the data back into the DF
            train_data[c] = col_vec

# TODO this doesnt work for test data right now
for c in test_data.columns:
    if c != 'startdate' and \
            c != 'lat' and \
            c != 'lon' and \
            'climate_region' not in c:
        col_vec = test_data[c].values
        if CLAMP_OR_SCALE == 'CLAMP':
            # Clamp the values below and above 2 stds
            col_vec[col_vec < new_min] = col_scaling_dict[c][0]
            col_vec[col_vec > new_max] = col_scaling_dict[c][1]
            # Put the data back into the DF
            test_data[c] = col_vec

# if we scale instead of clamp
if CLAMP_OR_SCALE == 'SCALE':
    cols = list(train_data.columns)  # List of all columns
    cols.remove('startdate')  # Remove the startdate col
    cols.remove('lat')  # Remove the latitude col
    cols.remove('lon')  # Remove the longitude col
    cols = [c for c in cols if 'climate_region' not in c]  # Remove our climate region cols
    sc = RobustScaler()
    data_ = train_data[cols].values
    train_data[cols] = sc.fit_transform(data_)

    test_data_ = test_data[cols].values
    test_data[cols] = sc.transform(test_data_)

    col_scaling_dict = sc

# Train/validation split (test data is separate)
x_train, x_valid, y_train, y_valid = train_test_split(train_data, label, test_size=0.33, random_state=42)

# Save our data to intermediate pickle files
with open('./tr.p', 'wb') as f:
    pickle.dump((x_train, y_train), f)
with open('./vl.p', 'wb') as f:
    pickle.dump((x_valid, y_valid), f)
with open('./te.p', 'wb') as f:
    pickle.dump(test_data.values, f)

print('Finished Preprocessing!')




