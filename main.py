import kagglegym
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor



# Kaggle Environment #
env = kagglegym.make()
observation = env.reset()
# End of Kaggle Environment #

# Feature Selection #

null_labels = [
    'technical_21',
    'technical_19',
    'technical_27',
    'technical_36',
    'technical_35',
    'technical_17',
    'technical_43',
    'technical_13',
    'fundamental_33',
    'technical_14',
    'technical_33',
    'fundamental_18',
    'fundamental_48',
    'fundamental_59',
    'technical_9',
    'technical_16',
    'technical_42',
    'technical_18',
    'fundamental_42',
    'fundamental_0',
    'fundamental_7',
    'fundamental_41',
    'technical_41',
    'fundamental_21',
    'fundamental_19',
    'technical_29',
    'technical_24',
    'derived_0',
    'derived_1',
    'fundamental_17',
    'technical_3',
    'fundamental_20',
    'fundamental_32',
    'fundamental_62',
    'fundamental_25',
    'technical_1',
    'fundamental_58',
    'derived_3',
    'technical_5',
    'fundamental_52',
    'technical_10',
    'technical_31',
    'technical_25',
    'technical_44',
    'technical_28',
    'fundamental_40',
    'fundamental_27',
    'fundamental_29',
    'fundamental_43',
    'fundamental_15',
    'fundamental_30',
    'fundamental_60',
    'fundamental_16',
    'fundamental_50',
    'fundamental_44',
    'fundamental_37',
    'fundamental_14',
    'fundamental_23',
    'fundamental_55',
    'fundamental_8',
    'fundamental_63',
    'fundamental_39',
    'fundamental_54',
    'derived_2',
    'derived_4',
    'fundamental_35',
    'fundamental_34',
    'fundamental_47',
    'fundamental_51',
    'fundamental_31',
    'fundamental_49',
    'fundamental_22',
    'fundamental_9',
    'fundamental_24',
    'fundamental_57',
    'fundamental_28',
    'fundamental_61',
    'fundamental_1',
    'fundamental_6',
    'fundamental_38',
    'fundamental_5']
etr_features = ['y_past',
 'tec20-30',
 'technical_30',
 'tec123',
 'technical_43',
 'technical_43_diff',
 'tec123_past',
 'technical_11_diff',
 'technical_2_diff',
 'technical_11',
 'technical_20',
 'technical_2',
 'fundamental_25_nan',
 'technical_14_diff',
 'technical_21_diff',
 'technical_9_nan',
 'technical_40',
 'technical_30_diff',
 'technical_6_diff',
 'technical_6',
 'technical_17_diff',
 'technical_17',
 'technical_14',
 'technical_7',
 'technical_19',
 'technical_44_nan',
 'fundamental_27_nan',
 'technical_18_nan',
 'technical_28_nan',
 'technical_21',
 'technical_42_nan',
 'technical_29_diff',
 'technical_20_diff',
 'technical_31_nan',
 'fundamental_53',
 'technical_24_nan',
 'technical_36',
 'technical_19_diff',
 'technical_27',
 'technical_29',
 'technical_35',
 'technical_22',
 'technical_41_nan',
 'fundamental_8',
 'fundamental_21',
 'fundamental_17_nan',
 'technical_34',
 'technical_16_nan',
 'technical_27_diff',
 'fundamental_33_nan',
 'fundamental_58',
 'derived_1_nan',
 'technical_10',
 'technical_25_nan',
 'fundamental_18',
 'fundamental_59',
 'technical_40_diff',
 'null_count',
 'fundamental_5_nan',
 'fundamental_48',
 'fundamental_47_nan',
 'technical_36_diff',
 'fundamental_41_nan',
 'fundamental_42_nan',
 'fundamental_0_nan',
 'fundamental_50',
 'fundamental_40',
 'technical_3_nan',
 'fundamental_23',
 'fundamental_49_nan',
 'fundamental_36',
 'technical_44',
 'fundamental_2',
 'fundamental_0',
 'technical_41',
 'fundamental_62_diff',
 'technical_38_diff',
 'fundamental_22_nan',
 'technical_12',
 'fundamental_62',
 'technical_37_diff',
 'fundamental_44',
 'technical_29_nan',
 'fundamental_24_nan',
 'technical_10_nan',
 'fundamental_46',
 'technical_1',
 'fundamental_54_nan',
 'fundamental_0_diff',
 'technical_12_diff',
 'technical_35_diff',
 'derived_3_nan',
 'fundamental_63_nan',
 'fundamental_31_nan',
 'fundamental_40_nan',
 'fundamental_35_nan',
 'technical_3',
 'fundamental_13']
 
xgb_features = ['technical_38_diff',
 'technical_17_diff',
 'technical_14_diff',
 'technical_35_diff',
 'technical_11_diff',
 'fundamental_50',
 'fundamental_35_nan',
 'fundamental_48',
 'technical_29_diff',
 'tec123',
 'technical_41',
 'technical_19',
 'technical_2',
 'technical_11',
 'technical_35',
 'fundamental_2',
 'fundamental_0_diff',
 'fundamental_36',
 'technical_37_diff',
 'technical_30_diff',
 'fundamental_44',
 'technical_21',
 'technical_36',
 'technical_40',
 'technical_1',
 'fundamental_23',
 'technical_27_diff',
 'technical_7',
 'technical_30',
 'null_count',
 'technical_9_nan',
 'fundamental_18',
 'technical_43_diff',
 'tec123_past',
 'technical_12_diff',
 'fundamental_13',
 'fundamental_58',
 'technical_40_diff',
 'fundamental_8',
 'technical_41_nan',
 'technical_17',
 'technical_36_diff',
 'technical_43',
 'technical_31_nan',
 'technical_21_diff',
 'technical_20_diff',
 'y_past',
 'technical_27',
 'fundamental_62_diff',
 'fundamental_21',
 'tec20-30',
 'technical_3',
 'technical_19_diff',
 'fundamental_53',
 'technical_2_diff']
seed = 17
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
origin_features = [c for c in observation.train.columns if c not in excl]
origin_features_exclude_y = [c for c in observation.train.columns if c not in ['y']]
diff_features = [feature + '_diff' for feature in origin_features]
# xgb_features = diff_features[:10] + normal_features[:-10] + nan_features[:-10] #+ normal_features[20::2]
# etr_features = diff_features[:2] +  normal_features + nan_features
linear_features = ['technical_20_diff', 'tec20-30']
# End of Feature Selection #
d_mean = observation.train.median(axis=0)

last_stamp = observation.train.loc[observation.train.timestamp ==
                                   observation.train.timestamp.max(), origin_features_exclude_y]
                                   
# add diffs #
def add_diff(data):
    data.sort_values(['id', 'timestamp'], inplace=True)
    data['id_diff'] = data.id.diff()
    for feature in origin_features:
        diff_tag = feature + '_diff'
        data[diff_tag] = data[feature].diff()
        d_mean[diff_tag] = 0
    data.loc[data.id_diff != 0, diff_features] = 0
# end of diffs #


# add Nan tags #
def add_nan(data):
    Nan_counts = data.isnull().sum(axis=1)
    for feature in null_labels:
        data[feature + '_nan'] = pd.isnull(data[feature])
        d_mean[feature + '_nan'] = 0
    data['null_count'] = Nan_counts
# end of Nan tags #

def R_sign(y_pred, y):
    '''
    input: ypred, y
    return: R2 score of prediction
    '''
    u = np.mean(y)
    R2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - u))
    R = np.sign(R2) * np.sqrt(np.abs(R2))
    return R


def predict_y_past(x):
    # w =  np.array([-8.36489105,  9.19544792]).T
    w = np.array([-8.08872128,  8.89837742]).T
    c = -0.00026617524826966221
    return x.dot(w) + c #- 0.00020021698404804056
    
    
ymean_dict = dict(observation.train.groupby(["id"])["y"].median())

print('Processing data...')

train = observation.train
add_diff(train)
add_nan(train)
train = train.fillna(d_mean)

train['tec20-30'] = train.technical_20 - train.technical_30
train['tec123'] = train['tec20-30'] + train.technical_13
train['tec123_past'] = train.tec123.shift()
train['y_past'] = train.y.shift()
train.loc[train.id_diff != 0, ['tec123_past', 'y_past']] = 0

low_y_cut = -0.075
high_y_cut = 0.075
y_above_cut = (train.y > high_y_cut)
y_below_cut = (train.y < low_y_cut)
y_within_cut = (~y_above_cut & ~y_below_cut)
train.fillna(0, inplace=True)
# Generate models...
ridge_1 = Ridge()
ridge_2 = Ridge()
etr = ExtraTreesRegressor(n_estimators=248, max_depth=6, min_samples_leaf=27, max_features=0.6, n_jobs=-1, random_state=seed, verbose=0)
xgb = xgb = XGBRegressor(n_estimators=80, nthread=-1, max_depth=3, learning_rate=0.1, reg_lambda=1, subsample=1.0,
                   colsample_bytree=0.5, seed=seed)

print('Training Linear Model...\n', len(linear_features), 'features')
ridge_2.fit(train.loc[y_within_cut, linear_features], train.loc[y_within_cut, 'y'])
ridge_1.fit(np.array(train.loc[y_within_cut, linear_features[0]]).reshape(-1, 1), train.loc[y_within_cut, 'y'])

print('Training XGBoost Model...\n', len(xgb_features), 'features')
xgb.fit(train[xgb_features], train.y)

print('Training ETR Model...\n', len(etr_features), 'features')
etr.fit(train[etr_features], train.y)
# end of Generate models.
# full_df = pd.read_hdf('../input/train.h5')

train = 0
w_etr = 0.38
w_lr = 1
w_xgb = 0.38

reward = -1
# predicting...
print('Predicting...')
r_true = []
y_lr_1_p=0
y_lr_2_p=0
a = 0.5
b = 0.5
while True:
    timestamp = observation.features.timestamp[0]
    test = observation.features
    test = pd.concat([test, last_stamp])
    
    add_diff(test)
    test['tec20-30'] = test.technical_20 - test.technical_30
    test['tec123'] = test['tec20-30'] + test.technical_13
    test['tec123_past'] = test.tec123.shift()
    test.loc[test.id_diff != 0, 'tec123_past'] = 0
    test = test.loc[test.timestamp == timestamp]
    test.sort_index(inplace=True)
    add_nan(test)
    test.fillna(d_mean, inplace=True)
   
    test['y_past'] = predict_y_past(test[['tec123_past', 'tec123']])
    test.fillna(0, inplace=True)
    last_stamp = test.loc[test.timestamp == timestamp, origin_features_exclude_y]
    
    pred = observation.target
    
    # y_t = full_df.loc[full_df.timestamp == timestamp, 'y']
    y_etr = etr.predict(test[etr_features])
    y_xgb = xgb.predict(test[xgb_features]).clip(low_y_cut, high_y_cut)
    y_lr_2 = ridge_2.predict(test[linear_features]).clip(low_y_cut, high_y_cut)
    y_lr_1 = ridge_1.predict(np.array(test[linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    # r_1 = R_sign(y_lr_1, y_etr)
    # r_2 = R_sign(y_lr_2, y_etr)
    # r_1_t = R_sign(y_lr_1, y_t)
    # r_2_t = R_sign(y_lr_2, y_t)
    # r_true.append((r_1 > r_2) == (r_1_t > r_2_t))
    pred['y'] = y_lr_1 * 0.04 + y_lr_2 * 0.14 + y_etr * 0.54 + y_xgb * 0.28
    pred['y'] = pred.apply(lambda r: 0.98 * r['y'] + 0.02 * ymean_dict[r['id']]
                           if r['id'] in ymean_dict else r['y'], axis=1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]

    observation, reward, done, info = env.step(pred)
    if done:
        print("R score ...", info["public_score"])
        # print("true predict", np.array(r_true).mean())
        break
    if timestamp % 100 == 0:
        print('timestamp:', timestamp, '---->', reward)
        # print("true predict", np.array(r_true).mean())