#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[ ]:


#get_ipython().system('wget -q https://raw.githubusercontent.com/kairess/toy-datasets/master/tn_travel_%E1%84%8B%E1%85%A7%E1%84%92%E1%85%A2%E1%86%BC_A.csv')
#get_ipython().system('wget -q https://raw.githubusercontent.com/kairess/toy-datasets/master/tn_traveller_master_%E1%84%8B%E1%85%A7%E1%84%92%E1%85%A2%E1%86%BC%E1%84%80%E1%85%A2%E1%86%A8%20Master_A.csv')
#get_ipython().system('wget -q https://raw.githubusercontent.com/kairess/toy-datasets/master/tn_visit_area_info_%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%B5%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%87%E1%85%A9_A.csv')


# In[ ]:


#get_ipython().run_line_magic('pip', 'install pandas')


# In[3]:


import pandas as pd


# In[5]:


df_place = pd.read_csv('tn_visit_area_info_방문지정보_A.csv')
df_traveler = pd.read_csv('tn_traveller_master_여행객 Master_A.csv')
df_travel = pd.read_csv('tn_travel_여행_A.csv')
#df_city_info = pd.read_csv('tc_sgg_시군구코드.csv')


# In[ ]:


df_place


# In[ ]:


filtered_df = df_place[df_place['VISIT_AREA_TYPE_CD']==11]
filtered_df


# In[ ]:


df_place['VISIT_AREA_TYPE_CD'].unique()


# In[ ]:


df_travel


# In[ ]:


df_traveler


# In[ ]:


df_traveler['AGE_GRP'].unique()


# In[ ]:


#df_city_info


# In[ ]:


df = pd.merge(df_place, df_travel, on='TRAVEL_ID', how='left')
df = pd.merge(df, df_traveler, on='TRAVELER_ID', how='left')

df


# In[ ]:


df[df['TRAVEL_ID'] == 'a_a015688']


# In[ ]:


df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()

df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

df_filter


# In[ ]:


df_filter = df_filter[[
    'GENDER',
    'AGE_GRP',
    'TRAVEL_STYL_1', 
    'TRAVEL_STYL_2', 
    'TRAVEL_STYL_3', 
    'TRAVEL_STYL_4', 
    'TRAVEL_STYL_5', 
    'TRAVEL_STYL_6', 
    'TRAVEL_STYL_7', 
    'TRAVEL_STYL_8',
    'VISIT_AREA_NM',
    'DGSTFN',
]]

#df_filter.loc[:, 'GENDER'] = df_filter['GENDER'].map({'남': 0, '여': 1})

df_filter = df_filter.dropna()

df_filter

"""
age_grp : 나이대
travel_style_1 : 자연 <> 도시
travel_style_2 : 숙박 <> 당일
travel_style_3 : 새로운 <> 익숙한
travel_style_4 : 편함비쌈숙소 <> 불편함저렴함숙소
travel_style_5 : 휴양휴식 <> 엑티비티
travel_style_6 : 안유명한 <> 유명한
travel_style_7 : 계획 <> 상황
travel_style_8 : 사진중요 <> 사진안중요
DGSTFN : 만족도
"""


# In[ ]:


categorical_features_names = [
    'GENDER',
    # 'AGE_GRP',
    'TRAVEL_STYL_1', 
    'TRAVEL_STYL_2', 
    'TRAVEL_STYL_3', 
    'TRAVEL_STYL_4', 
    'TRAVEL_STYL_5', 
    'TRAVEL_STYL_6', 
    'TRAVEL_STYL_7', 
    'TRAVEL_STYL_8',
    'VISIT_AREA_NM',
    # 'DGSTFN',
]

df_filter[categorical_features_names[1:-1]] = df_filter[categorical_features_names[1:-1]].astype(int)

df_filter


# In[ ]:


#%pip install scikit-learn


# In[ ]:


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df_filter, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)


# In[ ]:


test_data


# In[ ]:


#%pip install -q catboost


# In[ ]:


from catboost import CatBoostRegressor, Pool

train_pool = Pool(train_data.drop(['DGSTFN'], axis=1),
                  label=train_data['DGSTFN'],
                  cat_features=categorical_features_names)

test_pool = Pool(test_data.drop(['DGSTFN'], axis=1),
                 label=test_data['DGSTFN'],
                 cat_features=categorical_features_names)


# In[ ]:


#%pip install ipywidgets


# In[ ]:


model = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='MAE',
    task_type='CPU',
    depth=6,
    learning_rate=0.01,
    n_estimators=3000)

model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=500,
    plot=True)


# In[ ]:


from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

y_pred = model.predict(test_pool)

y_test = test_pool.get_label()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

feature_importances = model.get_feature_importance()
features = train_pool.get_feature_names()

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()


# In[ ]:


test_data.iloc[0]


# In[ ]:


model.predict(test_data.iloc[0].drop(['DGSTFN']))


# In[ ]:


i = 22385

print(test_data.loc[i])

print(model.predict(test_data.loc[i].drop(['DGSTFN'])))


# In[ ]:


model.get_feature_importance(prettified=True)


# In[ ]:


area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()

area_names


# In[ ]:



#임의의 입력 설정
traveler = {
    'GENDER': '1',
    'AGE_GRP': 20.0,
    'TRAVEL_STYL_1': 7,
    'TRAVEL_STYL_2': 1,
    'TRAVEL_STYL_3': 1,
    'TRAVEL_STYL_4': 1,
    'TRAVEL_STYL_5': 7,
    'TRAVEL_STYL_6': 7,
    'TRAVEL_STYL_7': 7,
    'TRAVEL_STYL_8': 1,
}


results = pd.DataFrame([], columns=['AREA', 'SCORE'])

for area in area_names['VISIT_AREA_NM']:
    input = list(traveler.values())
    input.append(area)

    score = model.predict(input)

    results = pd.concat([results, pd.DataFrame([[area, score]], columns=['AREA', 'SCORE'])])

print(results.sort_values('SCORE', ascending=False)[:20])


# In[ ]:




