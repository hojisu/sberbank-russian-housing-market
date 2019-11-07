
import numpy as np
import pandas as pd
import utils.correlation as corr
import statsmodels.api as sm
import utils.statsmodel_helper as smh
from sklearn.preprocessing import OneHotEncoder

def merge(df_1, df_2, on_col):
    df_tm = pd.merge(df_1, df_2, on=[on_col, on_col])
    df_tm_cols = df_tm.columns.tolist()
    df_tm_cols = df_tm_cols[:290] + df_tm_cols[291:] + [df_tm_cols[290]]
    df_tm = df_tm[df_tm_cols]
    return df_tm

def clean_data(df):
    # build_year 1500이전 nan으로
    df.loc[df.build_year < 1500, 'build_year'] = np.nan
    df.loc[df.build_year > 2016, 'build_year'] = np.nan
    
    # floor가 0이면 nan으로
    df.loc[df.floor==0, 'floor'] = np.nan
    
    # max_floor가 0이면 nan으로
    df.loc[df.max_floor==0, 'max_floor'] = np.nan
    
    # max_floor가 floor보다 크면 nan으로
    df.loc[df.floor>df.max_floor, 'max_floor'] = np.nan
    
    # full_sq, life_sq, kitch_sq가 0이면 nan으로
    df.loc[df.full_sq==0, 'full_sq'] = np.nan
    df.loc[df.life_sq==0, 'life_sq'] = np.nan
    df.loc[df.kitch_sq==0, 'kitch_sq'] = np.nan
    
    # full_sq가 life_sq보다 작으면 nan으로
    df.loc[df.life_sq>df.full_sq, 'life_sq'] = np.nan
    
    # kitch_sq가 life_sq보다 크면 nan으로
    df.loc[df.kitch_sq>df.life_sq, 'kitch_sq'] = np.nan
    
    df.loc[df.state == 33, 'state'] = 3
    
    df.loc[df.full_sq > 210, 'full_sq'] == np.nan
    df.loc[df.full_sq > 200, 'full_sq'] == np.nan    

    df.loc[df.num_room < 0, 'num_room'] = np.nan
    
    df['material'].fillna(0, inplace=True)
    
    # 이상한 숫자값들 45,34 ...
    if 'modern_education_share' in df: del df['modern_education_share']
    if 'old_education_build_share' in df: del df['old_education_build_share']
    if 'child_on_acc_pre_school' in df: del df['child_on_acc_pre_school']
        
    consts = [col for col in df.columns if len(df[col].value_counts().index) == 1]
    for const in consts:
        del df[const]
        
    df = df.replace(['no data'], ['nodata'])
    
#     # 뉴머릭한 카테고리컬 독립변수들인데 유니크값이 너무 많아서 없앤다.
#     del df['ID_railroad_station_walk']
#     del df['ID_railroad_station_avto']
#     del df['ID_big_road1']
#     del df['ID_big_road2']
#     del df['ID_railroad_terminal']
#     del df['ID_bus_terminal']
#     del df['ID_metro']
#     # too many dummy variables
#     del df['sub_area']
    
#     50% 이상 미싱 데이터가 있으면 없애버린다
    if 'provision_retail_space_sqm' in df: del df['provision_retail_space_sqm']
    if 'theaters_viewers_per_1000_cap' in df: del df['theaters_viewers_per_1000_cap']
    if 'museum_visitis_per_100_cap' in df: del df['museum_visitis_per_100_cap']
    
    # material은 카테고리
#     df['material'] = df['material'].astype(np.str, copy=False)
#     df['material'] = df['material'].replace([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['a', 'b', 'c', 'd', 'e', 'f', 'e'])
#     return df


def col_renames(df):
    df.rename(columns={'build_count_1921-1945': 'build_count_1921_1945', 'build_count_1946-1970': 'build_count_1946_1970', 'build_count_1971-1995': 'build_count_1971_1995'}, inplace=True)
    return df

def del_many_unique(df):
 # 뉴머릭한 카테고리컬 독립변수들인데 유니크값이 너무 많아서 없앤다.
    del df['ID_railroad_station_walk']
    del df['ID_railroad_station_avto']
    del df['ID_big_road1']
    del df['ID_big_road2']
    del df['ID_railroad_terminal']
    del df['ID_bus_terminal']
    del df['ID_metro']
    # too many dummy variables
    del df['sub_area']
    del df['prom_part_3000']
    return df

def categorize(df):
    df['material'] = df['material'].astype(np.object, copy=False)
    df['material'] = df['material'].replace([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['a', 'b', 'c', 'd', 'e', 'f', 'e'])
# def find_missing_data_columns(df):
#     missing_df = df.isnull().sum(axis=0).reset_index()
#     missing_df.columns = ['missing_column', 'missing_count']
#     missing_df = missing_df.loc[missing_df['missing_count'] > 0]
#     return missing_df


def impute_num_mode(df):
    for col in df._get_numeric_data().columns[df._get_numeric_data().isnull().any()]:
        df[col].fillna(df[col].mean(), inplace=True)

def imput_cat_mode(df):
    for col in df.column[df.isnull().any()].tolist():
        df[col].fillna(df[col].mean(), inplace=True)
        
def apply_log(df, numeric_cols):
    for col in numeric_cols:
        min_val = min(df[col].value_counts().index)
        if min_val < 0:
            df[col] -= min_val
            df[col] += 1
        else:
            df[col] += 1
    df[numeric_cols].apply(np.log)

def scale_up_positive(df, numeric_cols):
    for col in numeric_cols:
        min_val = min(df[col].value_counts().index)
        if min_val < 0:
            df[col] -= min_val
            df[col] += 1
        else:
            df[col] += 1
            
def remove_outliers(df, formula, repeat=1):
    result = None
    for i in range(repeat):
        model = sm.OLS.from_formula(formula, data=df)
        result = model.fit()
        influence = result.get_influence()
        distances, pvalues = influence.cooks_distance
        threshold = 4/(len(distances) - len(df.columns.drop(['_price_doc']))-1)
        outliers = [idx for idx, d in enumerate(distances) if d > threshold]
        df.drop(df.index[outliers], inplace=True)
    return df, model, result

def remove_features_by_vif(df):
    features_to_remove = [ 
        'raion_popul', \
        'preschool_education_centers_raion', \
        'school_education_centers_raion', \
        'sport_objects_raion', \
        'office_raion', \
        'young_all', \
        'work_all', \
        'ekder_all', \
        '0_17_all', \
        'raion_build_count_with_material_info', \
        'raion_build_count_with_builddate_info', \
        'build_count_1946-1970', \
        'metro_min_avto', \
        'metro_km_avto', \
        'metro_min_walk', \
        'school_km', \
        'park_km', \
        'railroad_station_walk_min', \
        'railroad_station_avto_min', \
        'ttk_km', \
        'sadovoe_km', \
        'bulvar_ring_km', \
        'kremlin_km', \
        'zd_vokzaly_avto_km', \
        'bus_terminal_avto_km', \
        'oil_chemistry_km', \
        'nuclear_reactor_km', \
        'radiation_km', \
        'power_transmission_line_km', \
        'thermal_power_plant_km', \
        'ts_km', \
        'swim_pool_km', \
        'ice_rink_km', \
        'stadium_km', \
        'basketball_km', \
        'detention_facility_km', \
        'public_healthcare_km', \
        'university_km', \
        'workplaces_km', \
        'shopping_centers_km', \
        'preschool_km', \
        'big_church_km', \
        'mosque_km', \
        'theater_km', \
        'museum_km', \
        'exhibition_km', \
        'cafe_count_500', \
        'cafe_sum_500_min_price_avg', \
        'cafe_avg_price_500', \
        'office_count_1000', \
        'cafe_count_1000', \
        'cafe_sum_1000_min_price_avg', \
        'cafe_sum_1000_max_price_avg', \
        'cafe_avg_price_1000', \
        'cafe_count_1000_na_price', \
        'cafe_count_1000_price_1000', \
        'cafe_count_1000_price_1500', \
        'office_count_1500', \
        'cafe_count_1500', \
        'cafe_sum_1500_max_price_avg', \
        'cafe_avg_price_1500', \
        'cafe_count_1500_na_price', \
        'cafe_count_1500_price_500', \
        'cafe_count_1500_price_1000', \
        'cafe_count_1500_price_1500', \
        'cafe_count_1500_price_2500', \
        'cafe_count_1500_price_high', \
        'leisure_count_1500', \
        'sport_count_1500', \
        'green_part_2000', \
        'office_count_2000', \
        'office_sqm_2000', \
        'trc_count_2000', \
        'cafe_count_2000', \
        'cafe_sum_2000_min_price_avg', \
        'cafe_sum_2000_max_price_avg', \
        'cafe_avg_price_2000', \
        'cafe_count_2000_na_price', \
        'cafe_count_2000_price_500', \
        'cafe_count_2000_price_1000', \
        'cafe_count_2000_price_1500', \
        'cafe_count_2000_price_2500', \
        'cafe_count_2000_price_high', \
        'sport_count_2000', \
        'green_part_3000', \
        'office_count_3000', \
        'office_sqm_3000', \
        'trc_count_3000', \
        'cafe_count_3000', \
        'cafe_count_3000_na_price', \
        'cafe_count_3000_price_500', \
        'cafe_count_3000_price_1000', \
        'cafe_count_3000_price_1500', \
        'cafe_count_3000_price_2500', \
        'cafe_count_3000_price_4000', \
        'cafe_count_3000_price_high', \
        'big_church_count_3000', \
        'church_count_3000', \
        'leisure_count_3000', \
        'sport_count_3000', \
        'green_part_5000',\
        'office_count_5000',\
        'office_sqm_5000',\
        'trc_count_5000',\
        'trc_sqm_5000',\
        'cafe_count_5000',\
        'cafe_count_5000_na_price', \
        'cafe_count_5000_price_500', \
        'cafe_count_5000_price_1000', \
        'cafe_count_5000_price_1500', \
        'cafe_count_5000_price_2500', \
        'cafe_count_5000_price_4000', \
        'cafe_count_5000_price_high', \
        'big_church_count_5000', \
        'church_count_5000', \
        'leisure_count_5000', \
        'sport_count_5000', \
        'market_count_5000', \
        'avg_price_ID_metro', \
        'avg_price_ID_railroad_station_walk', \
        'avg_price_ID_big_road1', \
        'avg_price_ID_big_road2', \
        'avg_price_ID_railroad_terminal', \
        'avg_price_ID_bus_terminal', \
        'avg_price_sub_area' \
    ]
    for f in features_to_remove:
        if f in df_train:
            del df_train[f]

def scale_up_positive(df, numeric_cols):
    for col in numeric_cols:
        min_val = min(df[col].value_counts().index)
        if min_val < 0:
            df[col] -= min_val
            df[col] += 1
        else:
            df[col] += 1

def remove_features_by_high_corr(df):            
    features_to_remove = [
        'children_preschool', 'children_school', 'male_f', \
        'female_f', 'young_male', 'young_female', 'work_male', \
        'work_female', 'ekder_male', 'ekder_female', '16_29_all',\
        '0_6_all', '0_6_male', '0_6_female',\
        '7_14_all', '7_14_male', '7_14_female', '0_17_male', '0_17_female',\
        '16_29_male', '16_29_female', '0_13_all', '0_13_male', '0_13_female',\
        'metro_km_walk', 'railroad_station_walk_km',\
        'railroad_station_avto_km', 'public_transport_station_km' \
    ]
    for f in features_to_remove:
        del df[f]
        
def find_missing_data_columns(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['missing_column', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count'] > 0]
    return missing_df

def imput_by_interpolate(df):
    missing_df = find_missing_data_columns(df)
    for col in missing_df['missing_column']:
        df[col] = df[col].interpolate(mathod='linear')
    return df

def impute_by_regression(df, repeat, corr_thresh):
    for i in range(repeat):
        pairs = []
        missing_df = find_missing_data_columns(df)
        for missing_col in missing_df['missing_column']:
            if not np.issubdtype(df[missing_col], np.number) : continue
            corrs = [ (missing_col, c, abs(df[missing_col].corr(df[c]))) for c in df._get_numeric_data().columns if c != missing_col ]
            corrs.sort(key=lambda item : item[2], reverse=True)
            for item in corrs:
                if item[2] >= corr_thresh:
                    pairs.append(item)
                else:
                    break
        df_nan_col_with_high_corr_col = pd.DataFrame(pairs, columns=['missing_col', 'highest corr with', 'corr'])
        for row in df_nan_col_with_high_corr_col.iterrows():
            if df[row[1][0]].isnull().sum() <= 0 : continue
            nan_col = row[1][0]
            high_corr_col = row[1][1]
            corr = row[1][1]
            
            df_temp = pd.DataFrame(df[[high_corr_col, nan_col]], columns=[high_corr_col, nan_col])
            df_temp = df_temp.dropna()
            
            df_temp = sm.add_constant(df_temp)
            X = df_temp.values[:, :2]
            y = df_temp.values[:, 2]
            result = sm.OLS(y, X).fit()
            
            dfX = sm.add_constant(df[high_corr_col])
            predicted = result.predict(dfX)
            
            df = pd.merge(df, predicted.to_frame('predicted'), left_index=True, right_index=True)
            df[nan_col].fillna(df['predicted'], inplace=True)
            del df['predicted']
    return df

def impute_by_regression2(df, repeat, corr_thresh):
    for i in range(repeat):
        pairs = []
        missing_df = find_missing_data_columns(df)
        for missing_col in missing_df['missing_column']:
            if not np.issubdtype(df[missing_col], np.number) : continue
            corrs = [ (missing_col, c, abs(df[missing_col].corr(df[c]))) for c in df._get_numeric_data().columns if c != missing_col ]
            corrs.sort(key=lambda item : item[2], reverse=True)
            for item in corrs:
                if item[2] >= corr_thresh:
                    pairs.append(item)
                else:
                    break
        df_nan_col_with_high_corr_col = pd.DataFrame(pairs, columns=['missing_col', 'highest corr with', 'corr'])
        for row in df_nan_col_with_high_corr_col.iterrows():
            if df[row[1][0]].isnull().sum() <= 0 : continue
            nan_col = row[1][0]
            high_corr_col = row[1][1]
            corr = row[1][1]
            
            df_temp = pd.DataFrame(df[[high_corr_col, nan_col]], columns=[high_corr_col, nan_col])
            df_temp = df_temp.dropna()
            
            df_temp = sm.add_constant(df_temp)
            X = df_temp.values[:, :2]
            y = df_temp.values[:, 1]
            result = sm.OLS(y, X).fit()
            
            dfX = sm.add_constant(df[high_corr_col])
            predicted = result.predict(dfX)
            
            df = pd.merge(df, predicted.to_frame('predicted'), left_index=True, right_index=True)
            df[nan_col].fillna(df['predicted'], inplace=True)
            del df['predicted']
    return df
            
