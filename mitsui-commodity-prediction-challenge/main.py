# Source: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/writeups/3-rd-place-solution-directional-trends-over-vola
# Note: This file relies on/contains code from the source above which may be under its own license (e.g., Apache 2.0).
import pandas as pd
import numpy as np
import warnings

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial

train_df = pd.read_csv('data/train.csv')
train_labels_df = pd.read_csv("data/train_labels.csv")
target_pairs_df = pd.read_csv('data/target_pairs.csv')

target_pairs_df.iloc[9]

def generate_log_returns(data, lag):
    log_returns = pd.Series(np.nan, index=data.index)

    # Compute log returns based on the rules
    for t in range(len(data)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                numer_idx = t + lag + 1
                denom_idx = t + 1
                numer = data.iloc[numer_idx]
                denom = data.iloc[denom_idx]
                if (pd.isna(numer) or pd.isna(denom) or denom == 0 or numer <= 0 or denom <= 0):
                    log_returns.iloc[t] = np.nan
                    continue
                ratio = numer / denom
                if ratio <= 0 or np.isinf(ratio) or np.isnan(ratio):
                    log_returns.iloc[t] = np.nan
                else:
                    log_returns.iloc[t] = np.log(ratio)
            except Exception:
                log_returns.iloc[t] = np.nan
    return log_returns

def generate_targets(column_a: pd.Series, column_b: pd.Series, lag : int) -> pd.Series :
    a_returns = generate_log_returns(data=column_a, lag=lag)
    b_returns = generate_log_returns(data=column_b, lag=lag)
    return a_returns - b_returns

def get_data_for_day(row, index):
    """
    Return:
      target_col, x_train, x_test, y_train, y_test

    - row: one row from target_pairs_df (target, lag, pair)
    - index: integer day index (use len(train_df)-1 for latest)
    """
    target_column = row.iloc[0]
    lag = int(row.iloc[1])
    feature_string = str(row.iloc[2])

    feature_list = [p.strip() for p in feature_string.split('-') if p.strip() != ""]
    feature_a = feature_list[0]
    series_a = train_df[feature_a].iloc[:index].copy()

    if (len(feature_list) > 1):
        feature_b = feature_list[1]
        series_b = train_df[feature_b].iloc[:index].copy()
    else:
        # If only one feature present, use zeros (or optionally use series_a to just return a_returns)
        series_b = pd.Series(0.0, index=train_df.index[:index])
    

    # Build feature_data using only existing columns
    available_cols = [c for c in feature_list if c in train_df.columns]
    feature_data = train_df[available_cols].iloc[:index].copy()
    feature_data = feature_data.ffill().bfill()

    # Compute raw y_series (may contain NaNs)
    y_series_raw = generate_targets(series_a, series_b, lag)

    # === Fill NaNs in the whole target series BEFORE slicing ===
    # (forward-fill then back-fill is typical for time-series)
    y_series = y_series_raw.ffill().bfill()

    if y_series.isna().all():
        y_series = pd.Series(0.0, index=y_series_raw.index)

    # Align features and targets (account for lag)
    # x_train uses rows [0 .. index-lag-1]  (length = index - lag)
    # y_train uses rows [lag .. index-1]     (same length)
    max_idx_for_train = index - lag  # exclusive upper bound for x_train iloc
    if max_idx_for_train <= 0:
        # Not enough history to produce training rows — return empty arrays
        x_train = feature_data.iloc[0:0].copy()
        y_train = pd.Series(dtype='float64')
    else:
        x_train = feature_data.iloc[:max_idx_for_train].copy()
        y_train = y_series.iloc[lag:index].copy()
    
    # Test feature row: features at time (index - lag)
    test_row_index = index - lag
    # Guard: if test_row_index is out of bounds use last available
    if test_row_index < 0:
        test_row_index = 0
    elif test_row_index >= len(feature_data):
        test_row_index = len(feature_data) - 1
    
    x_test = feature_data.iloc[[test_row_index]].copy()

    # y_test: target corresponding to prediction time (aligned to how generate_targets is defined)
    # We use the filled series to avoid NaNs
    if test_row_index < 0 or test_row_index >= len(y_series):
        # fallback to last available
        y_test = y_series.iloc[-1]
    else:
        y_test = y_series.iloc[test_row_index]

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True) if len(y_train) > 0 else y_train

    return target_column, x_train, x_test, y_train, y_test

def get_test_data_for_day(full_test_df, row):
    target_column = row.iloc[0]
    lag = int(row.iloc[1])
    feature_string = str(row.iloc[2])

    feature_list = [p.strip() for p in feature_string.split('-') if p.strip() != ""]
    if hasattr(full_test_df, "to_pandas"):
        full_test_df = full_test_df.to_pandas()
    test_data = full_test_df[feature_list].ffill().bfill()

    return target_column, test_data

day_id = 11
row = target_pairs_df.iloc[day_id]  
_, x_train, x_test, y_train, y_test = get_data_for_day(row, day_id)
print("="*50)
print(x_train)
print("="*50)
print(x_test)
print("="*50)
print(y_train)
print("="*50)
print(y_test)

def create_lags(data, lags):
    s = pd.Series(data).reset_index(drop=True)
    lags = [lags] if isinstance(lags, int) else lags
    lag_df = pd.DataFrame({f'lag_{n}' : s.shift(n) for n in lags})
    return lag_df

def create_rolling_features(data, windows, functions=['mean']):
    s = pd.Series(data).reset_index(drop=True)
    windows = [windows] if isinstance(windows, int) else windows
    functions = [functions] if isinstance(functions, str) else functions

    df = pd.DataFrame()
    for w in windows:
        rolled = s.rolling(window=w)
        for func in functions:
            if hasattr(rolled, func):
                df[f'rolled_{func}_{w}'] = getattr(rolled, func)
            else:
                raise ValueError(f"Unspported function: {func}")
    return df


def create_diff_features(data, lags):
    if isinstance(data, list ):
        data = pd.Series(data)
    if isinstance(lags, int):
        lags = [lags]
    
    diff_df = pd.DataFrame()
    for lag in lags:
        diff_df[f'diff_{lag}'] = data.diff(lag)
    return diff_df

s = pd.Series([10, 20, 30, 40, 50, 60])

print(create_lags(s, [1, 2, -1 , -2]))
print(create_rolling_features(s, windows=[2, 3], functions=['mean', 'max']))
print(create_diff_features(s, [1, 2 , -1 , -2]))

def prepare_features_for_col(
        col,
        col_name, 
        lag_values=None,
        win_values=None,
        win_methods=None,
        diff_values=None,
        is_a_target=False
):
    """
    Generate lag, rolling window, and difference features for a single column.
    
    Parameters:
    ----------
    col : list, pandas.Series, or numpy.ndarray
        The input column data.
    col_name : str
        Name of the column for naming generated features.
    lag_values : list[int]
        List of lag steps.
    win_values : list[int]
        List of window sizes for rolling features.
    win_methods : list[str]
        Methods for rolling aggregation: 'mean', 'max', 'min', 'sum', etc.
    diff_values : list[int]
        List of periods for calculating differences.
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with all generated features.
    """

    # Ensure col is a pandas Series
    if not isinstance(col, pd.Series):
        col = pd.Series(col)
    
    # Initialize result Dataframe
    features = pd.DataFrame(index=col.index)

    if not is_a_target:
        features[f"{col_name}"] = col
    
    # --- Lag Features ---
    if lag_values:
        for lag in lag_values:
            features[f'lag_{lag}_{col_name}'] = col.shift(lag)
    
    # --- Rolling Window Features ---
    if win_values and win_methods:
        for win in win_values:
            for method in win_methods:
                if hasattr(pd.Series.rolling(col, win), method):
                    features[f'win_{method}_{win}_{col_name}'] = getattr(col.rolling(win), method)()
                else:
                    raise ValueError(f"Method {method} is not supported for rolling windows.")
    
    # --- Difference Features ---
    if diff_values:
        for diff in diff_values:
            features[f'diff_{diff}_{col_name}'] = col.diff(diff)
    
    return features

df_features = prepare_features_for_col (
    col=s,
    col_name="sales",
    lag_values=[1, 2, -1, -2],
    win_values=[2, 3],
    win_methods=["mean", "max"],
    diff_values=[1, 2, -1, -2]
)

print(df_features)

def prepare_features_for_df(
        df : pd.DataFrame,
        lag_values=None,
        win_values=None,
        win_methods=None,
        diff_values=None
):
    to_return = pd.DataFrame()

    for col in list(df.columns):
        feature_df = prepare_features_for_col(
            col = df[col],
            col_name=col,
            lag_values=lag_values,
            win_values=win_values,
            win_methods=win_methods,
            diff_values=diff_values
        )
        to_return = pd.concat([to_return, feature_df], axis=1)
    
    to_return = to_return.ffill().bfill()

    return to_return

smpdf= pd.DataFrame(
    {
        'A':[3,4,5],
        'B':[4,5,6]
    }
)

smpfeatdf = prepare_features_for_df(
    smpdf,
    lag_values=[1, 2,5,7,10,15,20,-1,-2,-5,-7,-10,-20 ],
    win_values=[2, 3 , 5 , 7 , 10 , 15 , 30 ],
    win_methods=["mean", "max"],
    diff_values=[1, 2, 5 , 7 , 10 , 15 , 20 ,-1,-2, -5,-7,-10,-15,-20 ]
)
print(smpfeatdf)

# scaling
scaler = StandardScaler()
def scale(df):
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
    return scaled_df

print(scale(smpfeatdf.fillna(0)).head(2))

# logging
def log_transform_df(df : pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_log = df.copy()
    numeric_values = df_log[numeric_cols].to_numpy(dtype=float)
    numeric_values[numeric_values <= -1] = np.nan
    df_log[numeric_cols] = np.log1p(numeric_values) #log1p = log(x+1)
    return df_log

print(log_transform_df(smpfeatdf).head(2))

# ----------
# Model Engineering
#-----------

def train_and_get_result(x_train, y_train, x_test):
    base_learners = [
        ('lgbm', LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42, verbosity=-1)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=50, random_state=42))
    ]
    # Step 1: Train base model and get predictions on training data
    meta_features_train = []
    meta_features_test = []
    base_models = []
    for name, model in base_learners:
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        meta_features_train.append(train_pred.reshape(-1, 1))  # For stacking
        meta_features_test.append(test_pred.reshape(-1, 1))    # For prediction
        base_models.append(model)
    # Stack base learners' predictions as features for meta-model
    meta_x_train = np.hstack(meta_features_train)
    meta_x_test = np.hstack(meta_features_test)
    # Step 2: Train meta-model
    meta_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    meta_model.fit(meta_x_train, y_train)
    # Step 3: Predict using the meta-model
    final_pred = meta_model.predict(meta_x_test)[0]
    # Optional: Save the meta-model
    # joblib.dump(meta_model, "final_stacked_xgb_model.pkl")
    return final_pred, {"meta_model" : meta_model , "base_models" : base_models }

def predict_on_models(model_dict, test):
    """
    Use saved base models to generate predictions and feed them into the meta-model.
    """
    base_models = model_dict["base_models"]
    meta_model = model_dict["meta_model"]
    # Get base models' predictions on the test input
    meta_features_test = []
    for model in base_models:
        pred = model.predict(test)
        meta_features_test.append(pred.reshape(-1, 1))  # Ensure 2D shape
    # Stack all base predictions into one feature set
    meta_x_test = np.hstack(meta_features_test)
    # Final prediction using the meta-model
    final_pred = meta_model.predict(meta_x_test)
    return final_pred

def safe_log1p(X):
    X = np.array(X, dtype=float)
    X = np.where(X <= -1, np.nan, X)  # avoid log on invalid values
    return np.log1p(X)

day_id = 12
row = target_pairs_df.iloc[day_id]  
target_col , x_train, x_test, y_train, y_test = get_data_for_day(row, day_id)
x_train_feature = prepare_features_for_df(
    x_train,
    lag_values=[1, 2, -1, -2],
    win_values=[2, 3],
    win_methods=['mean', 'max'],
    diff_values=[1, 2, -1, -2]
)
x_test_feature = prepare_features_for_df(
    x_test,
    lag_values=[1, 2,-1,-2],
    win_values=[2, 3],
    win_methods=["mean", "max"],
    diff_values=[1, 2,-1,-2]
)
x_test_feature = x_test_feature.reindex(columns=x_train_feature.columns, fill_value=np.nan)
pipeline = Pipeline([
    ('log', FunctionTransformer(safe_log1p, validate=False)),
    ('replace_inf', FunctionTransformer(lambda X: np.where(np.isinf(X), np.nan, X), validate=False)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
x_train_feature_scaled = pipeline.fit_transform(x_train_feature)
x_test_feature_scaled = pipeline.transform(x_test_feature)

feature_names = x_train_feature.columns.tolist()
x_train_feature_scaled = pd.DataFrame(x_train_feature_scaled, columns=feature_names)
x_test_feature_scaled = pd.DataFrame(x_test_feature_scaled, columns=feature_names)

value , model_dict = train_and_get_result(x_train_feature_scaled, y_train, x_test_feature_scaled)
print(value)

TARGET_MODEL_POOL = {}
TARGET_MODEL_PIPELINE_POOL = {}
TARGET_FEATURE_COLUMNS = {}

def replace_inf_func(X):
    return np.where(np.isinf(X), np.nan, X)

def train_one_target(row, day_id):
    target_col, x_train, x_test, y_train, y_test = get_data_for_day(row, day_id)

    # Prepare features
    x_train_feat = prepare_features_for_df(
        x_train,
        lag_values=[1, 2, -1, -2],
        win_values=[2, 3],
        win_methods=["mean", "max"],
        diff_values=[1, 2, -1, -2]
    )
    x_test_feat = prepare_features_for_df(
        x_test,
        lag_values=[1, 2, -1, -2],
        win_values=[2, 3],
        win_methods=["mean", "max"],
        diff_values=[1, 2, -1, -2]
    )
    feature_cols = x_train_feat.columns.tolist()
    x_test_feat = x_test_feat.reindex(columns=feature_cols, fill_value=np.nan)

    # Build pipeline without lambdas
    pipeline = Pipeline([
        ('log', FunctionTransformer(safe_log1p, validate=False)),
        ('replace_inf', FunctionTransformer(replace_inf_func, validate=False)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Scale data
    x_train_feat_sc = pipeline.fit_transform(x_train_feat)
    x_test_feat_sc = pipeline.transform(x_test_feat)
    x_train_feat_sc = pd.DataFrame(x_train_feat_sc, columns=feature_cols)
    x_test_feat_sc = pd.DataFrame(x_test_feat_sc, columns=feature_cols)

    # Train model
    value, model_dict = train_and_get_result(x_train_feat_sc, y_train, x_test_feat_sc)

    return target_col, pipeline, model_dict, feature_cols

# Determine day_id
if len(train_df) >= 2:
    day_id = train_df.index[-2]
else:
    raise ValueError("Not enough data for training")

rows = [target_pairs_df.iloc[d] for d in range(len(target_pairs_df))]
# Use partial to avoid lambda
with ThreadPoolExecutor(max_workers=len(train_df)) as executor:
    for target_col, pipeline, model_dict, feature_cols in executor.map(partial(train_one_target, day_id=day_id), rows):
        TARGET_MODEL_PIPELINE_POOL[target_col] = pipeline
        TARGET_MODEL_POOL[target_col] = model_dict
        TARGET_FEATURE_COLUMNS[target_col] = feature_cols

def safe_fillna(df):
    # Replace NaN medians with 0 as fallback
    medians = df.median()
    medians = medians.fillna(0)
    return df.fillna(medians)

def _predict_one_target(row, full_test_df):
    target_col, test_df = get_test_data_for_day(full_test_df, row)

    # Fill before feature creation
    test_df = safe_fillna(test_df)

    # Create features
    test_df_feat = prepare_features_for_df(
        test_df,
        lag_values=[1, 2, -1, -2],
        win_values=[2, 3],
        win_methods=["mean", "max"],
        diff_values=[1, 2, -1, -2]
    )

    feature_cols = TARGET_FEATURE_COLUMNS.get(target_col)
    if feature_cols is not None:
        test_df_feat = test_df_feat.reindex(columns=feature_cols, fill_value=np.nan)
    else:
        feature_cols = test_df_feat.columns.tolist()

    pipeline = TARGET_MODEL_PIPELINE_POOL[target_col]
    test_df_feat_sc = pipeline.transform(test_df_feat)
    test_df_feat_sc = pd.DataFrame(test_df_feat_sc, columns=feature_cols)

    # Predict
    md = TARGET_MODEL_POOL[target_col]
    predictions = predict_on_models(md, test_df_feat_sc)

    return target_col, predictions

predict_invoke_count = 0

def predict_on_test(full_test_df):
    global predict_invoke_count
    print(f"predict invoked {predict_invoke_count}")
    rows = [target_pairs_df.iloc[t] for t in range(len(target_pairs_df))]
    preds_dict = {}

    with ThreadPoolExecutor() as executor:
        for target_col, predictions in executor.map(lambda row: _predict_one_target(row, full_test_df), rows):
            preds_dict[target_col] = predictions

    preds_df = pd.DataFrame(preds_dict)
    # preds_df.to_parquet("submission.parquet", index=False)
    predict_invoke_count += 1
    return preds_df

