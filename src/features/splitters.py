# src/data/splitters.py

import pandas as pd

def drop_leakage_columns(df: pd.DataFrame, raw_daily_col: str = "total_kwh") -> pd.DataFrame:
    """
    Drops all leakage-prone columns:
      1) The raw daily consumption column (raw_daily_col).
      2) Any half-hourly columns named "hh_0" ... "hh_47".
      3) Any intermediate consumption-pattern columns whose names match
         a predefined blacklist of prefixes (e.g. "mean_kwh", "peak_kwh", etc.).
    Returns a new DataFrame with those columns removed (if they existed).
    """
    df = df.copy()
    leakage_cols = [raw_daily_col]

    # 1) Half-hourly columns "hh_0" ... "hh_47"
    hh_cols = [
        col for col in df.columns
        if col.startswith("hh_") and col.replace("hh_", "").isdigit()
    ]
    leakage_cols.extend(hh_cols)

    # 2) Intermediate consumption-pattern columns
    forbidden_prefixes = [
        "mean_kwh", "std_kwh", "peak_kwh", "min_kwh",
        "morning_kwh", "afternoon_kwh", "evening_kwh", "night_kwh",
        "peak_period_kwh", "off_peak_kwh", "base_load", "load_factor",
        "daily_variability", "coefficient_of_variation",
        "usage_concentration", "peak_sharpness",
        "peak_to_mean_ratio", "peak_to_total_ratio", "day_night_ratio",
        "holiday_consumption_boost", "base_load_ratio", "consumption_sharpness"
    ]
    for col in df.columns:
        for prefix in forbidden_prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                leakage_cols.append(col)

    leakage_cols = list(set(leakage_cols))
    return df.drop(columns=leakage_cols, errors="ignore")


def _build_feature_groups(feature_cols: list, static_cols: list = None, group_cols: list = None) -> dict:
    """
    Given a list of feature column names, plus lists of any "static" (household-level)
    and "group" (cluster-level) column names, return a dictionary mapping
    feature-group names to lists of columns belonging to each group.

    Adjust the patterns below to match your actual column naming conventions.
    """
    feature_groups = {}

    # 1) Temporal features: common columns added by create_all_temporal_features
    temporal_patterns = ["dayofweek", "is_weekend", "is_holiday", "month", "day_of_year", "quarter"]
    temporal_cols = [
        c for c in feature_cols
        if any(pat == c or c.startswith(pat + "_") for pat in temporal_patterns)
    ]
    feature_groups["temporal"] = temporal_cols

    # 2) Weather features: common names from create_all_weather_features
    weather_patterns = ["temp", "temperature", "heating_degree", "cooling_degree", "precip", "rain", "snow"]
    weather_cols = [c for c in feature_cols if any(pat in c for pat in weather_patterns)]
    feature_groups["weather"] = weather_cols

    # 3) Lag features: columns starting with "lag"
    lag_cols = [c for c in feature_cols if c.startswith("lag")]
    feature_groups["lags"] = lag_cols

    # 4) Rolling window features: columns starting with "roll"
    roll_cols = [c for c in feature_cols if c.startswith("roll")]
    feature_groups["rolling"] = roll_cols

    # 5) Interaction features
    interaction_patterns = ["lag1_weekend", "lag1_holiday", "lag1_summer",
                            "lag7_weekend", "lag7_holiday", "lag7_summer"]
    interaction_cols = [c for c in feature_cols if any(c.startswith(pat) for pat in interaction_patterns)]
    feature_groups["interactions"] = interaction_cols

    # 6) Static household metadata (e.g., "Acorn_grouped", "stdorToU", etc.)
    if static_cols:
        static_household_cols = [c for c in feature_cols if c in static_cols]
    else:
        static_household_cols = []
    feature_groups["static_household"] = static_household_cols

    # 7) Static group/cluster metadata, if provided
    if group_cols:
        static_group_cols = [c for c in feature_cols if c in group_cols]
    else:
        static_group_cols = []
    feature_groups["static_group"] = static_group_cols

    # 8) Any remaining features that didn't match above patterns
    assigned = set(sum(feature_groups.values(), []))
    other_cols = [c for c in feature_cols if c not in assigned]
    feature_groups["other"] = other_cols

    return feature_groups


def prepare_forecasting_data(
    df_features: pd.DataFrame,
    target_col: str = "total_kwh",
    val_days: int = 30,
    test_days: int = 90
) -> tuple:
    """
    Prepare day-ahead forecasting data (h=1) in a leakage-safe way.
    Assumes that static household columns (e.g. "Acorn_grouped", "stdorToU")
    have already been merged into df_features.

    Steps:
      1) Sort by (LCLid, day)
      2) Create label_1 by shifting raw daily consumption (target_col) by -1
      3) Drop all leakage-prone columns (raw daily, half-hourly, consumption-pattern)
      4) Split into train/val/test by date cutoffs
      5) Build feature_cols and feature_groups

    Returns:
        train_df, val_df, test_df, feature_cols, target, feature_groups
    """

    # 1) Copy & ensure datetime
    df = df_features.copy()
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values(["LCLid", "day"])

    # 2) Create 1-day-ahead label
    df["label_1"] = df.groupby("LCLid")[target_col].shift(-1)
    df = df[df["label_1"].notna()].copy()

    # 3) Drop all leakage-prone columns at once
    df = drop_leakage_columns(df, raw_daily_col=target_col)

    # 4) Split into train/val/test by date
    max_date    = df["day"].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff  = test_cutoff - pd.Timedelta(days=val_days)

    train_df = df[df["day"] <= val_cutoff].copy()
    val_df   = df[(df["day"] > val_cutoff) & (df["day"] <= test_cutoff)].copy()
    test_df  = df[df["day"] > test_cutoff].copy()

    # 5) Identify feature columns vs. target
    non_feature_cols = ["LCLid", "day", "label_1"]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]
    target = "label_1"

    # Collect static household columns (those already present in df_features)
    # In your case, that might be: ["Acorn", "Acorn_grouped", "stdorToU"]
    # Adjust if you have renamed them (e.g., "tariff_type")
    static_cols = [c for c in train_df.columns if c in ["Acorn", "Acorn_grouped", "stdorToU"]]

    # If you have any group/cluster columns (e.g., "cluster_label"), list them here:
    group_cols = [c for c in train_df.columns if c.startswith("cluster_")]

    # 6) Build feature_groups
    feature_groups = _build_feature_groups(
        feature_cols=feature_cols,
        static_cols=static_cols,
        group_cols=group_cols
    )

    # 7) Print summary
    print("✅ Day‐ahead data prepared in a leakage‐safe manner")
    print(f"   ✅ Train rows: {len(train_df):,}  (Households: {train_df['LCLid'].nunique()})")
    print(f"   ✅ Val rows:   {len(val_df):,}  (Households: {val_df['LCLid'].nunique()})")
    print(f"   ✅ Test rows:  {len(test_df):,}  (Households: {test_df['LCLid'].nunique()})")
    print(f"   ✅ Train period: {train_df['day'].min().date()} to {train_df['day'].max().date()}")
    print(f"   ✅ Val period:   {val_df['day'].min().date()} to {val_df['day'].max().date()}")
    print(f"   ✅ Test period:  {test_df['day'].min().date()} to {test_df['day'].max().date()}")
    print(f"   ✅ Initial features (before static): {len(feature_cols) - len(static_cols) - len(group_cols)}")
    print(f"   ✅ Final features (including static): {len(feature_cols)}")
    print(f"   ✅ Feature groups: {len(feature_groups)} groups")
    print(f"   ✅ Target:       '{target}'")

    return train_df, val_df, test_df, feature_cols, target, feature_groups


def prepare_weekahead_data(
    raw_df: pd.DataFrame,
    df_features: pd.DataFrame,
    test_days: int = 90,
    val_days: int = 30
) -> tuple:
    """
    Prepare week-ahead forecasting data (h=7) in a leakage-safe way.
    Assumes that static household columns (e.g. "Acorn_grouped", "stdorToU")
    have already been merged into df_features.

    Steps:
      1) Extract raw daily "total_kwh" from raw_df and create label_7 by shifting -7
      2) Merge label_7 back onto df_features
      3) Drop all leakage-prone columns (raw daily, half-hourly, consumption-pattern)
      4) Split into train/val/test by date cutoffs
      5) Build feature_cols7 and feature_groups7

    Returns:
        train_df7, val_df7, test_df7, feature_cols7, target7, feature_groups7
    """

    # 1) Prepare raw daily consumption and create label_7
    df_raw = raw_df[["LCLid", "day", "total_kwh"]].copy()
    df_raw["day"] = pd.to_datetime(df_raw["day"])
    df_raw = df_raw.sort_values(["LCLid", "day"])
    df_raw["label_7"] = df_raw.groupby("LCLid")["total_kwh"].shift(-7)
    df_raw = df_raw[df_raw["label_7"].notna()].copy()

    # 2) Merge label_7 back onto all features
    temp = df_features.copy()
    temp["day"] = pd.to_datetime(temp["day"])
    df_merged = temp.merge(
        df_raw[["LCLid", "day", "label_7"]],
        on=["LCLid", "day"],
        how="left"
    )
    df_merged = df_merged[df_merged["label_7"].notna()].copy()

    # 3) Drop all leakage-prone columns at once
    df_merged = drop_leakage_columns(df_merged, raw_daily_col="total_kwh")

    # 4) Split into train/val/test by date
    max_date    = df_merged["day"].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff  = test_cutoff - pd.Timedelta(days=val_days)

    train_df7 = df_merged[df_merged["day"] <= val_cutoff].copy()
    val_df7   = df_merged[(df_merged["day"] > val_cutoff) & (df_merged["day"] <= test_cutoff)].copy()
    test_df7  = df_merged[df_merged["day"] > test_cutoff].copy()

    # 5) Identify feature columns vs. target
    non_feature_cols = ["LCLid", "day", "label_7"]
    feature_cols7 = [c for c in train_df7.columns if c not in non_feature_cols]
    target7 = "label_7"

    # Collect static household columns (already present in df_features)
    static_cols = [c for c in train_df7.columns if c in ["Acorn", "Acorn_grouped", "stdorToU"]]

    # Collect any group/cluster columns (if present)
    group_cols = [c for c in train_df7.columns if c.startswith("cluster_")]

    # 6) Build feature_groups7
    feature_groups7 = _build_feature_groups(
        feature_cols=feature_cols7,
        static_cols=static_cols,
        group_cols=group_cols
    )

    # 7) Print summary
    print("✅ Week‐ahead data prepared in a leakage‐safe manner")
    print(f"   ✅ Train rows: {len(train_df7):,}  (Households: {train_df7['LCLid'].nunique()})")
    print(f"   ✅ Val rows:   {len(val_df7):,}  (Households: {val_df7['LCLid'].nunique()})")
    print(f"   ✅ Test rows:  {len(test_df7):,}  (Households: {test_df7['LCLid'].nunique()})")
    print(f"   ✅ Train period: {train_df7['day'].min().date()} to {train_df7['day'].max().date()}")
    print(f"   ✅ Val period:   {val_df7['day'].min().date()} to {val_df7['day'].max().date()}")
    print(f"   ✅ Test period:  {test_df7['day'].min().date()} to {test_df7['day'].max().date()}")
    initial_features = len(feature_cols7) - len(static_cols) - len(group_cols)
    print(f"   ✅ Initial features (before static): {initial_features}")
    print(f"   ✅ Final features (including static): {len(feature_cols7)}")
    print(f"   ✅ Feature groups: {len(feature_groups7)} groups")
    print(f"   ✅ Target:       '{target7}'")

    return train_df7, val_df7, test_df7, feature_cols7, target7, feature_groups7
