import ast
import pandas as pd
import numpy as np


def drop_highly_correlated_features(df, threshold, target='price' ):
    # Select numeric features only
    numeric_df = df.select_dtypes(include=[np.number])
    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Store columns to drop
    to_drop = set()
    # Loop over pairs
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                # If both not already marked for removal
                if col not in to_drop and row not in to_drop:
                    # Count missing values
                    na_col = df[col].isna().sum()
                    na_row = df[row].isna().sum()
                    # Correlation with target
                    corr_col = abs(df[[col, target]].corr().iloc[0, 1]) if col in df.columns else 0
                    corr_row = abs(df[[row, target]].corr().iloc[0, 1]) if row in df.columns else 0
                    # Decide which to drop
                    if corr_col < corr_row:
                        to_drop.add(col)
                    elif corr_row < corr_col:
                        to_drop.add(row)
                    if na_col > na_row:
                        to_drop.add(col)
                    else:
                        to_drop.add(row)
        # Drop selected columns
    df = df.drop(columns=to_drop)
    return df

def pre_processing(df):
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['host_response_rate'] = df['host_response_rate'].replace('%', '', regex=True).astype(float)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].replace('%', '', regex=True).astype(float)
    df = df.replace({'t': True, 'f': False})
    for col in df.columns:
        if df[col].dropna().isin([True, False]).all():
            df[col] = df[col].astype(bool)
    # handle missing values
    df['bathroom_missing'] = df['bathrooms'].isnull().astype(int)
    df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
    df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
    df['bedrooms'] = df['bedrooms'].replace(0, 0.1, regex=True).astype(float)
    df['reviews_per_month'].fillna(0, inplace=True)
    # feature engineering
    # add a column for price per bedroom
    df['price_per_bedroom'] = df['price'] / (df['bedrooms'])

    #categorize types in columns
    df['property_type'] = df['property_type'].astype('category').cat.codes
    df['room_type'] = df['room_type'].astype('category').cat.codes
    # proper lists of strings to facilitate counting sizes
    df['amenities'] = df['amenities'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    df['num_amenities'] = df['amenities'].apply(len)
    df = df.drop(columns=['amenities'])
    # remove listings where prices are too high
    df = df[df['price'] < 1000]
    # remove listings where rooms are not bookable
    df['has_availability'] = df['has_availability'].fillna(False)
    df = df[df['has_availability'] == True]
    # remove unnecessary features
    df = df.drop(columns=['license',
                      'calendar_updated',
                      'bathrooms_text',
                      'neighbourhood_group_cleansed',
                      'host_location',
                      'neighbourhood_cleansed',
                      'neighbourhood',
                      'host_verifications',
                      'host_neighbourhood',
                      'host_picture_url',
                      'host_thumbnail_url',
                      'host_response_time',
                      'host_about',
                      'host_url',
                      'picture_url',
                      'neighborhood_overview',
                      'description','source', 'listing_url',
                      "id", "name", "host_name", "scrape_id", 'host_id'])
    # convert dates

    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    today = pd.Timestamp.today()

    # Calculate days since the date
    df['last_review'] = (today - df['last_review']).dt.days
    #df = df.drop(columns=['last_review'])
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['first_review'] = (today - df['first_review']).dt.days
   # df = df.drop(columns=['first_review'])
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_since'] = (today - df['host_since']).dt.days
   # df = df.drop(columns=['host_since'])
    df['calendar_last_scraped'] = pd.to_datetime(df['calendar_last_scraped'], errors='coerce')
    df['calendar_last_scraped'] = (today - df['calendar_last_scraped']).dt.days
   # df = df.drop(columns=['calendar_last_scraped'])
    df['last_scraped'] = pd.to_datetime(df['last_scraped'], errors='coerce')
    df['last_scraped'] = (today - df['last_scraped']).dt.days
   # df = df.drop(columns=['last_scraped'])

    #drop features with standard deviation = 0
    df = df.drop(columns=df.std(numeric_only=True)[df.std(numeric_only=True) == 0].index)
   # drop features that are highly correlated
    df = drop_highly_correlated_features(df,0.96, target='price')
    # drop features with a risk of data leakage
    df = df.drop(columns=['estimated_revenue_l365d'])

    # drop the features with more than 50% of missing values
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)
    # drop listings with missing values
    df = df.dropna()

    return df

