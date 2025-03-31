import pandas as pd

__all__ = ["add_missing_values", "remove_empty_entries", "group_results", "summarize_results"]


# TODO see if this is still necessary and adapt
def add_missing_values(df, column):
    """Add missing values to dataframe when data is logged on event (mostly evaluations)"""
    # gather all logged on event numbers
    loe_numbers = sorted(df[column].unique())
    loe_numbers = pd.DataFrame(loe_numbers, columns=[column])

    # split dataframe into several, one for each run
    splits = []
    start = 0
    for i in df.index:
        if df.at[i, column] == df[column].max():
            splits.append(df.iloc[start:i+1])
            start = i+1

    # expand evaluation numbers for each run and fill missing data with those of previous evaluation
    dfs = [j.merge(loe_numbers, on=column, how='right') for j in splits]
    df_filled = [k.fillna(method='ffill') for k in dfs]

    # merge now completed dataframes once more
    return pd.concat(df_filled)


def remove_empty_entries(logs, column_name):
    """Remove rows where entries in specified columns are NaN
    (this may occur if two separate triggers were used for logging)"""
    for key, df in logs.items():
        df = df[df[column_name].notnull().values]
        logs[key] = df
    return logs


def group_results(logs, function, dimension):
    """Group the logs of one function and/or one dimensionality into a new dict;
    use '' if grouping by only one factor"""
    keys = [key for key in logs if function in key.lower() and dimension in key.lower()]
    grouped_log = {x: logs[x] for x in logs if x in keys}
    return grouped_log


def summarize_results(logs):
    """Summarize the final values of specific results"""
    columns = list(logs.values())[0].columns.tolist()
    columns.insert(0, 'log')
    df = pd.DataFrame(columns=columns)
    for key, value in logs.items():
        entries = (value.iloc[-1]).tolist()
        entries.insert(0, key)
        df.loc[len(df)+1] = entries
    return df
