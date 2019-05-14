init_col_map = {
    'col1':'newCol1',
}



def mapColumnNames(df):
    
    df = df.select([F.col(c).alias(init_col_map.get(c, c)) for c in df.columns])
    df = df[init_col_map.values()]
    print('Column names mapped!')
    return df