

def splitTrainTest(df, split_ratio = 0.8, split_by = 'invoice_date'):
    
    train_df = df.sort(split_by).limit(0.8 * df.count())
    test_df = df.sort(split_by, ascending = False).limit(0.2 * df.count())
    print('Data splitted!')
    return train_df, test_df

