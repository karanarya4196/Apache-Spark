
def computeDescriptionLength(df=None):

    df = df.withColumn('descriptionWordCount', F.size(F.split(F.col('description'), ' ')))
    print('Description length computed!')
    return df


def createAppearanceOfAnd(df=None):
    df = df.withColumn('andFeature', F.when(F.lower(df.description).contains(' and '), 1).otherwise(0))
    print("Appearance of 'and' calculated!")
    return df

def createDateFeatures(df=None):

    df = df.withColumn('invoice_year', F.year('invoice_date'))
    df = df.withColumn('invoice_month', F.month('invoice_date'))
    df = df.withColumn('line_item_duration', F.datediff(df.invoice_date, df.line_item_date_of_service))
    print('Date features generated!')
    return df

def createFeeType(df=None):

    df = df.withColumn('category_var', F.when(F.lower(df.task_code).contains('expense'), 'ExpenseGrp').when(F.lower(df.task_code).contains('fee'), 'FeeGrp').otherwise('OrdinaryGrp'))
    print('Fee types created!')
    return df


def fitUnitRateBag(df, col, bins):

    quantileD = (QuantileDiscretizer()
              .setInputCol(col)
              .setOutputCol(col + '_bag')
              .setNumBuckets(bins)
              .setHandleInvalid('keep') # 'error','keep', 'skip'
              .setRelativeError(0.01) # 0: exact quantiles calculated: expensive operation.
             )
    qBucketizer = quantileD.fit(df)
    temp = qBucketizer.transform(df)
    
    return temp


def fitTransformUnitRateBag(df):
    
    bin_dict = {'rate': 8, 'unit': 5}
    print('Fit train data to create rate and unit bins!')
    for col, bins in bin_dict.items():
        if col == 'rate':
            rate_bag_df = fitUnitRateBag(df, col, bins)
            df = df.join(rate_bag_df[['line_item_id', 'rate_bag']], on = 'line_item_id', how = 'left')
        if col == 'unit':
            unit_bag_df = fitUnitRateBag(df, col, bins)
            df = df.join(unit_bag_df[['line_item_id', 'unit_bag']], on = 'line_item_id', how = 'left')
    print('Rate and Unit bins created!')
    return df
