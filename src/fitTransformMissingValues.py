   

def fitMissing(df=None, y_var=None, num_fillBy='mean', cat_fillBy='mode'):
    assert df.select(F.count(F.when(F.isnull(y_var), y_var))).collect()[0][0] == 0
    
    num_vars = [c[0] for c in df.dtypes if (c[1] == 'int') | (c[1] == 'double')]
    cat_vars = [c[0] for c in df.dtypes if c[0] not in num_vars]
    num_vars_ = [var for var in num_vars if var!=y_var]
    cat_vars_ = [var for var in cat_vars if var!=y_var]
    
    def getNumMeans(df, num_vars):
        num_means = [df.select(F.mean(F.col(column))).collect()[0][0] for column in num_vars]
        num_means = dict(zip(num_vars, num_means))
        return num_means

    def getNumMedians(df, num_vars):
        num_medians = [df.select(F.median(F.col(column))).collect()[0][0] for column in num_vars]
        num_medians = dict(zip(num_vars, num_medians))
        return num_medians
    
    def getNumCustoms(df, num_vars, num_fillBy):
        num_custom_fill = {}

        if isinstance(num_fillBy, (int, float)):
            for column in num_vars:
                num_custom_fill[column] = num_fillBy
        elif type(num_fillBy) == dict:
            assert set(list(num_fillBy.keys())) == set(num_vars)
            for column in num_vars:
                if num_fillBy[column] == 'mean':
                    num_custom_fill[column] = df.select(F.mean(F.col(column))).collect()[0][0]
                elif num_fillBy[column] == 'median':
                    num_custom_fill[column] = df.select(F.median(F.col(column))).collect()[0][0]
                else:
                    num_custom_fill[column] = num_fillBy[column]
        else:
            raise TypeError('Invalid type "%s" of parameter "num_fillBy"' %(type(num_fillBy)))
        return num_custom_fill
    
    num_means = {}
    num_medians = {}
    cat_modes = {}
    
    for c in num_vars_:
        if num_fillBy == 'mean':
            num_means = getNumMeans(df, c)
            
        elif num_fillBy == 'median':
            num_medians = getNumMedians(df, c) 
            
        else:
            num_custom_fill = getNumCustoms(df, num_vars_, num_fillBy)
            
            
    def getCatModes(df, cat_vars):
        cat_modes = [df.select(F.mode(F.col(column))).collect()[0][0] for column in cat_vars]
        cat_modes = dict(zip(cat_vars, cat_modes))
        return cat_modes

    def getCatCustoms(df, cat_vars, cat_fillBy):
        cat_custom_fill = {}

        if isinstance(cat_fillBy, (str, type(datetime.strptime('1/1/1000', '%m/%d/%Y')))):
            for column in cat_vars:
                cat_custom_fill[column] = cat_fillBy
        elif type(cat_fillBy) == type({}):
            logging.warning(len(set(list(cat_fillBy.keys()))))
            logging.warning(len(set(cat_vars)))
            logging.warning(set(list(cat_fillBy.keys())))
            logging.warning(set(cat_vars))
            assert set(list(cat_fillBy.keys())) == set(cat_vars) # make sure operation is assigned to each categorical column
            for column in cat_vars:
                if cat_fillBy[column] == 'mode':
                    cat_custom_fill[column] = df.select(F.mode(F.col(column))).collect[0][0]
                else:
                    cat_custom_fill[column] = cat_fillBy[column]
        else:
            raise TypeError('Invalid type "%s" of parameter "cat_fillBy"' %(type(num_fillBy)))
        return cat_custom_fill
        
        
    if cat_fillBy == 'mode':
        cat_modes = getCatModes(df, cat_vars_)
    else:
        cat_custom_fill = getCatCustoms(df, cat_vars_, cat_fillBy)
    print('Fitting on missing data completed!')
    return (num_vars_, cat_vars_, num_means, num_medians, num_custom_fill, cat_modes, cat_custom_fill)

def transformMissing(df):
    
    if num_fillBy == 'mean':
        num_fillMap = num_means
    elif num_fillBy == 'median':
        num_fillMap = num_medians
    else:
        num_fillMap = num_custom_fill
        
    if cat_fillBy == 'mode':
        cat_fillMap = cat_modes
    else:
        cat_fillMap = cat_custom_fill

    for column in num_vars_:
        df = df.fillna({column: num_fillMap[column]})
    for column in cat_vars_:
        df = df.fillna({column: cat_fillMap[column]})
    print('Imputing missing data completed!')
    return df