
def computeSparsity(df):
    
    df_nullStats = df.select([F.count(F.when(F.isnull(c), c))
                              .alias(c) for c in df.columns]).toPandas().T
    df_nullStats.rename(columns = {0:'Null Counts'}, inplace = True)
    sparsity = float(df_nullStats['Null Counts'].sum())/(df.count() * len(df.columns))
    print 'Sparsity = ', sparsity
    if sparsity != 0:
        print(df_nullStats)
   