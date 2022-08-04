import matplotlib.pyplot as plt
#funcion que recibe un dataframe y el nombre de una columna y saca la kurtosis y el sego, asi como crea un box plot y un histograma
def graph_of_normalization(df,columna):
    #pruebas estadisticas para comprobar si es o no normal
    kurtosis = df[[columna]].kurtosis()
    skew = df[[columna]].skew()
    
    fig = plt.figure(figsize=(10,10),constrained_layout=True)
    gs = fig.add_gridspec(nrows=2,ncols=2)
    fig.suptitle('Pruebas de Normalidad: '+columna + '\n kurtosis: '+ str(kurtosis[0]) + '\n Sesgo: ' + str(skew[0]), fontsize=16, color='w')
    fig.set_facecolor('#5e6e65')

    ax1 = fig.add_subplot(gs[0,0])
    df[[columna]].boxplot(ax=ax1)
    #datos.budget.plot.kde(ax=ax1,secondary_y=True)
    ax1.set_title('Boxplot',color='w')
    
    ax2 = fig.add_subplot(gs[0,1])
    df[[columna]].hist(ax=ax2)
    #datos.revenue.plot.kde(ax=ax2,secondary_y=True)
    ax2.set_title('hist',color='w')
    
    plt.show()
    
######################
#remove outliers
#function that get outliers from multiple columns
#return a list of index of outliers
def get_outlier(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    low_bound = Q1 - 1.5 * IQR
    up_bound = Q3 + 1.5 * IQR
    
    outlier_list = df.index[(df[col] < low_bound) | (df[col] > up_bound)]
    
    return outlier_list
#function that remove the outliers based on the list
#return a cleaned data frame

def remove_outliers(df,outliers_list):
    outliers_list = sorted(set(outliers_list))
    df = df.drop(outliers_list)
    
    return df


def clean_dataset(df,cols):
    index_list = []
    for col in df.columns[cols]:
        index_list.extend(get_outlier(df,col))
    #return a new clean dataset
    return remove_outliers(df,index_list)
