import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from pmdarima.arima import auto_arima
import pandas as pd
from prophet import Prophet

#funcion que recibe un dataframe y el nombre de una columna y saca la kurtosis y el sego, asi como crea un box plot y un histograma
def graph_of_normalization(df, columna):
    #pruebas estadisticas para comprobar si es o no normal
    kurtosis = df[[columna]].kurtosis()
    skew = df[[columna]].skew()

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2)
    fig.suptitle('Pruebas de Normalidad: ' + columna + '\n kurtosis: ' +
                 str(kurtosis[0]) + '\n Sesgo: ' + str(skew[0]),
                 fontsize=16,
                 color='w')
    fig.set_facecolor('#5e6e65')

    ax1 = fig.add_subplot(gs[0, 0])
    df[[columna]].boxplot(ax=ax1)
    #datos.budget.plot.kde(ax=ax1,secondary_y=True)
    ax1.set_title('Boxplot', color='w')

    ax2 = fig.add_subplot(gs[0, 1])
    df[[columna]].hist(ax=ax2)
    #datos.revenue.plot.kde(ax=ax2,secondary_y=True)
    ax2.set_title('hist', color='w')

    plt.show()


######################
#remove outliers
#function that get outliers from multiple columns
#return a list of index of outliers
def get_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    low_bound = Q1 - 1.5 * IQR
    up_bound = Q3 + 1.5 * IQR

    outlier_list = df.index[(df[col] < low_bound) | (df[col] > up_bound)]

    return outlier_list


#function that remove the outliers based on the list
#return a cleaned data frame


def remove_outliers(df, outliers_list):
    outliers_list = sorted(set(outliers_list))
    df = df.drop(outliers_list)

    return df


def clean_dataset(df, cols):
    index_list = []
    for col in df.columns[cols]:
        index_list.extend(get_outlier(df, col))
    #return a new clean dataset
    return remove_outliers(df, index_list)


#Dickey_Fuller_test for arima model stationary
def Dickey_Fuller_test(data):
    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def split_train_test(data, train_porcentage=0.8):
    #get the test size from the data
    TEST_SIZE = int(len(data) * train_porcentage)
    train, test = data.y.iloc[:TEST_SIZE], data.y.iloc[TEST_SIZE:]
    n_periods_val = len(test)
    return train, test, n_periods_val


def fix_df(df):
    new_df = df
    df['Date'] = new_df['ds']
    df = df.set_index(['Date'])
    return df


def arima_model(data, train, test, n_periods_val):
    #Create model
    model = auto_arima(data.y,
                           start_p=1,
                           start_q=1,
                           test='adf',
                           max_p=5,
                           max_q=5,
                           m=12,
                           d=1,
                           seasonal=True,
                           start_P=0,
                           D=None,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    print(model.summary())
    print(model.aic())
    # fit the model with the train data
    model.fit(train)
    #create a predicition with the test values
    print('Number of periods: ',n_periods_val)
    prediction = model.predict(n_periods=n_periods_val)
    #create a df withe the values of the predciton
    prediction = pd.DataFrame(prediction,columns=['y'],index=test.index)
    pd.concat([data.y,prediction],axis=1).plot()
    
    return model,prediction

def Prophet_model(data,interval_width = 0.95):
    model = Prophet(interval_width = interval_width)
    model.fit(data)
    future_date = model.make_future_dataframe(periods=36, freq='MS')
    future_date.tail()
    
    prediction = model.predict(future_date)
    prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    model.plot(prediction,uncertainty=True)
    model.plot_components(prediction)
    
    return model,prediction,future_date