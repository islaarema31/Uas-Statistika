from scipy import stats
import pandas as pd
from scipy.stats import kstest, norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Walmart.csv',usecols=['Store','Date','Weekly_Sales','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment'])

class Nomor1() :
    print('1.b')
    store_id = 4
    df_filtered = df[df['Store'] == store_id]
    
    weekly_sale_stats = df_filtered['Weekly_Sales'].describe()
    holiday_flag_stats = df_filtered['Holiday_Flag'].describe()
    temperature_stats = df_filtered['Temperature'].describe()
    fuel_price_stats = df_filtered['Fuel_Price'].describe()
    customer_price_index_stats = df_filtered['CPI'].describe()
    unemployment_stats = df_filtered['Unemployment'].describe()
    
    print("Statistics for Store ID = 4:")
    print("Weekly Sale:")
    print(weekly_sale_stats)

    print("Holiday Flag:")
    print(holiday_flag_stats)

    print("Temperature:")
    print(temperature_stats)

    print("Fuel Price:")
    print(fuel_price_stats)

    print("Customer Price Index:")
    print(customer_price_index_stats)

    print("Unemployment:")
    print(unemployment_stats)
    
    print('1.c')
    fuel_price_q1 = df_filtered['Fuel_Price'].quantile(0.25)
    fuel_price_q2 = df_filtered['Fuel_Price'].quantile(0.50)
    fuel_price_q3 = df_filtered['Fuel_Price'].quantile(0.75)
    fuel_price_iqr = fuel_price_q3 - fuel_price_q1

    cpi_q1 = df_filtered['CPI'].quantile(0.25)
    cpi_q2 = df_filtered['CPI'].quantile(0.50)
    cpi_q3 = df_filtered['CPI'].quantile(0.75)
    cpi_iqr = cpi_q3 - cpi_q1

    unemployment_q1 = df_filtered['Unemployment'].quantile(0.25)
    unemployment_q2 = df_filtered['Unemployment'].quantile(0.50)
    unemployment_q3 = df_filtered['Unemployment'].quantile(0.75)
    unemployment_iqr = unemployment_q3 - unemployment_q1
    
    print("Statistics for Store ID = 4:")
    print("Fuel Price:")
    print("Q1:", fuel_price_q1)
    print("Q2:", fuel_price_q2)
    print("Q3:", fuel_price_q3)
    print("IQR:", fuel_price_iqr)

    print("Customer Price Index:")
    print("Q1:", cpi_q1)
    print("Q2:", cpi_q2)
    print("Q3:", cpi_q3)
    print("IQR:", cpi_iqr)

    print("Unemployment:")
    print("Q1:", unemployment_q1)
    print("Q2:", unemployment_q2)
    print("Q3:", unemployment_q3)
    print("IQR:", unemployment_iqr)

    print('1d')
    grouped_data = df.groupby('Holiday_Flag')['Weekly_Sales'].var()
    print("Variance Description:")
    for flag, variance in grouped_data.items():
        if flag == 1:
            print("Holiday Week:")
        else:
            print("Non-Holiday Week:")
        print("Variance:", variance)
        print()
        
    print('1e')
    average_sales_by_store = df.groupby('Store')['Weekly_Sales'].mean()
    is_average_sales_equal = average_sales_by_store.nunique() == 1
    if is_average_sales_equal:
        print("Rata-rata Weekly Sales di setiap toko sama.")
    else:
        print("Rata-rata Weekly Sales di setiap toko berbeda.")

    print('1f')
    max_cpi_by_store = df.groupby('Store')['CPI'].max()
    higher_cpi_by_store = max_cpi_by_store.idxmax()
    higher_cpi_value = max_cpi_by_store.max()
    
    print("CPI yang lebih tinggi di setiap toko:")
    for store_id in max_cpi_by_store.index:
        cpi_value = max_cpi_by_store.loc[store_id]
        print("Store ID:", store_id)
        print("CPI:", cpi_value)
        print()
        
    print('1g')
    average_cpi_holiday = df[df['Holiday_Flag'] == 1]['CPI'].mean()
    average_cpi_non_holiday = df[df['Holiday_Flag'] == 0]['CPI'].mean()
    if average_cpi_holiday > average_cpi_non_holiday:
        print("Rata-rata CPI pada holiday week lebih tinggi.")
    elif average_cpi_holiday < average_cpi_non_holiday:
        print("Rata-rata CPI pada non-holiday week lebih tinggi.")
    else:
        print("Rata-rata CPI pada holiday week dan non-holiday week sama.")

class Nomor2() :
    weekly_sales = df['Weekly_Sales']
    fuel_price = df['Fuel_Price']
    alpha = 0.05

    print('Nomor 2\n')

    statistic, p_value = kstest(weekly_sales, norm.fit(weekly_sales))
    print("Uji Normalitas Weekly Sales:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
    if p_value > alpha:
        print("Weekly Sales didistribusikan secara normal")
    else:
        print("Weekly Sales tidak didistribusikan secara normal")

    statistic, p_value = kstest(fuel_price, norm.fit(fuel_price))
    print("Uji Normalitas Fuel Price:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
    if p_value > alpha:
        print("Fuel Price didistribusikan secara normal")
    else:
        print("Fuel Price tidak didistribusikan secara normal")

class Nomor3() :
    print()
    print('Nomer 3')
    print('3a')
    correlation = df[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
    print("Nilai korelasi antara variabel independen dan variabel dependen:")
    print(correlation['Weekly_Sales'])
    
    print('3b')    
    correlation = df[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
    negative_correlations = correlation[correlation['Weekly_Sales'] < 0]
    negative_correlations = negative_correlations['Weekly_Sales'].drop('Weekly_Sales', errors='ignore')
    if negative_correlations.empty:
        print("Tidak ada pasangan variabel independen dan dependen dengan korelasi negatif.")
    else:
        print("Pasangan variabel independen dan dependen dengan korelasi negatif:")
        print(negative_correlations)

class Nomor4() :
    print('Nomer 4')
    # Mengambil data yang diperlukan
    data = df[['Fuel_Price', 'Weekly_Sales']]

    # Pisahkan variabel independen (X) dan dependen (y)
    X = data[['Fuel_Price']]
    y = data['Weekly_Sales']

    # Inisialisasi model regresi linear
    model = LinearRegression()

    # Melatih model menggunakan data
    model.fit(X, y)

    # Mendapatkan koefisien a dan b
    a = model.intercept_
    b = model.coef_[0]

    # Menampilkan model
    print("Model regresi: y = {} + {}x".format(a, b))

    # Mengambil data yang diperlukan
    data = df[['Fuel_Price', 'Weekly_Sales']]

    # Pisahkan variabel independen (X) dan dependen (y)
    X = data[['Fuel_Price']]
    y = data['Weekly_Sales']

    # Inisialisasi model regresi linear
    model = LinearRegression()

    # Melatih model menggunakan data
    model.fit(X, y)

    # Prediksi nilai y berdasarkan X
    y_pred = model.predict(X)

    # Menampilkan scatter plot data
    plt.scatter(X, y, color='blue', label='Data')

    # Menampilkan garis regresi
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    # Menampilkan label dan judul pada grafik
    plt.xlabel('Fuel_Price')
    plt.ylabel('Weekly_Sales')
    plt.title('Linear Regression')

    # Menampilkan legenda
    plt.legend()

    # Menampilkan grafik
    plt.show()