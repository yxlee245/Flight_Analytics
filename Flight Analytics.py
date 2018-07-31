### Analytics for flight delays

# Import libraries
from data_libraries import *
import datetime


## Preprocessing

# List of column headers
column_headers_list = ['Flight_ID', 'Callsign', 'Registration', 'Origin', 'Destination',
                      'Acft_Type', 'Scheduled_Block_Out', 'Scheduled_Block_In', 'Actual_Block_Out', 'Actual_Dep_Time',
                      'Actual_Arr_Time', 'Actual_Block_In', 'Enroute', 'Route', 'Op_Type', 'Cancelled']

# Read in flights data
input_filename_str = input('Enter file name of csv file: ')
flights_raw_df = pd.read_csv(input_filename_str, names=column_headers_list,
                            skiprows=1)

# Print size of raw dataframe
print('No of rows:', len(flights_raw_df))
print('No of columns:', len(flights_raw_df.columns))

# Display top 5 rows
flights_raw_df.head()

# Remove unnecessary columns for easier processing of data
# Mask for slicing dataframe
mask = [header for header in flights_raw_df.columns if header not in ['Registration', 'Route', 'Cancelled', 'Enroute',
                                                                     'Actual_Dep_Time', 'Actual_Arr_Time']]
# Slice dataframe
flights_reduced_df = flights_raw_df.loc[:, mask]

# Remove rows where airport code has more than 4 characters
# Keep flights where both origin and destination codes have only 4 characters
mask2_list = [i for i in flights_reduced_df.index if len(flights_reduced_df.Origin[i]) == 4 and len(flights_reduced_df.Destination[i]) == 4]
flights_reduced_df = flights_reduced_df.iloc[mask2_list, :]

# Print size of sliced dataframe
print('No of rows:', len(flights_reduced_df))
print('No of columns:', len(flights_reduced_df.columns))

# Display top 5 rows of reduced dataframe
flights_reduced_df.head()

# Function to count number of NaNs in specified column of raw dataframe
def count_nan(input_series):
    '''Takes in a specifc column from pandas dataframe, in series format
    Returns an int indicating the number of NaNs in the input series'''
    count_int = 0
    for i in input_series.index:
        try:
            if np.isnan(input_series[i]):
                count_int += 1
        except TypeError:
            continue
    return count_int

# Function to parse datetime
def parse_datetime(input_series, format_str):
    '''Takes in a panda Series containing tiemdate values in string format.
    and a string which represents the format to parse to datetime object
    Returns a Series containing datetime objects'''
    datetime_list = []
    for datetime_str in input_series:
        # Parse each string object in input series
        datetime_datetime = datetime.datetime.strptime(datetime_str, format_str)
        datetime_list.append(datetime_datetime)
    return pd.Series(datetime_list)

# Function to obtain time difference
def series_time_diff(series1, series2):
    '''Takes in 2 panda series containing panda datetime objects, and performs the operation series1 - series2
    Returns a Series containing the time difference in min in float format'''
    diff_series = series1 - series2
    diff_list = []
    for time_diff in diff_series:
        time_diff_float = time_diff.total_seconds() / 60
        diff_list.append(time_diff_float)
    return pd.Series(diff_list)

# Function to convert delay series to boolean (1/0) series
def convert_delay_bool(input_series, threshold_float=15):
    '''Converts a series containing delay times in minutes (float format), to Boolean values which indicates delay,
    based on specified input threshold (default threshold value of 15 minutes).
    Returns a series containing Boolean values indicating delay (1 - delay, 0 - no delay)'''
    indicator_int = []
    for value in input_series:
        if np.isnan(value):
            indicator_int.append(np.NaN)
        elif value > threshold_float:
            indicator_int.append(1)
        else:
            indicator_int.append(0)
    return pd.Series(indicator_int)

# Determine number of NaNs in Actual_Block_In, Actual_Block_Out, Scheduled_Block_Out and Scheduled_Block_In
actual_dep_count_int = count_nan(flights_reduced_df.Actual_Block_Out)
sch_dep_count_int = count_nan(flights_reduced_df.Scheduled_Block_Out)
actual_arr_count_int = count_nan(flights_reduced_df.Actual_Block_In)
sch_arr_count_int = count_nan(flights_reduced_df.Scheduled_Block_In)       

# Display number of NaNs
print('No of NaNs for:')
print('Actual Departure Time -', actual_dep_count_int)
print('Scheduled Departure Time -', sch_dep_count_int)
print('Actual Arrival Time -', actual_arr_count_int)
print('Scheduled Arrival Time -', sch_arr_count_int)

# Delete rows with NaN
flights_nonan_df = flights_reduced_df.dropna(inplace=False)
flights_nonan_df.reset_index(drop=True, inplace=True)

# Print size of sliced dataframe
print('No of rows:', len(flights_nonan_df))
print('No of columns:', len(flights_nonan_df.columns))

# Display top 5 rows
flights_nonan_df.head()

# Determine Gate Departure and Arrival Delay
dt_format_str = '%Y-%m-%d %H:%M'
dep_delay_float_series = series_time_diff(series1=parse_datetime(flights_nonan_df.Actual_Block_Out, format_str=dt_format_str),
                                    series2=parse_datetime(flights_nonan_df.Scheduled_Block_Out, format_str=dt_format_str))
arr_delay_float_series = series_time_diff(series1=parse_datetime(flights_nonan_df.Actual_Block_In, format_str=dt_format_str),
                                    series2=parse_datetime(flights_nonan_df.Scheduled_Block_In, format_str=dt_format_str))

# Display length of Dep Delay and Arr Delay series
print('Length of Departure Delay Float Series:', len(dep_delay_float_series))
print('Length of Arrival Delay Float Series:', len(arr_delay_float_series))

# Derive boolean series for depature and arrival delay, based on threshold value
dep_delay_bool_series = convert_delay_bool(dep_delay_float_series)
arr_delay_bool_series = convert_delay_bool(arr_delay_float_series)

# Display length of Dep and Arr late indicator series
print('Length of Departure Delay Float Series:', len(dep_delay_bool_series))
print('Length of Arrival Delay Float Series:', len(arr_delay_bool_series))

# Assign dataframe with fields from original dataframe and derived series
column_headers_list2 = ['Flight_ID', 'Callsign', 'Origin', 'Destination', 'Acft_type',
                       'Dep_Delay_Float', 'Dep_Delay_Bool', 'Arr_Delay_Float', 'Arr_Delay_Bool', 'Op_Type']
flights_processed_df = pd.DataFrame({
    'Flight_ID': flights_nonan_df.Flight_ID,
    'Callsign': flights_nonan_df.Callsign,
    'Origin': flights_nonan_df.Origin,
    'Destination': flights_nonan_df.Destination,
    'Acft_type': flights_nonan_df.Acft_Type,
    'Dep_Delay_Float': dep_delay_float_series,
    'Dep_Delay_Bool': dep_delay_bool_series,
    'Arr_Delay_Float': arr_delay_float_series,
    'Arr_Delay_Bool': arr_delay_bool_series,
    'Op_Type': flights_nonan_df.Op_Type
}, columns=column_headers_list2)

# Print size of processed dataframe
print('No of rows:', len(flights_processed_df))
print('No of columns:', len(flights_processed_df.columns))

# Print top 5 rows of dataframe
flights_processed_df.head()


## Exploratory Data Analysis

# Find mean and SD for all Arrival Delay Values
print('Mean:', np.mean(flights_processed_df.Arr_Delay_Float))
print('SD:', np.std(flights_processed_df.Arr_Delay_Float))

# Find correlation matrix
print(flights_processed_df.corr())

# Scatter plot of Arr_Delay_Float against Dep_Delay_Float
plt.scatter(flights_processed_df.Dep_Delay_Float, flights_processed_df.Arr_Delay_Float)
plt.xlabel('Departure Delay (min)')
plt.ylabel('Arrival Delay (min)')
plt.title('Scatter Plot of Arrival Delay against Departure Delay')
plt.show()

# Linear Regression Line for Arrival vs Departure Delay

# Filtering out cases where departure delay < -500 min
filter_list = [i for i in flights_processed_df.index if flights_processed_df.Dep_Delay_Float[i] > -500]
flights_filtered_df = flights_processed_df.iloc[filter_list,]

# Initialize linear regression model
initial_model = lm.LinearRegression()

# Fit model to data
initial_model.fit(flights_filtered_df.Dep_Delay_Float[:, np.newaxis], flights_filtered_df.Arr_Delay_Float)

# Show plot for regression line
plt.scatter(flights_processed_df.Dep_Delay_Float, flights_processed_df.Arr_Delay_Float, label='Actual Data')
plt.plot(flights_filtered_df.Dep_Delay_Float[:, np.newaxis],
         initial_model.predict(flights_filtered_df.Dep_Delay_Float[:, np.newaxis]), 'r-',
        label='Regression Line')
plt.xlabel('Departure Delay (min)')
plt.ylabel('Arrival Delay (min)')
plt.title('Scatter and Regression Plot of Arrival Delay against Departure Delay')
plt.legend(loc='best')
plt.show()


# Improving Dataset (Attempt #1)

# Determine Mean and SD of departure and arrival delays for each airport

# Getting unique series of airports
dep_airports_series = flights_processed_df.Origin.drop_duplicates()
arr_airports_series = flights_processed_df.Destination.drop_duplicates()

# Determine Mean and SD for Departure and Arrival Delays
mean_dep_delays_list = []
sd_dep_delays_list = []
for dep_airport in dep_airports_series:
    # Mean
    mean_dep_delay = np.mean(flights_processed_df.Dep_Delay_Float[flights_processed_df.Origin == dep_airport])
    mean_dep_delays_list.append(mean_dep_delay)
    # SD
    sd_dep_delay = np.std(flights_processed_df.Dep_Delay_Float[flights_processed_df.Origin == dep_airport])
    sd_dep_delays_list.append(sd_dep_delay)
    
mean_arr_delays_list = []
sd_arr_delays_list = []
for arr_airport in arr_airports_series:
    # Mean
    mean_arr_delay = np.mean(flights_processed_df.Arr_Delay_Float[flights_processed_df.Destination == arr_airport])
    mean_arr_delays_list.append(mean_arr_delay)
    # SD
    sd_arr_delay = np.std(flights_processed_df.Arr_Delay_Float[flights_processed_df.Destination == arr_airport])
    sd_arr_delays_list.append(sd_arr_delay)

# Creating data frame for statistics of departure and arrival delays (by airport)
dep_delays_ap_df = pd.DataFrame({
    'Airport_Code': dep_airports_series,
    'Delay_Mean': mean_dep_delays_list,
    'Delay_SD': sd_dep_delays_list
}, columns=['Airport_Code', 'Delay_Mean', 'Delay_SD'])
arr_delays_ap_df = pd.DataFrame({
    'Airport_Code': arr_airports_series,
    'Delay_Mean': mean_arr_delays_list,
    'Delay_SD': sd_arr_delays_list
}, columns=['Airport_Code', 'Delay_Mean', 'Delay_SD'])

# Display dataframe for WSSS (Changi Airport)
print(dep_delays_ap_df[dep_delays_ap_df.Airport_Code == 'WSSS'])
print(arr_delays_ap_df[arr_delays_ap_df.Airport_Code == 'WSSS'])


# Extract from raw dataframe flights with NaN in Actual Departure and Arrival Times with correct airport code format
flights_nan_df = flights_reduced_df[:]  # Create copy of dataframe to prevent accidental corruption
flights_nan_df.reset_index(drop=True, inplace=True)
nan_filter_list = []
for i in flights_nan_df.index:
    try:
        if (np.isnan(flights_nan_df.Actual_Block_In[i]) or
           np.isnan(flights_nan_df.Actual_Block_Out[i]) or
           np.isnan(flights_nan_df.Scheduled_Block_In[i]) or
           np.isnan(flights_nan_df.Scheduled_Block_Out[i])):
            nan_filter_list.append(i)
    except TypeError:
        continue
flights_nan2_df = flights_nan_df.iloc[nan_filter_list, :]
flights_nan2_df.reset_index(drop=True, inplace=True)
flights_nan2_df.head()

# Compute / Estimate Delay Times
dep_delay_nan_list = []
arr_delay_nan_list = []
for i in flights_nan2_df.index:
    # Dep Delays
    if (type(flights_nan2_df.Actual_Block_Out[i]) == str) and (type(flights_nan2_df.Scheduled_Block_Out[i]) == str):
        actual_dep_datetime = datetime.datetime.strptime(flights_nan2_df.Actual_Block_Out[i], dt_format_str)
        sch_dep_datetime = datetime.datetime.strptime(flights_nan2_df.Scheduled_Block_Out[i], dt_format_str)
        dep_delay_td = actual_dep_datetime - sch_dep_datetime
        dep_delay_float = dep_delay_td.total_seconds() / 60
    else:
        delay_mean_flt = dep_delays_ap_df.Delay_Mean[dep_delays_ap_df.Airport_Code == flights_nan2_df.Origin[i]]
        delay_sd_flt = dep_delays_ap_df.Delay_SD[dep_delays_ap_df.Airport_Code == flights_nan2_df.Origin[i]]
        dep_delay_array = np.random.normal(delay_mean_flt, delay_sd_flt)
        if len(dep_delay_array) == 0:  # airport not found
            dep_delay_float = np.NaN
        else:
            dep_delay_float = int(dep_delay_array)
    dep_delay_nan_list.append(dep_delay_float)
    
    # Arr Delays
    if (type(flights_nan2_df.Actual_Block_In[i]) == str) and (type(flights_nan2_df.Scheduled_Block_In[i]) == str):
        actual_arr_datetime = datetime.datetime.strptime(flights_nan2_df.Actual_Block_In[i], dt_format_str)
        sch_arr_datetime = datetime.datetime.strptime(flights_nan2_df.Scheduled_Block_In[i], dt_format_str)
        arr_delay_td = actual_arr_datetime - sch_arr_datetime
        arr_delay_float = arr_delay_td.total_seconds() / 60
    else:
        delay_mean_flt = arr_delays_ap_df.Delay_Mean[arr_delays_ap_df.Airport_Code == flights_nan2_df.Destination[i]]
        delay_sd_flt = arr_delays_ap_df.Delay_SD[arr_delays_ap_df.Airport_Code == flights_nan2_df.Destination[i]]
        arr_delay_array = np.random.normal(delay_mean_flt, delay_sd_flt)
        if len(arr_delay_array) == 0:  # airport not found
            arr_delay_float = np.NaN
        else:
            arr_delay_float = int(arr_delay_array)
    arr_delay_nan_list.append(arr_delay_float)

# Create Boolean series for departure and arrival of NaN flights
dep_delay_nan_bool_list = convert_delay_bool(pd.Series(dep_delay_nan_list))
arr_delay_nan_bool_list = convert_delay_bool(pd.Series(arr_delay_nan_list))
len(dep_delay_nan_bool_list), len(arr_delay_nan_bool_list)

# Assign dataframe with fields from NaN dataframe and derived series
column_headers_list2 = ['Flight_ID', 'Callsign', 'Origin', 'Destination', 'Acft_type',
                       'Dep_Delay_Float', 'Dep_Delay_Bool', 'Arr_Delay_Float', 'Arr_Delay_Bool', 'Op_Type']
flights_temp_df = pd.DataFrame({
    'Flight_ID': flights_nan2_df.Flight_ID,
    'Callsign': flights_nan2_df.Callsign,
    'Origin': flights_nan2_df.Origin,
    'Destination': flights_nan2_df.Destination,
    'Acft_type': flights_nan2_df.Acft_Type,
    'Dep_Delay_Float': dep_delay_nan_list,
    'Dep_Delay_Bool': dep_delay_nan_bool_list,
    'Arr_Delay_Float': arr_delay_nan_list,
    'Arr_Delay_Bool': arr_delay_nan_bool_list,
    'Op_Type': flights_nan2_df.Op_Type
}, columns=column_headers_list2)

# Remove NaNs
flights_temp_df.dropna(inplace=True)

# Combine flights_processed_df with flights_temp_df
flights_processed2_df = pd.concat([flights_processed_df, flights_temp_df], ignore_index=True)

# Display size of dataframe
print('No of rows:', len(flights_processed2_df))
print('No of columns:', len(flights_processed2_df.columns))

# Histogram
plt.figure(figsize=(12, 6))

plt.suptitle('Histogram Plots', fontsize=16)

plt.subplot(121)
plt.hist(flights_processed2_df.Dep_Delay_Float, bins=30)
plt.xlabel('Departure Delay (min)')

plt.subplot(122)
plt.hist(flights_processed2_df.Arr_Delay_Float, bins=30)
plt.ylabel('Arrival Delay (min)')

plt.show()

# Boxplot
flights_processed2_df.loc[:, ['Dep_Delay_Float', 'Arr_Delay_Float']].boxplot()
plt.title('Boxplot', fontsize=16)
plt.show()

# Scatter plot of Arr_Delay_Float against Dep_Delay_Float
plt.scatter(flights_processed2_df.Dep_Delay_Float, flights_processed2_df.Arr_Delay_Float)
plt.xlabel('Departure Delay (min)')
plt.ylabel('Arrival Delay (min)')
plt.title('Scatter Plot of Arrival Delay against Departure Delay')
plt.show()

# Correlation matrix
print(flights_processed2_df.loc[:, ['Dep_Delay_Float', 'Arr_Delay_Float']].corr())

# Determine Count, Mean and SD of departure and arrival delays for each airport (from flights_processed2_df)

# Getting unique series of airports
dep_airports2_series = flights_processed2_df.Origin.drop_duplicates()
arr_airports2_series = flights_processed2_df.Destination.drop_duplicates()

# Determine Count, Mean and SD for Departure and Arrival Delays
count_dep_delays2_list = []
mean_dep_delays2_list = []
sd_dep_delays2_list = []
for dep_airport in dep_airports2_series:
    # Count
    count_dep_delay = len(flights_processed2_df.loc[flights_processed2_df.Origin == dep_airport,])
    count_dep_delays2_list.append(count_dep_delay)
    # Mean
    mean_dep_delay = np.mean(flights_processed2_df.Dep_Delay_Float[flights_processed2_df.Origin == dep_airport])
    mean_dep_delays2_list.append(mean_dep_delay)
    # SD
    sd_dep_delay = np.std(flights_processed2_df.Dep_Delay_Float[flights_processed2_df.Origin == dep_airport])
    sd_dep_delays2_list.append(sd_dep_delay)

count_arr_delays2_list = []
mean_arr_delays2_list = []
sd_arr_delays2_list = []
for arr_airport in arr_airports2_series:
    # Count
    count_arr_delay = len(flights_processed2_df.loc[flights_processed2_df.Destination == arr_airport,])
    count_arr_delays2_list.append(count_arr_delay)
    # Mean
    mean_arr_delay = np.mean(flights_processed2_df.Arr_Delay_Float[flights_processed2_df.Destination == arr_airport])
    mean_arr_delays2_list.append(mean_arr_delay)
    # SD
    sd_arr_delay = np.std(flights_processed2_df.Arr_Delay_Float[flights_processed2_df.Destination == arr_airport])
    sd_arr_delays2_list.append(sd_arr_delay)
    
# Creating data frame for statistics of departure and arrival delays (by airport)
dep_delays_ap2_df = pd.DataFrame({
    'Airport_Code': dep_airports2_series,
    'Count': count_dep_delays2_list,
    'Delay_Mean': mean_dep_delays2_list,
    'Delay_SD': sd_dep_delays2_list
}, columns=['Airport_Code', 'Count', 'Delay_Mean', 'Delay_SD'])
arr_delays_ap2_df = pd.DataFrame({
    'Airport_Code': arr_airports2_series,
    'Count': count_arr_delays2_list,
    'Delay_Mean': mean_arr_delays2_list,
    'Delay_SD': sd_arr_delays2_list
}, columns=['Airport_Code', 'Count', 'Delay_Mean', 'Delay_SD'])

# Display dataframe for WSSS (Changi Airport)
print(dep_delays_ap2_df[dep_delays_ap2_df.Airport_Code == 'WSSS'])
print(arr_delays_ap2_df[arr_delays_ap2_df.Airport_Code == 'WSSS'])

# Get list of arrival counts for every flight origin and destination airport
arr_counts_list = []
for arr_airport in flights_processed2_df.Destination:
    arr_count_int = int(arr_delays_ap2_df.Count[arr_delays_ap2_df.Airport_Code == arr_airport])
    arr_counts_list.append(arr_count_int)

## Confirmatory Data Analysis

# Build first version of regression model

# Independent Variables
X = pd.DataFrame({
    'Arrival_Count': arr_counts_list,
    'Dep_Delay': flights_processed2_df.Dep_Delay_Float
})

# Dependent Variable
y = flights_processed2_df.Arr_Delay_Float

# Correlation Matrix
print(X.corr())

# Continue with regression model since low correlation between arrival count and departure delay

# Split data for train/test datasets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=1)

# Create fitted model and display summary
lm1_model = sm.OLS(y_train, X_train).fit()
print(lm1_model.summary())

# Improving dataset (Attempt #2)

# Extract list of aircraft types from flights_processed2_df
flights_processed2_df.Acft_type.to_csv(path='Aircraft Type.csv', header=True, index_label=False, index=False)

# Read in dataframe of airctraft types with associated Wake Turbulence Category (WTC)
# WTC information from https://www.skybrary.aero and http://www.aviationfanatic.com
acft_wtc_df = pd.read_csv('Aircraft Type WTC.csv')
acft_wtc_df.head()

# Create list of WTC (Heavy, Medium, Light, Unknown) for all flights in flights_processed2_df
acft_wtc_list = []
for acft_type in flights_processed2_df.Acft_type:
    wtc_str = acft_wtc_df.WTC[acft_wtc_df.Acft_type == acft_type].values[0]
    acft_wtc_list.append(wtc_str)

# Make a copy of flights_processed2_df and add column for WTC
flights_processed3_df = flights_processed2_df[:]
flights_processed3_df.loc[:,'WTC'] = acft_wtc_list
flights_processed3_df.head()

# Compute proportion of aircrafts with WTC of 'Heavy' for every arrival airport
heavy_prop_list = []
for i in arr_delays_ap2_df.index:
    # Get subset containing all flights at a specific arrival airport
    flights_subset_df = flights_processed3_df.loc[flights_processed3_df.Destination == arr_delays_ap2_df.Airport_Code[i],]
    # Further obtain the flights of WTC 'Heavy'
    flights_heavy_df = flights_subset_df.loc[flights_subset_df.WTC == 'Heavy',]
    heavy_count_int = len(flights_heavy_df)
    heavy_prop_float = heavy_count_int / int(arr_delays_ap2_df.Count[i])
    heavy_prop_list.append(heavy_prop_float)

# Create new column in arr_delays_ap2_df for proportion of 'Heavy' flights
arr_delays_ap2_df.loc[:,'Heavy_Prop'] = heavy_prop_list
print(len(arr_delays_ap2_df))

# Create list of proportion of WTC 'Heavy' for all flights in flights_processed3_df 
flights_heavy_prop_list = []
for arr_airport in flights_processed3_df.Destination:
    heavy_prop_float = arr_delays_ap2_df.Heavy_Prop[arr_delays_ap2_df.Airport_Code == arr_airport].values[0]
    flights_heavy_prop_list.append(heavy_prop_float)

# Create new column in flights_processed3_df for arr_counts_list and heavy_prop
flights_processed3_df.loc[:, 'Heavy_Prop'] = flights_heavy_prop_list
flights_processed3_df.loc[:, 'Arrival_Count'] = arr_counts_list

# Build second version of regression model

# Independent Variables
X = flights_processed3_df[['Dep_Delay_Float', 'Arrival_Count', 'Heavy_Prop']]

# Dependent Variable
y = flights_processed3_df.Arr_Delay_Float

# Correlation Matrix
print(X.corr())

### Continue with regression model since low correlation between arrival count and departure delay

# Split data for train/test datasets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=1)

# Create fitted model and display summary
lm2_model = sm.OLS(y_train, X_train).fit()
print(lm2_model.summary())
