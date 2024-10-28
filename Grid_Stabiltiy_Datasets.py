#!/usr/bin/env python
# coding: utf-8

# In[1]:

    
#DATASETS PREPROCESSING AND ANALYSIS

import pandas as pd

pd.set_option('display.max_rows', None)  # Show all rows for "variable understanding" part of the project
pd.set_option('display.max_columns', None)  # Show all columns for "variable understanding" part of the project

#Specifing path to Excel file
excel_file_path = 'Pathname Wind-Turbine-SCADA-signals-2017_0.xlsx'

#Load SCADA dataset file into a pandas DataFrame
df_SCADA = pd.read_excel(excel_file_path)

#Output total count of values and unique count for each variable
df_SCADAColumnAnalysis = pd.DataFrame({"columns": df_SCADA.columns,
                                 "data Type": df_SCADA.dtypes.values,
                                 "Total Count": df_SCADA.count().values,
                                 "Unique Count": df_SCADA.nunique().values
                                })
df_SCADAColumnAnalysis


# In[2]:


#Converting non-numerical data types to int/float because "object" dt cannot be used in regression based ml models
#Remove the 'T' prefix from the 'Turbine_ID' values
df_SCADA['Turbine_ID'] = df_SCADA['Turbine_ID'].astype(str).str.replace('T', '')

#Convert the "Turbine_ID" column to integers
df_SCADA['Turbine_ID'] = df_SCADA['Turbine_ID'].astype(int)

#Convert the "Timestamp" column to datetime format
df_SCADA['Timestamp'] = pd.to_datetime(df_SCADA['Timestamp'])

#Checking the data types to confirm changes
df_SCADA.dtypes


# In[3]:


#Specifing path to Excel file
excel_file_path = 'Pathname Onsite-MetMast-SCADA-data-2017.xlsx'

#Load MetMast dataset file into a pandas DataFrame
df_MetMast = pd.read_excel(excel_file_path)

#Output total count of values and unique count for each variable
df_MetMastColumnAnalysis = pd.DataFrame({"columns": df_MetMast.columns,
                                 "data Type": df_MetMast.dtypes.values,
                                 "Total Count": df_MetMast.count().values,
                                 "Unique Count": df_MetMast.nunique().values
                                })
df_MetMastColumnAnalysis


# In[4]:


#Convert the 'Timestamp' column to datetime format because "object" dt
#cannot be used in regression based ml models
df_MetMast['Timestamp'] = pd.to_datetime(df_MetMast['Timestamp'])

#Check the data types to confirm changes
df_MetMast.dtypes


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from pandas.plotting import autocorrelation_plot

# Assuming df_SCADA and df_MetMast are already loaded and need fixing of the Timestamp data type
# Convert Timestamp columns to datetime if they are not already

# Exploring time frequency for df_SCADA and df_MetMast dataset

# For df_SCADA
scada_min_time = df_SCADA['Timestamp'].min()
scada_max_time = df_SCADA['Timestamp'].max()
scada_time_diff = df_SCADA['Timestamp'].diff().dropna()  # Calculate differences between consecutive timestamps and drop the first NaN value

# Print results
print("SCADA Dataset:")
print(f"Minimum Recorded Time: {scada_min_time}")
print(f"Maximum Recorded Time: {scada_max_time}")
print(f"Common Frequency: {scada_time_diff.mode()[0]}")  # mode() finds the most common timedelta

#Get the earliest and latest 'Timestamp' for each Turbine ID
timestamp_range = df_SCADA.groupby('Turbine_ID')['Timestamp'].agg(['min', 'max'])

# The result is a DataFrame with the oldest and most recent 'Timestamp' for each 'Turbine_ID'
print(timestamp_range)

# Process for df_SCADA
df_SCADA['Time_Diff'] = df_SCADA['Timestamp'].diff().dt.total_seconds().dropna()

# Descriptive Statistics for df_SCADA
print("Descriptive Statistics for Time Differences - SCADA:")
print(df_SCADA['Time_Diff'].describe())

# Histogram Analysis for df_SCADA
plt.figure(figsize=(10, 6))
df_SCADA['Time_Diff'].plot(kind='hist', bins=30, title='Histogram of Time Differences - SCADA')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')
plt.show()

# Fourier Analysis for df_SCADA
fft_values_scada = fft(df_SCADA['Time_Diff'].dropna().values)
frequencies_scada = np.abs(fft_values_scada)
plt.figure(figsize=(10, 6))
plt.plot(frequencies_scada, label='Magnitude of FFT Coefficients - SCADA')
plt.title('Fourier Analysis of Time Differences - SCADA')
plt.xlabel('Frequency Component')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

# For df_MetMast
metmast_min_time = df_MetMast['Timestamp'].min()
metmast_max_time = df_MetMast['Timestamp'].max()
metmast_time_diff = df_MetMast['Timestamp'].diff().dropna()

print("\nMetMast Dataset:")
print(f"Minimum Recorded Time: {metmast_min_time}")
print(f"Maximum Recorded Time: {metmast_max_time}")
print(f"Common Frequency: {metmast_time_diff.mode()[0]}")

# Same process for df_MetMast
df_MetMast['Time_Diff'] = df_MetMast['Timestamp'].diff().dt.total_seconds().dropna()

# Descriptive Statistics for df_MetMast
print("Descriptive Statistics for Time Differences - MetMast:")
print(df_MetMast['Time_Diff'].describe())

# Histogram Analysis for df_MetMast
plt.figure(figsize=(10, 6))
df_MetMast['Time_Diff'].plot(kind='hist', bins=30, title='Histogram of Time Differences - MetMast')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')
plt.show()

# Fourier Analysis for df_MetMast
fft_values_metmast = fft(df_MetMast['Time_Diff'].dropna().values)
frequencies_metmast = np.abs(fft_values_metmast)
plt.figure(figsize=(10, 6))
plt.plot(frequencies_metmast, label='Magnitude of FFT Coefficients - MetMast')
plt.title('Fourier Analysis of Time Differences - MetMast')
plt.xlabel('Frequency Component')
plt.ylabel('Magnitude')
plt.legend()
plt.show()


# In[6]:


#Merging
#Outer join is performed to ensure no data is lost from either dataset.
#Since there are multiple turbines per timestamp in the SCADA data, each turbine entry is 
#matched with the corresponding MetMast entry
df_merged = pd.merged(df_SCADA, df_MetMast, on='Timestamp', how='outer', suffixes=('', '_MetMast'))

#Output total count of values and unique count for each variable for merged dataset
df_mergedColumnAnalysis = pd.DataFrame({"columns": df_merged.columns,
                                 "data Type": df_merged.dtypes.values,
                                 "Total Count": df_merged.count().values,
                                 "Unique Count": df_merged.nunique().values
                                })
df_mergedColumnAnalysis


# In[7]:


#Datasets merged with different number of "Total Count" for columns => have to find out why
#Check entire df for null values
ContainsNullValues = df_merged.isnull().values.any() 
print("does dataset contain null values: ", ContainsNullValues)
if ContainsNullValues:
    print("null values found in columns: \n", df_merged.isnull().sum())


# In[10]:


#Checking count of records per day for merged dataset and making a plot
daily_counts = df_merged.set_index('Timestamp').resample('D').size()

plt.figure(figsize=(10, 6))

daily_counts.plot()
plt.title('Daily Data Points Count Merged Dataset')
plt.xlabel('Date')
plt.ylabel('Number of Data Points')

plt.tight_layout()
plt.show()

#Count of records per day for SCADA dataset and a plot

daily_counts = df_SCADA.set_index('Timestamp').resample('D').size()

plt.figure(figsize=(10, 6))

daily_counts.plot()
plt.title('Daily Data Points Count in SCADA')
plt.xlabel('Date')
plt.ylabel('Number of Data Points')

plt.tight_layout()
plt.show()

#Count of records per day for MetMast dataset and a plot
daily_counts = df_MetMast.set_index('Timestamp').resample('D').size()

plt.figure(figsize=(10, 6))

daily_counts.plot()
plt.title('Daily Data Points Count in MetMast')
plt.xlabel('Date')
plt.ylabel('Number of Data Points')

plt.tight_layout()
plt.show()


# In[11]:


#Dropping rows with null values in merged dataset where data hasn't been recorded
df_merged = df_merged.dropna()

ContainsNullValues = df_merged.isnull().values.any() 
print("does dataset contain null values: ", ContainsNullValues)
if ContainsNullValues:
    print("null values found in columns: \n", df_merged.isnull().sum())


# In[ ]:





# In[13]:


import seaborn as sns

#Extracting the hour from the 'Timestamp' for frequency analysis in merged dataset
df_merged['Hour'] = df_merged['Timestamp'].dt.hour

plt.figure(figsize=(10, 6))
sns.histplot(df_merged['Hour'], bins=24, kde=False)
plt.title('Frequency of Recorded Times by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.xticks(range(24))  # To show every hour on the x-axis

plt.show()

#Unique turbine IDs
turbine_ids = df_merged['Turbine_ID'].unique()

#Plotting the frequency distribution of times for each turbine
for turbine_id in turbine_ids:
    
    df_turbine = df_merged[df_merged['Turbine_ID'] == turbine_id]
    
    plt.figure(figsize=(12, 6))
    df_turbine['Hour'].hist(bins=24, range=(0,24), alpha=0.7, edgecolor='black')

    plt.title(f'Frequency of Recorded Times by Hour for Turbine {turbine_id}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.xticks(range(24))  
    plt.grid(False)  
    plt.show()


# In[14]:


#Time frequency analysis for df_merged
merged_min_time = df_merged['Timestamp'].min()
merged_max_time = df_merged['Timestamp'].max()
merged_time_diff = df_merged['Timestamp'].diff().dropna()

print("\nMerged Dataset:")
print(f"Minimum Recorded Time: {merged_min_time}")
print(f"Maximum Recorded Time: {merged_max_time}")
print(f"Common Frequency: {merged_time_diff.mode()[0]}")

df_merged['Time_Diff'] = df_merged['Timestamp'].diff().dt.total_seconds().dropna()

#Descriptive Statistics for df_merged
print("Descriptive Statistics for Time Differences - Merged Dataset:")
print(df_merged['Time_Diff'].describe())

#Histogram Analysis for df_merged
plt.figure(figsize=(10, 6))
df_merged['Time_Diff'].plot(kind='hist', bins=30, title='Histogram of Time Differences - Merged Dataset')
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')
plt.show()

#Fourier Analysis for the merged dataset
fft_values_merged = fft(df_merged['Time_Diff'].dropna().values)
frequencies_merged = np.abs(fft_values_metmast)
plt.figure(figsize=(10, 6))
plt.plot(frequencies_merged, label='Magnitude of FFT Coefficients - Merged Dataset')
plt.title('Fourier Analysis of Time Differences - Merged Datasett')
plt.xlabel('Frequency Component')
plt.ylabel('Magnitude')
plt.legend()
plt.show()


# In[15]:


#Automatically selecting numeric variables for plotting
variables = df_merged.select_dtypes(include=['float64', 'int64']).columns

#Determine the grid size for plotting because of a large number of variables
n_vars = len(variables)
ncols = 4  
nrows = (n_vars + ncols - 1) // ncols  

#Figure to hold subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*3))
fig.tight_layout(pad=3.0)

#Flatten the axes array for easier iteration when handling multi-dimensional data
axes = axes.flatten()

#Plot each variable
for i, var in enumerate(variables):
    ax = axes[i]
    df_merged[var].dropna().hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(var, fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Hide any unused subplot areas
for j in range(i+1, nrows*ncols):
    fig.delaxes(axes[j])

plt.subplots_adjust(top=0.95)
plt.suptitle('Histograms of Variables in Merged Dataset', fontsize=16, y=1.02)
plt.show()


# In[16]:


#Check entire dataset for null values:
ContainsNullValues = df_merged.isnull().values.any() 
print("does dataset contain null values: ", ContainsNullValues)
if ContainsNullValues:
    print("null values found in columns: \n", df_merged.isnull().sum())


# In[17]:


df_merged = df_merged.drop('Time_Diff', axis=1)


# In[18]:


from sklearn.ensemble import IsolationForest

#Select numerical features excluding 'Turbine_ID' and 'Timestamp'
features = df_merged.drop(['Turbine_ID', 'Timestamp'], axis=1)

#Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

#Fit the model on the selected features
iso_forest.fit(features)

#Predict the outliers (-1 for outliers, 1 for inliers)
outliers = iso_forest.predict(features)

#Add the outliers predictions as a new column named 'Anomaly'
df_merged['Anomaly'] = outliers

#Output how many anomalies were detected
print(df_merged['Anomaly'].value_counts())

print(df_merged.head())


# In[19]:


df_mergedColumnAnalysis = pd.DataFrame({"columns": df_merged.columns,
                                 "data Type": df_merged.dtypes.values,
                                 "Total Count": df_merged.count().values,
                                 "Unique Count": df_merged.nunique().values
                                })
df_mergedColumnAnalysis


# In[20]:


#Specify path to Excel file
excel_file_path = '/Pathname Wind Turbines Logs 2017.xlsx'

#Load the Excel file into a pandas DataFrame
logs_df = pd.read_excel(excel_file_path)

#Convert the timestamp columns to datetime 
logs_df['Time_Detected'] = pd.to_datetime(logs_df['Time_Detected'])

logs_dfColumnAnalysis = pd.DataFrame({"columns": logs_df.columns,
                                 "data Type": logs_df.dtypes.values,
                                 "Total Count": logs_df.count().values,
                                 "Unique Count": logs_df.nunique().values
                                })
logs_dfColumnAnalysis


# In[21]:


#Merging the datasets on 'Turbine_ID'/'Turbine_Identifier' and 'Timestamp'/'Time_Detected'
#Remove the 'T' prefix from the 'Turbine_ID' values in Logs dataset
logs_df['Turbine_Identifier'] = logs_df['Turbine_Identifier'].astype(str).str.replace('T', '')

#Convert the 'Turbine_ID' column to integers
logs_df['Turbine_Identifier'] = logs_df['Turbine_Identifier'].astype(int)
#Filtering the df_merged for outliers before merging
outliers_df = df_merged[df_merged['Anomaly'] == -1]
merged_with_logs = pd.merge(outliers_df,
                            logs_df,
                            how='left',
                            left_on=['Turbine_ID', 'Timestamp'],
                            right_on=['Turbine_Identifier', 'Time_Detected'])


# In[22]:


merged_with_logsColumnAnalysis = pd.DataFrame({"columns": merged_with_logs.columns,
                                 "data Type": merged_with_logs.dtypes.values,
                                 "Total Count": merged_with_logs.count().values,
                                 "Unique Count": merged_with_logs.nunique().values
                                })
merged_with_logsColumnAnalysis


# In[23]:


pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
#Check for matching records in the logs
#Add a new column to indicate if there's a log entry for the outlier
merged_with_logs['Has_Log'] = merged_with_logs['Remark'].notna()

#Select only outliers with logs and normal observations filtering to get only the outliers with corresponding logs
valid_outliers = merged_with_logs[merged_with_logs['Has_Log']]

#Filtering the original df_merged to get all normal observations
normal_observations = df_merged[df_merged['Anomaly'] != -1]

#Match valid outliers with normal observations
df_merged_cleaned = pd.concat([normal_observations, valid_outliers])

df_merged_cleaned.reset_index(drop=True, inplace=True)

#Display the updated DataFrame
print(df_merged_cleaned.shape)
print(df_merged_cleaned.head())

#Show summary to verify the number of outliers with and without logs
print("Summary of anomalies:")
print(df_merged_cleaned['Anomaly'].value_counts())


# In[24]:


#Rename df_merged_cleaned to df_merged
df_merged = df_merged_cleaned

#Drop the last several columns from the DataFrame
columns_to_drop = ['Hour', 'Anomaly', 'Time_Detected', 'Time_Reset', 
                   'Turbine_Identifier', 'Remark', 'Unit_Title_Destination', 'Has_Log', 'Time_Diff_MetMast']
df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

df_mergedColumnAnalysis = pd.DataFrame({"columns": df_merged.columns,
                                 "data Type": df_merged.dtypes.values,
                                 "Total Count": df_merged.count().values,
                                 "Unique Count": df_merged.nunique().values})
df_mergedColumnAnalysis
                       


# In[25]:


#Select numeric variables for plotting
variables = df_merged.select_dtypes(include=['float64', 'int64']).columns

#Plotting
n_vars = len(variables)
ncols = 4 
nrows = (n_vars + ncols - 1) // ncols  

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*3))
fig.tight_layout(pad=3.0)

axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    # Check for and drop NA values for a clean histogram
    df_merged[var].dropna().hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(var, fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

for j in range(i+1, nrows*ncols):
    fig.delaxes(axes[j])

plt.subplots_adjust(top=0.95)
plt.suptitle('Histograms of Variables in Merged Cleaned Dataset', fontsize=16, y=1.02)
plt.show()


# In[26]:


#A new statistical summary for merged dataset after outlier cleaning
statistical_summary = df_merged.describe()

full_statistical_summary = df_merged.describe(include='all')

print(statistical_summary)


# In[27]:


#Making the correlation matrix
numeric_features = df_merged.select_dtypes(include=[np.number])

correlation_matrix = numeric_features.corr()

plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Features', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[28]:


#Drop columns as they don't provide any valuable information and don't correlate with other data

columns_to_drop = [
    'Min_Raindetection','Avg_Raindetection', 'Anemometer1_Offset', 'Anemometer1_CorrOffset',
    'Anemometer2_Offset', 'Anemometer2_CorrOffset','AirRessureSensorZeroOffset', 
    'Anemometer1_CorrOffset', 'Anemometer2_CorrGain', 'DistanceAirPress', 'Anemometer1_CorrGain',
    'Anemometer2_Freq', 'Anemometer1_Freq', 'Anemometer2_CorrOffset'
    
]

#Dropping the columns from the DataFrame
df_merged = df_merged.drop(columns=columns_to_drop)

#Selecting numeric features for correlation analysis
numeric_features = df_merged.select_dtypes(include=[np.number])

correlation_matrix = numeric_features.corr()

#Plotting a new heatmap of the correlation matrix
plt.figure(figsize=(20, 18))  
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Features', fontsize=20)
plt.xticks(fontsize=10)  
plt.yticks(fontsize=10) 
plt.show()


# In[29]:


#Saving df_merged
df_merged.to_csv('Pathname to save df_merged.csv', index=False)


# In[ ]:




