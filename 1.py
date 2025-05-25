import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("D:\portfolio\waze\waze_dataset.csv")
df.head(10)
print(df.head(10)) 
df.info()


# Isolate rows with null values
null_df = df[df['label'].isnull()]
# Display summary stats of rows with null values
null_df.describe()

# Isolate rows without null values
not_null_df = df[~df['label'].isnull()]
# Display summary stats of rows without null values
not_null_df.describe()

# Get count of null values by device
null_df['device'].value_counts()

# Calculate % of iPhone nulls and Android nulls
null_df['device'].value_counts(normalize=True)

# Calculate % of iPhone users and Android users in full dataset
df['device'].value_counts(normalize=True)

# Calculate counts of churned vs. retained
print(df['label'].value_counts())
print()
print(df['label'].value_counts(normalize=True))

# Calculate median values of all columns for churned and retained users
df.groupby('label').median(numeric_only=True)

# Add a column to df called `km_per_drive`
df['km_per_drive'] = df['driven_km_drives'] / df['drives']

# Group by `label`, calculate the median, and isolate for km per drive
median_km_per_drive = df.groupby('label').median(numeric_only=True)[['km_per_drive']]
median_km_per_drive

# Add a column to df called `km_per_driving_day`
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# Group by `label`, calculate the median, and isolate for km per driving day
median_km_per_driving_day = df.groupby('label').median(numeric_only=True)[['km_per_driving_day']]
median_km_per_driving_day

# Add a column to df called `drives_per_driving_day`
df['drives_per_driving_day'] = df['drives'] / df['driving_days']

# Group by `label`, calculate the median, and isolate for drives per driving day
median_drives_per_driving_day = df.groupby('label').median(numeric_only=True)[['drives_per_driving_day']]
median_drives_per_driving_day

# For each label, calculate the number of Android users and iPhone users
df.groupby(['label', 'device']).size()

# For each label, calculate the percentage of Android users and iPhone users
df.groupby('label')['device'].value_counts(normalize=True)

# Preprocess data
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['drives_per_driving_day'] = df['drives'] / df['driving_days']

# Visualization 1: Distribution of km_per_drive
sns.histplot(df['km_per_drive'], bins=30, kde=True)
plt.title('Distribution of Kilometers per Drive')
plt.xlabel('Kilometers per Drive')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Boxplot of km_per_driving_day by label SALIO MALA
sns.boxplot(x='label', y='km_per_driving_day', data=df)
plt.title('Kilometers per Driving Day by Label')
plt.xlabel('Label')
plt.ylabel('Kilometers per Driving Day')
plt.show()

# Visualization 3: Bar plot of median km_per_drive by label
median_km_per_drive = df.groupby('label')['km_per_drive'].median().reset_index()
sns.barplot(x='label', y='km_per_drive', data=median_km_per_drive)
plt.title('Median Kilometers per Drive by Label')
plt.xlabel('Label')
plt.ylabel('Median Kilometers per Drive')
plt.show()

# Visualization 4: Stacked bar chart of device distribution by label
device_distribution = df.groupby(['label', 'device']).size().unstack()
device_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Device Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.legend(title='Device')
plt.show()

# Visualization 5: Pie chart of Android vs iPhone users by label
device_percentage = df.groupby('label')['device'].value_counts(normalize=True).unstack() * 100

for label in device_percentage.index:
    plt.figure(figsize=(8, 8))
    device_percentage.loc[label].plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=device_percentage.columns)
    plt.title(f'Percentage of Android vs iPhone Users for Label {label}')
    plt.ylabel('')  # Remove y-axis label for better aesthetics
    plt.show()


# Visualization 6: Scatter plot of drives_per_driving_day vs km_per_driving_day
sns.scatterplot(x='drives_per_driving_day', y='km_per_driving_day', hue='label', data=df)
plt.title('Drives per Driving Day vs Kilometers per Driving Day')
plt.xlabel('Drives per Driving Day')
plt.ylabel('Kilometers per Driving Day')
plt.show()

# Visualization 7: Count plot of labels
sns.countplot(x='label', data=df)
plt.title('Count of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

##############################################

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['sessions'], fliersize=1)
plt.title('sessions box plot');
plt.show()
 

# Histogram
plt.figure(figsize=(5,3))
sns.histplot(x=df['sessions'])
median = df['sessions'].median()
plt.axvline(median, color='red', linestyle='--')
plt.text(75,1200, 'median=56.0', color='red')
plt.title('sessions box plot');
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['drives'], fliersize=1)
plt.title('drives box plot');
plt.show()

# Helper function to plot histograms based on the
# format of the `sessions` histogram
def histogrammer(column_str, median_text=True, **kwargs):    # **kwargs = any keyword arguments
                                                             # from the sns.histplot() function
    median=round(df[column_str].median(), 1)
    plt.figure(figsize=(5,3))
    ax = sns.histplot(x=df[column_str], **kwargs)            # Plot the histogram
    plt.axvline(median, color='red', linestyle='--')         # Plot the median line
    if median_text==True:                                    # Add median text unless set to False
        ax.text(0.25, 0.85, f'median={median}', color='red',
            ha='left', va='top', transform=ax.transAxes)
    else:
        print('Median:', median)
    plt.title(f'{column_str} histogram');

# Histogram
histogrammer('drives')
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['total_sessions'], fliersize=1)
plt.title('total_sessions box plot');
plt.show()


# Histogram
histogrammer('total_sessions')
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=1)
plt.title('n_days_after_onboarding box plot');
plt.show()

# Histogram
histogrammer('n_days_after_onboarding', median_text=False)
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driven_km_drives'], fliersize=1)
plt.title('driven_km_drives box plot');
plt.show()

# Histogram
histogrammer('driven_km_drives')
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['duration_minutes_drives'], fliersize=1)
plt.title('duration_minutes_drives box plot');
plt.show()

# Histogram
histogrammer('duration_minutes_drives')
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['activity_days'], fliersize=1)
plt.title('activity_days box plot');
plt.show()

# Histogram
histogrammer('activity_days', median_text=False, discrete=True)
plt.show()

# Box plot
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driving_days'], fliersize=1)
plt.title('driving_days box plot');
plt.show()


# Histogram
histogrammer('driving_days', median_text=False, discrete=True)
plt.show()

# Pie chart
fig = plt.figure(figsize=(3,3))
data=df['device'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Users by device');
plt.show()

# Pie chart
fig = plt.figure(figsize=(3,3))
data=df['label'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Count of retained vs. churned');
plt.show()

# Histogram
plt.figure(figsize=(12,4))
label=['driving days', 'activity days']
plt.hist([df['driving_days'], df['activity_days']],
         bins=range(0,33),
         label=label)
plt.xlabel('days')
plt.ylabel('count')
plt.legend()
plt.title('driving_days vs. activity_days');
plt.show()

print(df['driving_days'].max())
print(df['activity_days'].max())

# Scatter plot
sns.scatterplot(data=df, x='driving_days', y='activity_days')
plt.title('driving_days vs. activity_days')
plt.plot([0,31], [0,31], color='red', linestyle='--');
plt.show()

# Histogram
plt.figure(figsize=(5,4))
sns.histplot(data=df,
             x='device',
             hue='label',
             multiple='dodge',
             shrink=0.9
             )
plt.title('Retention by device histogram');
plt.show()

# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Call `describe()` on the new column
df['km_per_driving_day'].describe()

# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# 2. Confirm that it worked
df['km_per_driving_day'].describe()

# Histogram
plt.figure(figsize=(12,5))
sns.histplot(data=df,
             x='km_per_driving_day',
             bins=range(0,1201,20),
             hue='label',
             multiple='fill')
plt.ylabel('%', rotation=0)
plt.title('Churn rate by mean km per driving day');
plt.show()

# Histogram
plt.figure(figsize=(12,5))
sns.histplot(data=df,
             x='driving_days',
             bins=range(1,32),
             hue='label',
             multiple='fill',
             discrete=True)
plt.ylabel('%', rotation=0)
plt.title('Churn rate per driving day');
plt.show()

df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

df['percent_sessions_in_last_month'].median()

# Histogram
histogrammer('percent_sessions_in_last_month',
             hue=df['label'],
             multiple='layer',
             median_text=False)

df['n_days_after_onboarding'].median()
plt.show()

# Histogram
data = df.loc[df['percent_sessions_in_last_month']>=0.4]
plt.figure(figsize=(5,3))
sns.histplot(x=data['n_days_after_onboarding'])
plt.title('Num. days after onboarding for users with >=40% sessions in last month');
plt.show()


def outlier_imputer(column_name, percentile):
    # Calculate threshold
    threshold = df[column_name].quantile(percentile)
    # Impute threshold for values > than threshold
    df.loc[df[column_name] > threshold, column_name] = threshold

    print('{:>25} | percentile: {} | threshold: {}'.format(column_name, percentile, threshold))
plt.show()


for column in ['sessions', 'drives', 'total_sessions',
               'driven_km_drives', 'duration_minutes_drives']:
               outlier_imputer(column, 0.95)



