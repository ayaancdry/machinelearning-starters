# %%
# Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from dataloader.py, import the dataset df
from dataloader import df

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.describe()

# %%
df.shape

# %%
# Check for missing values
df.isnull().sum()

# %% [markdown]
# We can infer that there are no missing/NaN values in the dataset

# %%
df

# %%
# Looking at the dataset, we see the categorical features and the continuous features. 
categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
continuous_features = ["Age", "Height", "Weight", "CH2O","FAF", "TUE"]
target_variable = ['NObeyesdad']

# %% [markdown]
# # Data Preprocessing and Analysis Workflow
# In this workflow, we will fllow a systematic approach to understand and prepare our dataset for modelling. The key steps involved are as follows : 
# 
# 1. **Encoding the categorical features** : 
# - Encoding the categorical features with string values before splitting ensures consistency in the feature space across training and test datasets.
# 2. **Exploratory Data Analysis (EDA)** : 
# - Conduct comprehensive EDA to uncover insigihts about the data
# - Generate various visualizations to get a better understanding of feature distributions, relationships and potential outliers
# 3. **Cleaning the data** : 
# - Removing the outliers in continuous features 
# 4. **Train-Test Split** : 
# - Split the dataset into training and testing sets to evaluate model performance
# - This ensures that the model is trained on one portion of the data and validated on an unseen portion. 
# 5. **Scaling the continuous features** : 
# - Apply `StandardScaler` to the continuous features to standardize their values.
# - This transformation will center the data around a mean of `0` and scale it to have a standard deviation of `1`

# %% [markdown]
# ## Encoding

# %%
# LabelEncode the Categorical Features with dtype = "Object"

from sklearn.preprocessing import LabelEncoder

features_dtype_object = df.select_dtypes(include=[object]).columns
label_encoder = LabelEncoder()
for col in features_dtype_object:
    df[col] = label_encoder.fit_transform(df[col])

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Correlation Matrix

# %%
plt.figure(figsize=(16,8))
correlation = df.corr()
sns.heatmap(abs(correlation), linewidths=1, annot=True)
plt.show()

# %% [markdown]
# ### Pair Plots

# %%
sns.pairplot(df, hue='NObeyesdad')
plt.show()

# %% [markdown]
# ### Histplots for the Continuous Features
# 

# %%
# continuous_features = ["Age", "Height", "Weight", "CH2O","FAF", "TUE"]
fig, ax = plt.subplots(3, 2, figsize = (10,12))

sns.histplot(df['Age'], bins=20, ax=ax[0,0], kde=True)
ax[0,0].set_title('Distribution of Age')

sns.histplot(df['Height'], bins=20, ax=ax[0,1], kde=True)
ax[0,1].set_title('Distribution of Height')

sns.histplot(df['Weight'], bins=20, ax=ax[1,0], kde=True)
ax[1,0].set_title('Distribution of Weight')

sns.histplot(df['CH2O'], bins=20, ax=ax[1,1], kde=True)
ax[1,1].set_title('Distribution of CH2O')

sns.histplot(df['FAF'], bins=20, ax=ax[2,0], kde=True)
ax[2,0].set_title('Distribution of FAF')

sns.histplot(df['TUE'], bins=20, ax=ax[2,1], kde=True)
ax[2,1].set_title('Distribution of TUE')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Distribution of the Continuous Features with respect to the Target Variable

# %%
# continuous_features = ["Age", "Height", "Weight", "CH2O","FAF", "TUE"]
fig, axes = plt.subplots(3, 2, figsize=(12,12))
fig.tight_layout(pad=4.0)

axes[0,0].set_title("Distribution of Age by Target Variable")
sns.kdeplot(data=df, x='Age', hue='NObeyesdad', ax=axes[0,0])

axes[0,1].set_title("Distribution of Height by Target Variable")
sns.kdeplot(data=df, x='Height', hue='NObeyesdad', ax=axes[0,1])

axes[1,0].set_title("Distribution of Weight by Target Variable")
sns.kdeplot(data=df, x='Weight', hue='NObeyesdad', ax=axes[1,0])

axes[1,1].set_title("Distribution of CH2O by Target Variable")
sns.kdeplot(data=df, x='CH2O', hue='NObeyesdad', ax=axes[1,1])

axes[2,0].set_title("Distribution of FAF by Target Variable")
sns.kdeplot(data=df, x='FAF', hue='NObeyesdad', ax=axes[2,0])

axes[2,1].set_title("Distribution of TUE by Target Variable")
sns.kdeplot(data=df, x='TUE', hue='NObeyesdad', ax=axes[2,1])

for ax in axes[:2, :2].flat:
    ax.set_xlabel("")
    ax.set_ylabel("Density")

plt.show()



# %% [markdown]
# ### Box Plots

# %%
# Box Plots will be plotted for the continuous features to detect outliers
# continuous_features = ["Age", "Height", "Weight", "CH2O","FAF", "TUE"]

plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
fig = df.boxplot(column='Age')
fig.set_title('Age')
fig.set_ylabel('Age')

plt.subplot(3,2,2)
fig = df.boxplot(column='Height')
fig.set_title('Height')
fig.set_ylabel('Height')

plt.subplot(3,2,3)
fig = df.boxplot(column='Weight')
fig.set_title('Weight')
fig.set_ylabel('Weight')

plt.subplot(3,2,4)
fig = df.boxplot(column='CH2O')
fig.set_title('CH2O')
fig.set_ylabel('CH2O')

plt.subplot(3,2,5)
fig = df.boxplot(column='FAF')
fig.set_title('FAF')
fig.set_ylabel('FAF')

plt.subplot(3,2,6)
fig = df.boxplot(column='TUE')
fig.set_title('TUE')
fig.set_ylabel('TUE')

plt.tight_layout()
plt.show()

# %%
# Remove Outlier
# Split
# Scaling

# See the project iml

# %% [markdown]
# We can see there are potential outliers under `Age` feature. We will remove them using IQR method.

# %% [markdown]
# ## Cleaning the data

# %%
# Removing outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Filter values within the IQR range
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

# %%
df = remove_outliers_iqr(df, 'Age')

# %%
# Plot the boxplot for Age to check
fig = df.boxplot(column='Age')
fig.set_title('Age')
fig.set_ylabel('Age')

plt.show()

# %%
# Check the updated dataset's size
df.shape

# %% [markdown]
# ## Train-Test Split

# %%
from sklearn.model_selection import train_test_split

# Separate the features and target
X = df.drop(columns='NObeyesdad')
y = df[target_variable]

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# %%
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# %% [markdown]
# ## Scaling the Continuous Features

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# %%
# Scale in the training set
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])

# Scale in the test set
X_test[continuous_features] = scaler.transform(X_test[continuous_features])

# %%
X_train

# %%
X_test


