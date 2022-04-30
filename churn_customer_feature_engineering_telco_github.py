##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem: It is requested to develop a machine learning model that can predict customers who will leave the company.
# Before developing the model, you are expected to perform the necessary data analysis and feature engineering steps.

# Telco customer churn data, providing home phone and Internet services to 7043 customers in California in the third quarter
# contains information about a fictitious telecom company. It includes which customers have left, stayed or signed up for the service.

#21 Variable 7043 Observation

# CustomerId : Customer ID
# Gender : Gender
# SeniorCitizen : Whether the client is old (1, 0)
# Partner : Whether the client has a partner (Yes, No) ? whether to be married
# Dependents : Whether the customer has dependents (Yes, No) (Child, mother, father, grandmother)
# tenure : The number of months the customer has stayed with the company
# PhoneService : Whether the customer has phone service (Yes, No)
# MultipleLines : Whether the customer has more than one line (Yes, No, No Telephone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup : Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport : Whether the customer receives technical support (Yes, No, no Internet service)
# StreamingTV : Indicates whether the customer is broadcasting TV (Yes, No, no Internet service) Indicates whether the customer uses the Internet service to stream television programs from a third-party provider
# StreamingMovies : Whether the customer is streaming movies (Yes, No, no Internet service) Indicates whether the customer is using the Internet service to stream movies from a third-party provider
# Contract : Contract duration of the client (Month to month, One year, Two years)
# PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
# PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges : The amount charged monthly from the customer
# TotalCharges : The total amount charged from the customer
# Churn : Whether the customer uses (Yes or No) - Customers who left in the last month or quarter


# Each row represents a unique customer.
# Variables contain information about customer service, account and demographic data.
# Services that customers sign up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they have been a customer, contract, payment method, paperless invoicing, monthly fees and total fees
# Demographic information about customers - gender, age range and whether they have partners and dependents


# TASK 1: DISCOVERY DATA ANALYSIS
           # Step 1: Examine the overall picture.
           # Step 2: Capture the numeric and categorical variables.
           # Step 3: Analyze the numerical and categorical variables.
           # Step 4: Perform target variable analysis. (The mean of the target variable according to the categorical variables, the mean of the numeric variables according to the target variable)
           # Step 5: Analyze outliers.
           # Step 6: Perform missing observation analysis.
           # Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
           # Step 1: Take necessary actions for missing and outlier values.
           # you can apply operations.
           # Step 2: Create new variables.
           # Step 3: Perform the encoding operations.
           # Step 4: Standardize for numeric variables.
           # Step 5: Create the model.


# Required Library and Functions
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("week_6/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df["Churn"].value_counts()

# TotalCharges must be a numeric variable
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
df["Churn"].mean()

df[(df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic") & (df["SeniorCitizen"] == 1)]["Churn"].mean()
df[(df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic") & (df["SeniorCitizen"] == 1)]["Churn"].shape

df[(df["Contract"] == "Two year") & (df["InternetService"] == "DSL") & (df["SeniorCitizen"] == 1)]["Churn"].mean()

df["SeniorCitizen"].value_counts()
##################################
# TASK 1: DISCOVERY DATA ANALYSIS
##################################

##################################
# GENERAL PICTURE
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



##################################
# CAPTURE OF NUMERICAL AND CATEGORICAL VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold value for numeric but categorical variables
        car_th: int, optional
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



##################################
# ANALYSIS OF CATEGORY VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

# About half of the customers in our dataset are men and the other half are women.
# About 50% of clients have a partner (married)
# Only 30% of total customers have dependents.
# 90% of customers receive telephone service.
# 53% of 90% of people who get phone service have more than one line
# There is a 21% segment that does not have an Internet service provider
# 63% of customers with an Internet service provider do not have online security service
# 55% of customers with an Internet service provider do not have an online backup service
# 56% of customers with an Internet service provider do not have device protection
# 62% of customers with Internet service providers do not receive technical support
# 50% of customers with Internet service providers do not receive TV broadcasting service
# 50% of customers with Internet service providers do not receive movie streaming service
# Most of the customers are contracting month to month. There are close numbers of customers on 1-year and 2-year contracts.
# 60% of customers have paperless invoices
# About 26% of customers left the platform in the last month
# 16% of the dataset consists of older customers So most of the customers in the data are young


##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# When we look at Tenure, we see that 1-month customers are too many, followed by 70-month customers.
# It may have happened due to different contracts, let's look at the tenure of people with a monthly contract and the tenure of people with a 2-year contract.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# According to MonthyChargers, customers with monthly contracts may have higher average monthly payments.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()



##################################
# ANALYSIS OF NUMERICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# When we look at the Tenure and Churn relationship, we see that non-churn customers have been customers for longer
# When monthlycharges and Churn are examined, the average monthly payments of customers with churn are higher.
# There seems to be more loss when total wages are lower


##################################
# ANALYSIS OF CATEGORY VARIABLES ACCORDING TO TARGET
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


# The percentage of churn in men and women is almost equal
# Customers with partners and dependents have a lower churn rate
# No difference in PhoneService and MultipleLines
# Loss rate is much higher in Fiber Optic Internet Services
# High churn rate for customers without services such as No OnlineSecurity , OnlineBackup and TechSupport
# A larger percentage of Customers with a monthly subscription churn compared to Customers with a one- or two-year contract
# Those with paperless billing have a higher churn rate
# Customers with ElectronicCheck PaymentMethod tend to leave the platform more compared to other options
# Older customers have a higher percentage of churn

##################################
# CORRELATION
##################################

df[num_cols].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers appears to be highly correlated with monthly fees and tenure


##################################
# TASK 2: FEATURE ENGINEERING
##################################

##################################
# MISSING VALUE ANALYSIS
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(0, inplace=True)
df[df["TotalCharges"].isnull()]["tenure"]
df.isnull().sum()



##################################
# BASE MODEL SETUP
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# Auc: 0.7282



##################################
# Outlier Analysis
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)




##################################
# FEATURE EXTRACTION
##################################

# Creating annual categorical variable from Tenure variable

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

df["NEW_TENURE_YEAR"].value_counts()

# Specify 1 or 2 year contract customers as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "No") or (x["DeviceProtection"] != "No") or (x["TechSupport"] != "No") else 0, axis=1)

# Customers with monthly contracts and young
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Total number of services received by the person

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# People who buy any streaming service
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# average monthly payment
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] +0.1)

# Current Price increase relative to average price
df["NEW_Increase"] = df["NEW_AVG_Charges"] / (df["MonthlyCharges"] + 1)

# Charge per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


##################################
# ENCODING
##################################

# Separation of variables according to their types

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding Process
# Update process of cat_cols list

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape
##################################
# MODELLING
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.8
# Recall: 0.66
# Precision: 0.51
# F1: 0.58
# Auc: 0.75

# Base Model
# # Accuracy: 0.7837
# # Recall: 0.6333
# # Precision: 0.4843
# # F1: 0.5489
# # Auc: 0.7282
df["Churn"].value_counts()
# Written for different catboots than in the lesson
def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')