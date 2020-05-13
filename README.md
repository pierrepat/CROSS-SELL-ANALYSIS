

**ASSIGNMENT 2 MACHINE LEARNING**  
**Building predictive models (train test split from SKLearn, TREES, KNN)**

*Pierre-Emmanuel Patrouillard*

GOAL: Know which customers will subscribe to Apprentice Chefs Cross-Sell service


```python
#DOWNLOADING PACKAGES
import pydotplus                     
import random                  as rand                        # random number gen
import pandas                  as pd                          # data science essentials
import matplotlib.pyplot       as plt                         # data visualization
import seaborn                 as sns                         # enhanced data viz
import numpy                   as np                          # mathematical essentials
import pandas                  as pd                          # data science essentials
import matplotlib.pyplot       as plt                         # data visualization
import statsmodels.formula.api as smf                         # smf
from sklearn.model_selection   import train_test_split        # train-test split
from sklearn.linear_model      import LogisticRegression      # logistic regression
from sklearn.linear_model      import LinearRegression        # linear regression
from sklearn.metrics           import confusion_matrix        # confusion matrix
from sklearn.metrics           import roc_auc_score           # auc score
from sklearn.neighbors         import KNeighborsClassifier    # KNN for classification
from sklearn.neighbors         import KNeighborsRegressor     # KNN for regression
from sklearn.preprocessing     import StandardScaler          # standard scaler
from sklearn.tree              import DecisionTreeClassifier  # classification trees
from sklearn.tree              import export_graphviz         # exports graphics
from sklearn.externals.six     import StringIO                # saves objects in memory
from IPython.display           import Image                   # displays on frontend
from sklearn.model_selection   import GridSearchCV            # hyperparameter tuning
from sklearn.metrics           import make_scorer             # customizable scorer
from sklearn.ensemble          import RandomForestClassifier  # random forest
from sklearn.ensemble          import GradientBoostingClassifier # gbm

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#File name
file = 'Apprentice_Chef_Dataset.xlsx'

#Reading the excel file into python
original_df = pd.read_excel(file)
```


```python
original_df.head(5)
```

# CHECKING FOR NULL VALUES / SAVING "CLEANED" DATA SET##


```python
#Checking for null values
original_df.isnull().sum()
```


```python
# creating a dropped dataset 
original_df = original_df.dropna()


original_df.to_excel("original_df_no_na.xlsx", index = False)


original_df.isnull().sum()
```

## *SPLITTING EMAILS/ CREATING COLUMNS FOR THEM*


```python
#Splitting emails 

original_df = pd.read_excel('original_df_no_na.xlsx')

#placeholder list
placeholder_lst = []

#loop to group by domain type
for index, col in original_df.iterrows():
    
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    placeholder_lst.append(split_email)

email_df = pd.DataFrame(placeholder_lst)
email_df
 
```


```python
#concatenating with original df

email_df.columns = ['0', 'email_domain']

#concatenating personal email with df
original_df = pd.concat([original_df, email_df['email_domain']],
                       axis = 1)


#seeing which email domain is most common
original_df['email_domain'].value_counts()
```


```python
# email domain types
personal_email = ['@gmail.com', '@yahoo.com', '@protonmail.com']
junk_email  = ['@me.com', '@aol.com', '@hotmail.com',
                       '@live.com', '@msn.com', '@passport.com']
prof_email  = ['@mmm.com', '@amex.com', '@apple.com',
                       '@boeing.com', '@caterpillar.com', '@chevron.com',
                       '@cisco.com', '@cocacola.com', '@disney.com', 
                       '@dupont.com', '@exxon.com', '@ge.org', 
                       '@goldmansacs.com', '@homedepot.com', '@ibm.com', 
                       '@intel.com', '@jnj.com', '@jpmorgan.com', 
                       '@mcdonalds.com', '@merck.com', '@microsoft.com', 
                       '@nike.com', '@pfizer.com', '@pg.com', 
                       '@travelers.com', '@unitedtech.com', '@unitedhealth.com',
                       '@verizon.com', '@visa.com', '@walmart.com']

placeholder_lst = []

#loop to group by email domain 
for domain in original_df['email_domain']:
    
    if '@' + domain in personal_email:
        placeholder_lst.append('personal')
        
    elif '@' + domain in junk_email:
        placeholder_lst.append('junk')
        
    elif '@' + domain in prof_email:
        placeholder_lst.append('professional')
        
    else:
        print('Unkown')
        
#concat w original df
original_df['domain_group'] = pd.Series(placeholder_lst)

#checking results 
original_df['domain_group'].value_counts()
```


```python
#dummies for domain group
dummies = pd.get_dummies(original_df['domain_group'])

# concatenating personal_email_domain with friends DataFrame
original_df = pd.concat([original_df, dummies],
                     axis = 1)

# converting the dummies to int64
original_df['junk'] = np.int64(original_df['junk'])
original_df['personal'] = np.int64(original_df['personal'])
original_df['professional'] = np.int64(original_df['professional'])

# double checking to make sure dummies are in place
original_df.head()
```


```python
df_corr = original_df.corr()
df_corr

df_corr['CROSS_SELL_SUCCESS'].sort_values(ascending = False)

#insight: customers using professional emails are more likely cross sell success
#         customers who use junk emails are less likely to cross sell success

```

## *BOXPLOTS CROSS SELL VS OTHERS*


```python
#Defining function for categorical boxplots

def categorical_boxplots(response, cat_var, data):
    """This function is for categorical variables
    
    Parameters
    ----------
    response : str, response variable
    cat_var : str, categorical variable
    data : DataFrame of the response and categorical variables
    """
    
    data.boxplot(column = response,
                    by = cat_var,
                    vert = False,
                    patch_artist = False,
                    meanline = True,
                    showmeans = True)
    plt.suptitle("")
    plt.show()
    
#Creating boxplots for CROSS_SELL_SUCCESS vs rest
categorical_boxplots(response = 'FOLLOWED_RECOMMENDATIONS_PCT',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'professional',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'CANCELLATIONS_BEFORE_NOON',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'MOBILE_NUMBER',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'REFRIGERATED_LOCKER',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'PACKAGE_LOCKER',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'MOBILE_LOGINS',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'CONTACTS_W_CUSTOMER_SERVICE',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'MASTER_CLASSES_ATTENDED',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'personal',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'MEDIAN_MEAL_RATING',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'LATE_DELIVERIES',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'AVG_PREP_VID_TIME',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'LARGEST_ORDER_SIZE',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'EARLY_DELIVERIES',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'AVG_TIME_PER_SITE_VISIT',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'TOTAL_MEALS_ORDERED',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'REVENUE',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'TOTAL_PHOTOS_VIEWED',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'UNIQUE_MEALS_PURCH',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'PRODUCT_CATEGORIES_VIEWED',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'WEEKLY_PLAN',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'AVG_CLICKS_PER_VISIT',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'PC_LOGINS',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'CANCELLATIONS_AFTER_NOON',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)

categorical_boxplots(response = 'junk',
                        cat_var = 'CROSS_SELL_SUCCESS',
                        data = original_df)               
```

## *SETTING THRESHOLDS/ ADDING THEM TO DF*


```python
#Setting thresholds from A1 assigment 
TOTAL_MEALS_ORDERED_HIGH           = 280
UNIQUE_MEALS_PURCH_HIGH            = 9
CONTACTS_W_CUSTOMER_SERVICE_HIGH   = 10
AVG_TIME_PER_SITE_VISIT_HIGH       = 250
CANCELLATIONS_BEFORE_NOON_HIGH     = 7
CANCELLATIONS_AFTER_NOON_HIGH      = 1.5
MOBILE_LOGINS_HIGH                 = 6.5
MOBILE_LOGINS_lo                 = 4.5
PC_LOGINS_HIGH                     = 2.5
PC_LOGINS_lo                     = 0.5
WEEKLY_PLAN_HIGH                   = 20
EARLY_DELIVERIES_HIGH              = 5
LATE_DELIVERIES_HIGH               = 10
AVG_PREP_VID_TIME_HIGH             = 300
LARGEST_ORDER_SIZE_HIGH            = 6
MASTER_CLASSES_ATTENDED_HIGH       = 2
MEDIAN_MEAL_RATING_HIGH            = 4.5 
AVG_CLICKS_PER_VISIT_lo          = 7.5
TOTAL_PHOTOS_VIEWED_HIGH           = 470
REVENUE_HIGH                       = 5700
```


```python
#Creating new columns for those thresholds

#Unique Meals purchased
original_df['TOTAL_MEALS_ORDERED_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'TOTAL_MEALS_ORDERED_HIGH'][original_df['TOTAL_MEALS_ORDERED_HIGH'] > TOTAL_MEALS_ORDERED_HIGH]

original_df['TOTAL_MEALS_ORDERED_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)
#CONTACTS_W_CUSTOMER_SERVICE_HIGH
original_df['CONTACTS_W_CUSTOMER_SERVICE_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'CONTACTS_W_CUSTOMER_SERVICE_HIGH'][original_df['CONTACTS_W_CUSTOMER_SERVICE_HIGH'] > CONTACTS_W_CUSTOMER_SERVICE_HIGH]

original_df['CONTACTS_W_CUSTOMER_SERVICE_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#CANCELLATIONS_BEFORE_NOON_HIGH
original_df['CANCELLATIONS_BEFORE_NOON_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'CANCELLATIONS_BEFORE_NOON_HIGH'][original_df['CANCELLATIONS_BEFORE_NOON_HIGH'] > CANCELLATIONS_BEFORE_NOON_HIGH]

original_df['CANCELLATIONS_BEFORE_NOON_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_AFTER_NOON
original_df['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_HIGH = original_df.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HIGH]

original_df['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_HIGH,
                                    value      = 1,
                                    inplace    = True)

# PC_LOGINS
original_df['out_PC_LOGINS'] = 0
condition_HIGH = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_HIGH]
condition_lo = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_lo]

original_df['out_PC_LOGINS'].replace(to_replace = condition_HIGH,
                                    value      = 1,
                                    inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# MOBILE_LOGINS
original_df['out_MOBILE_LOGINS'] = 0
condition_HIGH = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_HIGH]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_HIGH,
                                    value      = 1,
                                    inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#Unique Meals purchased
original_df['UNIQUE_MEALS_PURCHASED_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'UNIQUE_MEALS_PURCHASED_HIGH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HIGH]

original_df['UNIQUE_MEALS_PURCHASED_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#photos viewed
original_df['TOTAL_PHOTOS_VIEWED_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'TOTAL_PHOTOS_VIEWED_HIGH'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_HIGH]

original_df['TOTAL_PHOTOS_VIEWED_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#master classes attended
original_df['MASTER_CLASSES_ATTENDED_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'MASTER_CLASSES_ATTENDED_HIGH'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HIGH]

original_df['MASTER_CLASSES_ATTENDED_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#avg prep time
original_df['AVG_PREPARATION_TIME_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'AVG_PREPARATION_TIME_HIGH'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HIGH]

original_df['AVG_PREPARATION_TIME_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)
# EARLY_DELIVERIES
original_df['out_EARLY_DELIVERIES'] = 0
condition_HIGH = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HIGH]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_HIGH,
                                 value      = 1,
                                 inplace    = True)

#late deliveries
original_df['LATE_DELIVERIES_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'LATE_DELIVERIES_HIGH'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_HIGH]

original_df['LATE_DELIVERIES_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#weekly plan deliveries
original_df['WEEKLY_PLAN_DELIVERIES_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'WEEKLY_PLAN_DELIVERIES_HIGH'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_HIGH]

original_df['WEEKLY_PLAN_DELIVERIES_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

#avg time per visit 
original_df['AVG_TIME_PER_VISIT_HIGH'] = 0
condition_HIGH = original_df.loc[0:,'AVG_TIME_PER_VISIT_HIGH'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HIGH]

original_df['AVG_TIME_PER_VISIT_HIGH'].replace(to_replace = condition_HIGH,
                                value      = 1,
                                inplace    = True)

# LARGEST_ORDER_SIZE
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_HIGH = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HIGH]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_HIGH,
                                 value      = 1,
                                 inplace    = True)

# MEDIAN_MEAL_RATING
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_HIGH = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_HIGH]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_HIGH,
                                 value      = 1,
                                 inplace    = True)

# AVG_CLICKS_PER_VISIT
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                 value      = 1,
                                 inplace    = True)

original_df.tail()
```


```python

#Setting the x var that have some sort of significance based on boxplots and corr
x_var = ['REVENUE','TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE', 
         'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 
         'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS', 'PC_LOGINS', 'WEEKLY_PLAN',
        'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT',
        'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT',
        'TOTAL_PHOTOS_VIEWED', 'junk', 'professional', 'TOTAL_MEALS_ORDERED_HIGH', 'CONTACTS_W_CUSTOMER_SERVICE_HIGH',
        'CANCELLATIONS_BEFORE_NOON_HIGH', 'out_CANCELLATIONS_AFTER_NOON', 'out_PC_LOGINS', 'out_MOBILE_LOGINS', 'UNIQUE_MEALS_PURCHASED_HIGH',
        'TOTAL_PHOTOS_VIEWED_HIGH', 'MASTER_CLASSES_ATTENDED_HIGH', 'AVG_PREPARATION_TIME_HIGH', 'out_EARLY_DELIVERIES', 'LATE_DELIVERIES_HIGH', 
         'WEEKLY_PLAN_DELIVERIES_HIGH', 'AVG_TIME_PER_VISIT_HIGH', 'out_LARGEST_ORDER_SIZE', 'out_MEDIAN_MEAL_RATING', 'out_AVG_CLICKS_PER_VISIT']
   
```


```python
original_df = original_df.drop(labels=['NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME', 'email_domain', 'domain_group'], axis=1)

# Preparing a DataFrame to scale the data and use for train/test split
original_df_data   = original_df.drop(labels = 'CROSS_SELL_SUCCESS', axis = 1)


# Preparing the target variable
original_df_target = original_df.loc[:, 'CROSS_SELL_SUCCESS']

original_df.info()
```

## *SCALING AND MODEL BUILDING*


```python
scaler = StandardScaler()

temp_df = original_df.drop('CROSS_SELL_SUCCESS', axis = 1)

scaler.fit(temp_df)

X_scaled = scaler.transform(temp_df)

temp_df = pd.DataFrame(X_scaled)

temp_df = pd.concat([temp_df, original_df['CROSS_SELL_SUCCESS']], axis = 1)

temp_df.columns = ['REVENUE', 'TOTAL_MEALS_ORDERED',
                   'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
                   'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER',
                   'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES',
                   'MOBILE_LOGINS', 'PC_LOGINS','WEEKLY_PLAN', 'EARLY_DELIVERIES',
                   'LATE_DELIVERIES', 'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER',
                   'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE',
                   'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT',
                   'TOTAL_PHOTOS_VIEWED', 'junk', 'personal', 'professional',
                   'TOTAL_MEALS_ORDERED_HIGH', 'CONTACTS_W_CUSTOMER_SERVICE_HIGH',
                   'CANCELLATIONS_BEFORE_NOON_HIGH', 'out_CANCELLATIONS_AFTER_NOON',
                   'out_PC_LOGINS', 'out_MOBILE_LOGINS','UNIQUE_MEALS_PURCHASED_HIGH',
                   'TOTAL_PHOTOS_VIEWED_HIGH', 'MASTER_CLASSES_ATTENDED_HIGH',
                   'AVG_PREPARATION_TIME_HIGH', 'out_EARLY_DELIVERIES',
                   'LATE_DELIVERIES_HIGH', 'WEEKLY_PLAN_DELIVERIES_HIGH',
                   'AVG_TIME_PER_VISIT_HIGH', 'out_LARGEST_ORDER_SIZE',
                   'out_MEDIAN_MEAL_RATING', 'out_AVG_CLICKS_PER_VISIT', 'CROSS_SELL_SUCCESS']

original_df = temp_df.copy()

original_df.describe().round(2)
```


```python
# formatting each explanatory variable for statsmodels
for val in original_df:
    print(f"{val} +")
```


```python
#building a full model

#blueprinting model type
lm_full = smf.logit(formula = """CROSS_SELL_SUCCESS ~ REVENUE +
TOTAL_MEALS_ORDERED +
UNIQUE_MEALS_PURCH +
CONTACTS_W_CUSTOMER_SERVICE +
PRODUCT_CATEGORIES_VIEWED +
AVG_TIME_PER_SITE_VISIT +
MOBILE_NUMBER +
CANCELLATIONS_BEFORE_NOON +
CANCELLATIONS_AFTER_NOON +
TASTES_AND_PREFERENCES +
MOBILE_LOGINS +
PC_LOGINS +
WEEKLY_PLAN +
EARLY_DELIVERIES +
LATE_DELIVERIES +
PACKAGE_LOCKER +
REFRIGERATED_LOCKER +
FOLLOWED_RECOMMENDATIONS_PCT +
AVG_PREP_VID_TIME +
LARGEST_ORDER_SIZE +
MASTER_CLASSES_ATTENDED +
MEDIAN_MEAL_RATING +
AVG_CLICKS_PER_VISIT +
TOTAL_PHOTOS_VIEWED +
junk +
professional +
out_CANCELLATIONS_AFTER_NOON +
out_PC_LOGINS +
out_MOBILE_LOGINS +
UNIQUE_MEALS_PURCHASED_HIGH +
TOTAL_PHOTOS_VIEWED_HIGH +
MASTER_CLASSES_ATTENDED_HIGH +
AVG_PREPARATION_TIME_HIGH +
out_EARLY_DELIVERIES +
LATE_DELIVERIES_HIGH +
WEEKLY_PLAN_DELIVERIES_HIGH +
AVG_TIME_PER_VISIT_HIGH +
out_LARGEST_ORDER_SIZE +
out_MEDIAN_MEAL_RATING +
out_AVG_CLICKS_PER_VISIT 
""", 
                    data = original_df)

#telling python to run data through blueprint
results_full = lm_full.fit()

#printing the results
results_full.summary()
```

Final Regression Model after deleting P value variables > 0.1 


```python
lm_full = smf.logit(formula = """CROSS_SELL_SUCCESS ~ REVENUE +
MOBILE_NUMBER +
CANCELLATIONS_BEFORE_NOON +
TASTES_AND_PREFERENCES +
MOBILE_LOGINS +
PC_LOGINS +
REFRIGERATED_LOCKER +
FOLLOWED_RECOMMENDATIONS_PCT +
junk +
professional +
out_CANCELLATIONS_AFTER_NOON +
UNIQUE_MEALS_PURCHASED_HIGH +
TOTAL_PHOTOS_VIEWED_HIGH +
AVG_PREPARATION_TIME_HIGH +
WEEKLY_PLAN_DELIVERIES_HIGH 
""", 
                    data = original_df)

#telling python to run data through blueprint
results_full = lm_full.fit()

#printing the results
results_full.summary()
```

# *NEW DF WITH VARIABLES HAVING P VALUE < .1*


```python
df_data          = ['CANCELLATIONS_BEFORE_NOON',
                    'TASTES_AND_PREFERENCES', 
                    'MOBILE_LOGINS',
                    'PC_LOGINS',
                    'REFRIGERATED_LOCKER',
                    'FOLLOWED_RECOMMENDATIONS_PCT',
                    'junk',
                    'professional',
                    'out_CANCELLATIONS_AFTER_NOON',
                    'UNIQUE_MEALS_PURCHASED_HIGH',
                    'TOTAL_PHOTOS_VIEWED_HIGH', 
                    'AVG_PREPARATION_TIME_HIGH',
                    'WEEKLY_PLAN_DELIVERIES_HIGH']

df_target = ['CROSS_SELL_SUCCESS']
  

```


```python
logit_full = ['REVENUE',
                    'TOTAL_MEALS_ORDERED',
                    'UNIQUE_MEALS_PURCH',
                    'CONTACTS_W_CUSTOMER_SERVICE',
                    'PRODUCT_CATEGORIES_VIEWED',
                    'AVG_TIME_PER_SITE_VISIT',
                    'MOBILE_NUMBER',
                    'CANCELLATIONS_BEFORE_NOON',
                    'CANCELLATIONS_AFTER_NOON',
                    'TASTES_AND_PREFERENCES',
                    'MOBILE_LOGINS',
                    'PC_LOGINS',
                    'WEEKLY_PLAN',
                    'EARLY_DELIVERIES',
                    'LATE_DELIVERIES',
                    'PACKAGE_LOCKER',
                    'REFRIGERATED_LOCKER',
                    'FOLLOWED_RECOMMENDATIONS_PCT',
                    'AVG_PREP_VID_TIME',
                    'LARGEST_ORDER_SIZE',
                    'MASTER_CLASSES_ATTENDED',
                    'MEDIAN_MEAL_RATING',
                    'AVG_CLICKS_PER_VISIT',
                    'TOTAL_PHOTOS_VIEWED',
                    'junk',
                    'personal',
                    'professional',
                    'out_CANCELLATIONS_AFTER_NOON',
                    'out_PC_LOGINS',
                    'out_MOBILE_LOGINS', 
                    'UNIQUE_MEALS_PURCHASED_HIGH', 
                    'TOTAL_PHOTOS_VIEWED_HIGH',
                    'MASTER_CLASSES_ATTENDED_HIGH', 
                    'AVG_PREPARATION_TIME_HIGH', 
                    'out_EARLY_DELIVERIES', 
                    'LATE_DELIVERIES_HIGH', 
                    'WEEKLY_PLAN_DELIVERIES_HIGH', 
                    'AVG_TIME_PER_VISIT_HIGH', 
                    'out_LARGEST_ORDER_SIZE', 
                    'out_MEDIAN_MEAL_RATING', 
                    'out_AVG_CLICKS_PER_VISIT']
 
 # significant variables only
logit_sig =         ['REVENUE',
                    'MOBILE_NUMBER',
                    'CANCELLATIONS_BEFORE_NOON',
                    'TASTES_AND_PREFERENCES', 
                    'MOBILE_LOGINS',
                    'PC_LOGINS',
                    'REFRIGERATED_LOCKER',
                    'FOLLOWED_RECOMMENDATIONS_PCT',
                    'junk',
                    'professional',
                    'out_CANCELLATIONS_AFTER_NOON',
                    'UNIQUE_MEALS_PURCHASED_HIGH',
                    'TOTAL_PHOTOS_VIEWED_HIGH', 
                    'AVG_PREPARATION_TIME_HIGH',
                    'WEEKLY_PLAN_DELIVERIES_HIGH']
```

## *TRAIN TEST SPLIT/ REGRESSIONS/ KNN/ TREES*


```python
#original_df = original_df.drop(labels=['NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME', 'email_domain', 'domain_group'], axis=1)
# running train/test split again
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222)

original_train = pd.concat([X_train, y_train], axis = 1)
```


```python
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
```


```python
#We will create an empty list
model_performance = [['Model', 'Training Accuracy'
                      'Testing Accuracy', 'AUC Value']]

#Instantiating logistic regression model
logreg = LogisticRegression(solver       = 'lbfgs',
                            C            = 1,
                            random_state = 222)

#Fitting the training data
logreg_fit = logreg.fit(X_train, y_train)

#Predicting based on testing set
logreg_pred = logreg_fit.predict(X_test)

logreg_train_acc = logreg_fit.score(X_train, y_train).round(4)
logreg_test_acc  = logreg_fit.score(X_test, y_test).round(4)



logreg_auc       = roc_auc_score(y_true  = y_test,
                                 y_score = logreg_pred).round(4)

#Results
print('Training Accuracy:', logreg_fit.score(X_train, y_train).round(4))
print('Testing Accuracy:', logreg_fit.score(X_test, y_test).round(4))

model_performance.append(['Logistic Regression', 
                          logreg_train_acc,
                          logreg_test_acc,
                          logreg_auc])

for model in model_performance:
    print(model)
```


```python


#Making empty lists for accuracy sets
training_accuracy = []
test_accuracy     = []

#Visualising 1 - 21 neighbors
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    Clf = KNeighborsClassifier(n_neighbors = n_neighbors) 
    Clf.fit(X_train, y_train)

    training_accuracy.append(Clf.score(X_train, y_train)) 
    
    test_accuracy.append(Clf.score(X_test, y_test))       
    
#Visualisation 
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#Number of optimal number of neighbours
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print(f"""Optimal number of neighbors is {opt_neighbors}""")
```


```python

#KNN classification  
knn_opt = KNeighborsClassifier(n_neighbors = opt_neighbors)

#Fitting the data
knn_fit = knn_opt.fit(X_train, y_train)

#Predicting based on testing
knn_pred = knn_fit.predict(X_test)

#Recording results
print('Training Accuracy:', knn_fit.score(X_train, y_train).round(4))
print('Testing  Accuracy:', knn_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = knn_pred).round(4))
```


```python
full_tree        = DecisionTreeClassifier()

full_tree_fit    = full_tree.fit(X_train, y_train)

full_tree_pred   = full_tree.predict(X_test)
```


```python
print('Training Accuracy:',   full_tree_fit.score(X_train, y_train).round(4))
print('Testing Accuracy:',    full_tree_fit.score(X_test, y_test).round(4))
print('AUC Score:',           roc_auc_score(y_true = y_test,
                              y_score = full_tree_pred).round(4))
```


```python
#KNN Training and Testing
knn_train_acc = knn_fit.score(X_train, y_train).round(4)

knn_test_acc  = knn_fit.score(X_test, y_test).round(4)

knn_auc       = roc_auc_score(y_true  = y_test,
                              y_score = knn_pred).round(4)

def display_tree(tree, feature_df, height = 2300, width = 2300):

    dot_data = StringIO()

    
    export_graphviz(decision_tree      = tree,
                    out_file           = dot_data,
                    filled             = True,
                    rounded            = True,
                    special_characters = True,
                    feature_names      = feature_df.columns)


    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


    img = Image(graph.create_png(),
                height = height,
                width  = width)
    
    return img

def plot_feature_importances(model, train, export = False):
    
    n_features = X_train.shape[1]
    
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


model_performance.append(['KNN Classification',
                          knn_train_acc,
                          knn_test_acc,
                          knn_auc])

for model in model_performance:
    print(model)
    
display_tree(tree = full_tree_fit,
             feature_df = X_train)
```


```python
#Pruned tree visual
tree_pruned = DecisionTreeClassifier(max_depth = 4,
                                     min_samples_leaf = 20,
                                    random_state = 222)

tree_pruned_fit = tree_pruned.fit(X_train, y_train)

tree_pred = tree_pruned_fit.predict(X_test)

print('Training Accuracy:', tree_pruned_fit.score(X_train, y_train).round(3))
print('Testing Accuracy:', tree_pruned_fit.score(X_test, y_test).round(3))
print('AUC Score:', roc_auc_score(y_true  = y_test,
                                  y_score = tree_pred).round(3))

display_tree(tree       = tree_pruned_fit,
             feature_df = X_train)
```


```python
# Instantiating model without hyperparameters
full_gbm_default = GradientBoostingClassifier(loss          = 'deviance',
                                              learning_rate = 0.05,
                                              n_estimators  = 75,
                                              criterion     = 'friedman_mse',
                                              max_depth     = 1,
                                              warm_start    = False,
                                              random_state  = 222)


# Fit step 
full_gbm_default_fit = full_gbm_default.fit(X_train, y_train)


# Predicting based on testing data
full_gbm_default_pred = full_gbm_default_fit.predict(X_test)


# Getting Scores
print('Training ACCURACY:', full_gbm_default_fit.score(X_train, y_train).round(3))
print('Testing ACCURACY :', full_gbm_default_fit.score(X_test, y_test).round(3))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = full_gbm_default_pred).round(3))
```


```python
#Final scores
print('Training ACCURACY:', full_gbm_default_fit.score(X_train, y_train).round(3))
print('Testing ACCURACY :', full_gbm_default_fit.score(X_test, y_test).round(3))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = full_gbm_default_pred).round(3))
```
