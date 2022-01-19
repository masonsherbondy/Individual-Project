import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import mason_functions as mf


def calculate_column_nulls(df):

    '''
    This function  defines one parameter, a dataframe, and returns a dataframe that holds data regarding null values and ratios (pertaining to the whole column) 
    in the original frame.
    '''   
    
    #set an empty list
    output = []
    
    #gather columns
    df_columns = df.columns.to_list()
    
    #commence for loop
    for column in df_columns:
        
        #assign variable to number of rows that have null values
        missing = df[column].isnull().sum()
        
        #assign variable to ratio of rows with null values to overall rows in column
        ratio = missing / len(df)
    
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    column_nulls = pd.DataFrame(output, index = df_columns)
    
    #return the dataframe
    return column_nulls


def calculate_row_nulls(df):

    '''
    This function  defines one parameter, a dataframe, and returns a dataframe that holds data regarding null values and ratios (pertaining to the whole row) 
    in the original frame.
    '''   
    
    #create an empty list
    output = []
    
    #gather values in a series
    nulls = df.isnull().sum(axis = 1)
    
    #commence 4 loop
    for n in range(len(nulls)):
        
        #assign variable to nulls
        missing = nulls[n]
        
        #assign variable to ratio
        ratio = missing / len(df.columns)
        
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    row_nulls = pd.DataFrame(output, index = df.index)
    
    #return the dataframe
    return row_nulls



def handle_nulls(df, cr, rr):

    '''
    This function defines 3 parameters, the dataframe you want to calculate nulls for (df), the column ratio (cr) and the row ratio (rr),
    ratios that define the threshold of having too many nulls, and returns your dataframe with columns and rows dropped if they are above their respective threshold ratios.
    Note: This function calculates the ratio of nulls missing for rows AFTER it drops the columns with null ratios above the cr threshold.
    TL; DR: This function handles nulls for dataframes.
    '''
    
    #set an empty list
    output = []
    
    #gather columns
    df_columns = df.columns.to_list()
    
    #commence for loop
    for column in df_columns:
        
        #assign variable to number of rows that have null values
        missing = df[column].isnull().sum()
        
        #assign variable to ratio of rows with null values to overall rows in column
        ratio = missing / len(df)
    
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    column_nulls = pd.DataFrame(output, index = df_columns)

    #set a list of columns to drop
    null_and_void = []
    
    #commence loop
    for n in range(len(column_nulls)):
        
        #set up conditional to see if the ratio of nulls to total columns is over the entered column ratio (.cr)
        if column_nulls.iloc[n].null_ratio > cr:
            
            #add columns over the threshold with nulls to the list of columns to drop
            null_and_void.append(column_nulls.index[n])
    
    #drop the columns
    df = df.drop(columns = null_and_void)

    #create another list
    output = []
    
    #gather values in a series
    nulls = df.isnull().sum(axis = 1)
    
    #commence 4 loop
    for n in range(len(nulls)):
        
        #assign variable to nulls
        missing = nulls[n]
        
        #assign variable to ratio
        ratio = missing / len(df.columns)
        
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    row_nulls = pd.DataFrame(output, index = df.index)

    #set an empty index of rows to drop
    ice_em = []

    #commence loop
    for n in range(len(row_nulls)):
        
        #set up conditional to see if the ratio of nulls to total columns is over the entered row ratio (.rr)
        if row_nulls.iloc[n].null_ratio > rr:
            
            #add rows to index
            ice_em.append(row_nulls.index[n])
    
    #drop rows where the percentage of nulls is over the threshold
    df = df.drop(index = ice_em)

    #return the df with preferred drop parameters
    return df


#reference for splitting data in a classification setting
def class_split_data(df, target):

    '''
    Takes in a dataset and returns the train, validate, and test subset dataframes.
    Dataframe size for my test set is .2 or 20% of the original data. 
    Validate data is 30% of my training set, which is 24% of the original data. 
    Training data is 56% of the original data. 
    This function stratifies by the target variable.
    '''

    #import splitter
    from sklearn.model_selection import train_test_split
    
    #get my training and test data sets defined, stratify my target variable
    train, test = train_test_split(df, test_size = .2, random_state = 421, stratify = df[target])
    
    #get my validate set from the training set, stratify target variable again
    train, validate = train_test_split(train, test_size = .3, random_state = 421, stratify = train[target])
    
    #return the 3 dataframes
    return train, validate, test


def remove_outliers(df, k, col_list):

    ''' 
    Removes outliers from a list of columns in a dataframe and returns the dataframe.
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def acquire_attrition():

    '''
    This function acquires my IBM data by reading a downloadable .xls file, which is also conveniently in csv format.
    '''

    df = pd.read_csv('employee_attrition.xls', index_col = 0)

    return df


def prepare_attrition():

    '''
    This function loads and preps the data for exploration and modeling.
    '''
    
    #acquire data
    df = acquire_attrition()

    #set up if-conditional to see if there is a .csv available
    if os.path.isfile('attrition_IBM.csv'):

        #if there is, read the .csv into a dataframe
        cf = pd.read_csv('attrition_IBM.csv', index_col = 0)

        #and this .csv, as well
        df = pd.read_csv('net_attrition.csv', index_col = 0)

    #otherwise...
    else:

        #get rid of camel-case (lower case all of the columns) 
        df.columns = df.columns.str.lower()

        #rename columns
        df = df.rename(columns = {'businesstravel': 'business_travel',
                                'dailyrate': 'daily_rate',
                                'distancefromhome': 'distance_from_home',
                                'educationfield': 'education_field',
                                'employeecount': 'employee_count',
                                'employeenumber': 'employee_id',
                                'environmentsatisfaction': 'environment_satisfaction',
                                'hourlyrate': 'hourly_rate',
                                'jobinvolvement': 'job_involvement',
                                'joblevel': 'job_level',
                                'jobrole': 'job_role',
                                'jobsatisfaction': 'job_satisfaction',
                                'maritalstatus': 'marital_status',
                                'monthlyincome': 'monthly_income',
                                'monthlyrate': 'monthly_rate',
                                'numcompaniesworked': 'companies_worked',
                                'percentsalaryhike': 'percent_salary_hike',
                                'performancerating': 'performance_rating',
                                'relationshipsatisfaction': 'relationship_satisfaction',
                                'standardhours': 'standard_hours',
                                'stockoptionlevel': 'stock_option_level',
                                'totalworkingyears': 'total_working_years',
                                'trainingtimeslastyear': 'hours_trained_last_year',
                                'worklifebalance': 'work_life_balance',
                                'yearsatcompany': 'company_years',
                                'yearsincurrentrole': 'current_role_years',
                                'yearssincelastpromotion': 'years_since_last_promotion',
                                'yearswithcurrmanager': 'years_with_manager'
                                }
                    )

        #set a list of categorical columns based on the columns above
        categorical = ['attrition',
                    'business_travel',
                    'department',
                    'education_field',
                    'gender',
                    'job_role',
                    'marital_status',
                    'overtime'
                    ]

        #set a list of discrete columns
        discrete = ['education',
                    'environment_satisfaction',
                    'job_involvement',
                    'job_level',
                    'job_satisfaction',
                    'performance_rating',
                    'relationship_satisfaction', 
                    'stock_option_level',
                    'work_life_balance'
                ]

        #set a list of numeric columns
        numeric = ['daily_rate',
                'distance_from_home',
                'education',
                'environment_satisfaction',
                'hourly_rate',
                'job_involvement',
                'job_level',
                'job_role',
                'job_satisfaction',
                'monthly_income',
                'monthly_rate',
                'companies_worked',
                'percent_salary_hike',
                'performance_rating',
                'relationship_satisfaction',
                'stock_option_level',
                'total_working_years',
                'hours_trained_last_year',
                'work_life_balance',
                'company_years',
                'current_role_years',
                'years_since_last_promotion',
                'years_with_manager'
                ]

        #set a list of columns to drop
        superfluous = ['employee_count', 'over18', 'standard_hours']

        #drop these columns
        df = df.drop(columns = superfluous)

        #map simple categorical columns to 1 and 0
        df.attrition = df.attrition.map({'No': 0, 'Yes': 1})
        df.overtime = df.overtime.map({'No': 0, 'Yes': 1})

        #remove attrition and overtime since we just one-hot encoded them already
        categorical.remove('attrition')
        categorical.remove('overtime')

        #assign dataframe to dummied variables
        dummies = pd.get_dummies(df[categorical], drop_first = False)

        #lowercase all dummies. replace superfluous string with nothing. clean up columns
        dummies.columns = dummies.columns.str.lower().\
        str.replace(' ', '_').\
        str.replace('business_travel_', '').\
        str.replace('department_human_resources', 'hr_dept').\
        str.replace('department_', '').\
        str.replace('education_field_human_resources', 'hr_ed').\
        str.replace('education_field_', '').\
        str.replace('gender_', '').\
        str.replace('job_role_human_resources', 'hr_job').\
        str.replace('job_role_', '').\
        str.replace('marital_status_', '').\
        str.replace('&', '').\
        str.replace('__', '_').\
        str.replace('non-travel', 'travel_none').\
        str.replace('development', 'dev_dept').\
        str.replace('sales', 'sales_dept').\
        str.replace('human_resources', 'hr_ed', 1).\
        str.replace('life_sciences', 'life_sciences_ed').\
        str.replace('marketing', 'marketing_ed').\
        str.replace('medical', 'medical_ed').\
        str.replace('other', 'other_ed').\
        str.replace('technical_degree', 'tech_deg_ed').\
        str.replace('gender_', '').\
        str.replace('healthcare_representative', 'healthcare_rep_job').\
        str.replace('laboratory_technician', 'lab_tech_job').\
        str.replace('manager', 'manager_job').\
        str.replace('manufacturing_director', 'manufacturing_dir_job').\
        str.replace('research_director', 'research_dir_job').\
        str.replace('research_scientist', 'research_scientist_job').\
        str.replace('sales_dept_executive', 'sales_exec_job').\
        str.replace('sales_dept_representative', 'sales_rep_job')

        #drop extra column
        dummies = dummies.drop(columns = 'male')

        #concatenate dummy df with OG df and new cf
        df = pd.concat([df, dummies], axis = 1)
        cf = pd.concat([df, dummies], axis = 1)

        #drop redundant columns on cf
        cf = cf.drop(columns = categorical)

        #make new feature 'age' out of index. reset index to 'employee_id'
        cf['age'] = cf.index
        cf = cf.set_index('employee_id')
        df['age'] = df.index
        df = df.set_index('employee_id')

        #write dfs into .csv files for later ease of access
        cf.to_csv('attrition_IBM.csv')
        df.to_csv('net_attrition.csv')

    #split the data
    train, validate, test = class_split_data(cf, 'attrition')

    #return gross dataframe and modeling sets
    return df, train, validate, test