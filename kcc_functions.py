# Standard libraries
import numpy as np
import pandas as pd
import pandas as pd 
import warnings
# Functions created
from kcc_functions import *
# Plotting libraries
import matplotlib.pyplot as plt 
import seaborn as sns
# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Datetime libraries
import datetime as dt

# Function to summarize data
def summary_stats(data, title):  
    import pandas as pd
    """
    Generates a Summary table containing the most relevant information of a dataset
    Parameters:
    ----------
    data : dataframe
        Data to summarize
    title : str
        Title of the graph
    Returns:
    --------
    Dataframe
    """ 
    # Generate a general summary of the variables
    df_missingval = pd.DataFrame(data.isna().any(), columns=['Missing vals'])
    df_missingval['# missing'] = data.isna().sum()                                              # Check if there are any missing values
    df_types = pd.DataFrame(data.dtypes, columns=['Variable type'])                             # Obtain the datatypes of all colums
    df_describe = data.describe().round(decimals=2).transpose()                                 # Generate summary statistics
    _ = pd.merge(df_missingval, df_types, how='inner', left_index=True, right_index=True)       # Intermediate merge types and missing val
    df_var_summary = pd.merge(df_describe, _ , how='outer', left_index=True, right_index=True)  # Final merge 
    # df_var_summary.loc['date_of_birth', 'count'] = len(data.index)                             # Replace count of date_of_birth
    print(title.center(120))

    return df_var_summary

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def plot_piechart(data, column, title = 'Piechart', colors = None): 
    import pandas as pd
    """
    Generates a pieplot
    ----------
    data : dataframe
        Data to summarize
    column : str
        Name of the column to plot
    Returns:
    --------
    None 
    """ 
    type_frequencies = data[column].value_counts(normalize='true')    # Count values

    # Plot pie chart
    labels = pd.Series(type_frequencies).index
    fig, ax = plt.subplots()
    ax.pie(type_frequencies, labels=labels, 
            autopct='%1.1f%%', radius=2, colors=colors)
    fig.set_size_inches(3.5,3.5)
    circle = plt.Circle(xy=(0,0), radius=1, facecolor='White')
    plt.gca().add_artist(circle)
    ax.set_title(title, y=0.45)

    return None

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def plot_boxes(data, rows, cols, title=''):
    # Generate boxplots for each variable
    fig, ax = plt.subplots(rows, cols)                                          # Create a rowsxcols grid of subplots
    fig.set_size_inches((11,8))
    fig.suptitle(title, y=1.01)
    plt.subplots_adjust(left=0.1,                                               # Adjust the space between the subplots
                        bottom=0.05, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.3)
    # Iterate over the plots and the quantitative variables
    # to create Kernel Density Estimations (KDE) plots
    col_index = 0
    for row in range(rows):                                                       
        for col in range(cols):                                                    
            current_col = list(data.columns)[col_index]                              
            ax[row][col].set_title(current_col)
            ax[row][col].tick_params(top=False, bottom=False,                   # Remove ticks
            left=True, right=False, labelleft=True, labelbottom=False)
            data.boxplot(column=[current_col], ax=ax[row][col])
            col_index += 1

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def calculate_age(data, date_col):
    # Generate age
    today = pd.Timestamp(dt.date.today())                                           # Get today's date
    data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True)     # Convert column into datetime
    data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True)     # Convert column into datetime
    data['age'] = data[date_col].apply(lambda x: (today - pd.Timestamp(x)).days)    # Calculate dif between dates
    data['age'] = round(data['age'] / 365, 0)                                                   # Convert into years
    data['age'] = data['age'].astype(int)                                                       # Convert column into integer
    data = data.drop(columns=[date_col])                                           # Drop date of birth and id column  

    return data