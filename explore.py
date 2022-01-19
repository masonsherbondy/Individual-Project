import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


sns.set()



def plot_distributions(df, quant_vars):

    '''
    This function accepts a dataframe and a list of features, and it plots histograms and boxplots for each feature.
    '''

    for cat in quant_vars:
        df[cat].hist(color = 'indigo')
        plt.title(cat, pad = 11)
        plt.xlabel(cat)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show();


def distributions_plot(df, quant_vars):

    plt.figure(figsize = (20, 11))   # create figure

    for i, cat in enumerate(quant_vars):    # loop through enumerated list
    
        plot_number = i + 1     # i starts at 0, but plot nos should start at 1
        
        plt.subplot(5, 5, plot_number)  # create subplot
        
        plt.title(cat)  # title
        
        df[cat].hist(color = 'indigo', edgecolor='black')   # display histogram for column

        plt.tight_layout(); # clean
    



#plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    '''
    This function shows the relationship between two variables (a categorical feature and a continuous one) on 3 different kind of plots (box, strip, violin)
    '''

    #plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');



def plot_variable_pairs(df, quant_vars):
    
    '''
    This function enumerates the number of features passed in a list, forms all possible pairs for the number of features it selects,
    and runs a pearson's correlation test on each pair, and then plots the relationship between each pair as well as a regression line, and titles
    each plot with Pearson's R and the respective p-value. This function currently accepts up to 11 features for pairing.
    '''

    #determine k
    k = len(quant_vars)

    #set up if-conditional to see how many features are being paired
    if k == 2:

        #determine correlation coefficient
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');


    #pair 3 features
    if k == 3:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');


    #pair 4 features
    if k == 4:
        
        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');


    #pair 5 features
    if k == 5:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        

        #plot relationships between continuous variables
        
        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

    #pair 6 features
    if k == 6:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])

        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');


    #pair 7 features
    if k == 7:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

        #plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

        #plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');

        #plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');


    #pair 8 features
    if k == 8:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');
    
    #pair 9 features
    if k == 9:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

    #pair 10 features
    if k == 10:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])




        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

    #pair 11 features
    if k == 11:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])





        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        #plot XXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        #plot XXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        #plot XXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        #plot XXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        #plot XXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        #plot XXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        #plot XXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        #plot XXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        #plot XXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        #plot XXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

if k == 12:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])





        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        #plot XXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        #plot XXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        #plot XXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        #plot XXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        #plot XXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        #plot XXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        #plot XXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        #plot XXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        #plot XXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        #plot XXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        #plot XXXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        #plot XXXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        #plot XXXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        #plot XXXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        #plot XXXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        #plot XXXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        #plot XXXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        #plot XXXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        #plot XXXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        #plot XXXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        #plot XXXXXXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');


if k == 13:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])




        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        #plot XXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        #plot XXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        #plot XXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        #plot XXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        #plot XXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        #plot XXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        #plot XXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        #plot XXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        #plot XXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        #plot XXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        #plot XXXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        #plot XXXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        #plot XXXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        #plot XXXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        #plot XXXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        #plot XXXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        #plot XXXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        #plot XXXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        #plot XXXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        #plot XXXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        #plot XXXXXXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        #plot XXXXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        #plot XXXXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        #plot XXXXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        #plot XXXXXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        #plot XXXXXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        #plot XXXXXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        #plot XXXXXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        #plot XXXXXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        #plot XXXXXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        #plot XXXXXXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        #plot XXXXXXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        #plot XXXXXXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

    if k == 14:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])



        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        #plot XXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        #plot XXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        #plot XXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        #plot XXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        #plot XXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        #plot XXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        #plot XXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        #plot XXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        #plot XXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        #plot XXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        #plot XXXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        #plot XXXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        #plot XXXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        #plot XXXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        #plot XXXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        #plot XXXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        #plot XXXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        #plot XXXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        #plot XXXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        #plot XXXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        #plot XXXXXXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        #plot XXXXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        #plot XXXXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        #plot XXXXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        #plot XXXXXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        #plot XXXXXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        #plot XXXXXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        #plot XXXXXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        #plot XXXXXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        #plot XXXXXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        #plot XXXXXXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        #plot XXXXXXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        #plot XXXXXXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        #plot XXXXXXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        #plot XXXXXXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        #plot XXXXXXXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        #plot XXXXXXXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        #plot XXXXXXXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        #plot XXXXXXXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        #plot XXXXXXXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        #plot XXXXXXXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        #plot XXXXXXXXVII
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        #plot XXXXXXXXVIII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        #plot XXXXXXXXIX
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        #plot XXXXXXXXX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');

        #plot XXXXXXXXXI
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr91, 3)} | P-value: {round(p91, 4)} \n -----------------');

    if k == 14:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])
        corr29, p29 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[0]])
        corr30, p30 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[1]])
        corr31, p31 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[2]])
        corr32, p32 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[3]])
        corr33, p33 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[4]])
        corr34, p34 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[5]])
        corr35, p35 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[6]])
        corr36, p36 = stats.pearsonr(df[quant_vars[8]], df[quant_vars[7]])
        corr37, p37 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[0]])
        corr38, p38 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[1]])
        corr39, p39 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[2]])
        corr40, p40 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[3]])
        corr41, p41 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[4]])
        corr42, p42 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[5]])
        corr43, p43 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[6]])
        corr44, p44 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[7]])
        corr45, p45 = stats.pearsonr(df[quant_vars[9]], df[quant_vars[8]])
        corr46, p46 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[0]])
        corr47, p47 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[1]])
        corr48, p48 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[2]])
        corr49, p49 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[3]])
        corr50, p50 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[4]])
        corr51, p51 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[5]])
        corr52, p52 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[6]])
        corr53, p53 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[7]])
        corr54, p54 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[8]])
        corr55, p55 = stats.pearsonr(df[quant_vars[10]], df[quant_vars[9]])
        corr56, p56 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[0]])
        corr57, p57 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[1]])
        corr58, p58 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[2]])
        corr59, p59 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[3]])
        corr60, p60 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[4]])
        corr61, p61 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[5]])
        corr62, p62 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[6]])
        corr63, p63 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[7]])
        corr64, p64 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[8]])
        corr65, p65 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[9]])
        corr66, p66 = stats.pearsonr(df[quant_vars[11]], df[quant_vars[10]])
        corr67, p67 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[0]])
        corr68, p68 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[1]])
        corr69, p69 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[2]])
        corr70, p70 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[3]])
        corr71, p71 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[4]])
        corr72, p72 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[5]])
        corr73, p73 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[6]])
        corr74, p74 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[7]])
        corr75, p75 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[8]])
        corr76, p76 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[9]])
        corr77, p77 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[10]])
        corr78, p78 = stats.pearsonr(df[quant_vars[12]], df[quant_vars[11]])
        corr79, p79 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[0]])
        corr80, p80 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[1]])
        corr81, p81 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[2]])
        corr82, p82 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[3]])
        corr83, p83 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[4]])
        corr84, p84 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[5]])
        corr85, p85 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[6]])
        corr86, p86 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[7]])
        corr87, p87 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[8]])
        corr88, p88 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[9]])
        corr89, p89 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[10]])
        corr90, p90 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[11]])
        corr91, p91 = stats.pearsonr(df[quant_vars[13]], df[quant_vars[12]])
        corr78, p78 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[0]])
        corr79, p79 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[1]])
        corr80, p80 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[2]])
        corr81, p81 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[3]])
        corr82, p82 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[4]])
        corr83, p83 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[5]])
        corr84, p84 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[6]])
        corr85, p85 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[7]])
        corr86, p86 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[8]])
        corr87, p87 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[9]])
        corr88, p88 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[10]])
        corr89, p89 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[11]])
        corr90, p90 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[12]])
        corr91, p91 = stats.pearsonr(df[quant_vars[14]], df[quant_vars[13]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');

        #plot XXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr29, 3)} | P-value: {round(p29, 4)} \n -----------------')

        #plot XXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr30, 3)} | P-value: {round(p30, 4)} \n -----------------');

        #plot XXXI
        sns.lmplot(x = quant_vars[2], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr31, 3)} | P-value: {round(p31, 4)} \n -----------------');

        #plot XXXII
        sns.lmplot(x = quant_vars[3], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr32, 3)} | P-value: {round(p32, 4)} \n -----------------');

        #plot XXXIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr33, 3)} | P-value: {round(p33, 4)} \n -----------------');

        #plot XXXIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr34, 3)} | P-value: {round(p34, 4)} \n -----------------');

        #plot XXXV
        sns.lmplot(x = quant_vars[6], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr35, 3)} | P-value: {round(p35, 4)} \n -----------------');

        #plot XXXVI
        sns.lmplot(x = quant_vars[7], y = quant_vars[8], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr36, 3)} | P-value: {round(p36, 4)} \n -----------------');

        #plot XXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr37, 3)} | P-value: {round(p37, 4)} \n -----------------');

        #plot XXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr38, 3)} | P-value: {round(p38, 4)} \n -----------------');

        #plot XXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr39, 3)} | P-value: {round(p39, 4)} \n -----------------')

        #plot XXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr40, 3)} | P-value: {round(p40, 4)} \n -----------------');

        #plot XXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr41, 3)} | P-value: {round(p41, 4)} \n -----------------');

        #plot XXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr42, 3)} | P-value: {round(p42, 4)} \n -----------------');

        #plot XXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr43, 3)} | P-value: {round(p43, 4)} \n -----------------');

        #plot XXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr44, 3)} | P-value: {round(p44, 4)} \n -----------------');

        #plot XXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[9], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr45, 3)} | P-value: {round(p45, 4)} \n -----------------');

        #plot XXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr46, 3)} | P-value: {round(p46, 4)} \n -----------------');

        #plot XXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr47, 3)} | P-value: {round(p47, 4)} \n -----------------');

        #plot XXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr48, 3)} | P-value: {round(p48, 4)} \n -----------------');

        #plot XXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr49, 3)} | P-value: {round(p49, 4)} \n -----------------')

        #plot XXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr50, 3)} | P-value: {round(p50, 4)} \n -----------------');

        #plot XXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr51, 3)} | P-value: {round(p51, 4)} \n -----------------');

        #plot XXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr52, 3)} | P-value: {round(p52, 4)} \n -----------------');

        #plot XXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr53, 3)} | P-value: {round(p53, 4)} \n -----------------');

        #plot XXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr54, 3)} | P-value: {round(p54, 4)} \n -----------------');

        #plot XXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[10], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr55, 3)} | P-value: {round(p55, 4)} \n -----------------');

        #plot XXXXXVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr56, 3)} | P-value: {round(p56, 4)} \n -----------------');

        #plot XXXXXVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr57, 3)} | P-value: {round(p57, 4)} \n -----------------');

        #plot XXXXXVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr58, 3)} | P-value: {round(p58, 4)} \n -----------------');

        #plot XXXXXIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr59, 3)} | P-value: {round(p59, 4)} \n -----------------');

        #plot XXXXXX
        sns.lmplot(x = quant_vars[4], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr60, 3)} | P-value: {round(p60, 4)} \n -----------------')

        #plot XXXXXXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr61, 3)} | P-value: {round(p61, 4)} \n -----------------');

        #plot XXXXXXII
        sns.lmplot(x = quant_vars[6], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr62, 3)} | P-value: {round(p62, 4)} \n -----------------');

        #plot XXXXXXIII
        sns.lmplot(x = quant_vars[7], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr63, 3)} | P-value: {round(p63, 4)} \n -----------------');

        #plot XXXXXXIV
        sns.lmplot(x = quant_vars[8], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr64, 3)} | P-value: {round(p64, 4)} \n -----------------');

        #plot XXXXXXV
        sns.lmplot(x = quant_vars[9], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr65, 3)} | P-value: {round(p65, 4)} \n -----------------');

        #plot XXXXXXVI
        sns.lmplot(x = quant_vars[10], y = quant_vars[11], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr66, 3)} | P-value: {round(p66, 4)} \n -----------------');

        #plot XXXXXXVII
        sns.lmplot(x = quant_vars[0], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr67, 3)} | P-value: {round(p67, 4)} \n -----------------');

        #plot XXXXXXVIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr68, 3)} | P-value: {round(p68, 4)} \n -----------------');

        #plot XXXXXXIX
        sns.lmplot(x = quant_vars[2], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr69, 3)} | P-value: {round(p69, 4)} \n -----------------');

        #plot XXXXXXX
        sns.lmplot(x = quant_vars[3], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr70, 3)} | P-value: {round(p70, 4)} \n -----------------');

        #plot XXXXXXXI
        sns.lmplot(x = quant_vars[4], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr71, 3)} | P-value: {round(p71, 4)} \n -----------------');

        #plot XXXXXXXII
        sns.lmplot(x = quant_vars[5], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr72, 3)} | P-value: {round(p72, 4)} \n -----------------')

        #plot XXXXXXXIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr73, 3)} | P-value: {round(p73, 4)} \n -----------------');

        #plot XXXXXXXIV
        sns.lmplot(x = quant_vars[7], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr74, 3)} | P-value: {round(p74, 4)} \n -----------------');

        #plot XXXXXXXV
        sns.lmplot(x = quant_vars[8], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr75, 3)} | P-value: {round(p75, 4)} \n -----------------');

        #plot XXXXXXXVI
        sns.lmplot(x = quant_vars[9], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr76, 3)} | P-value: {round(p76, 4)} \n -----------------');

        #plot XXXXXXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        #plot XXXXXXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        #plot XXXXXXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        #plot XXXXXXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        #plot XXXXXXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        #plot XXXXXXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        #plot XXXXXXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        #plot XXXXXXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        #plot XXXXXXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        #plot XXXXXXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        #plot XXXXXXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        #plot XXXXXXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        #plot XXXXXXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        #plot XXXXXXXXX
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');



        #plot XXXXXXXVII
        sns.lmplot(x = quant_vars[10], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr77, 3)} | P-value: {round(p77, 4)} \n -----------------');

        #plot XXXXXXXVIII
        sns.lmplot(x = quant_vars[11], y = quant_vars[12], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr78, 3)} | P-value: {round(p78, 4)} \n -----------------');

        #plot XXXXXXXIX
        sns.lmplot(x = quant_vars[0], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr79, 3)} | P-value: {round(p79, 4)} \n -----------------');

        #plot XXXXXXXX
        sns.lmplot(x = quant_vars[1], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr80, 3)} | P-value: {round(p80, 4)} \n -----------------');

        #plot XXXXXXXXI
        sns.lmplot(x = quant_vars[3], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr81, 3)} | P-value: {round(p81, 4)} \n -----------------');

        #plot XXXXXXXXII
        sns.lmplot(x = quant_vars[4], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr82, 3)} | P-value: {round(p82, 4)} \n -----------------');

        #plot XXXXXXXXIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr83, 3)} | P-value: {round(p83, 4)} \n -----------------');

        #plot XXXXXXXXIV
        sns.lmplot(x = quant_vars[6], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr84, 3)} | P-value: {round(p84, 4)} \n -----------------');

        #plot XXXXXXXXV
        sns.lmplot(x = quant_vars[7], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr85, 3)} | P-value: {round(p85, 4)} \n -----------------');

        #plot XXXXXXXXVI
        sns.lmplot(x = quant_vars[8], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr86, 3)} | P-value: {round(p86, 4)} \n -----------------');

        #plot XXXXXXXXVII
        sns.lmplot(x = quant_vars[9], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr87, 3)} | P-value: {round(p87, 4)} \n -----------------');

        #plot XXXXXXXXVIII
        sns.lmplot(x = quant_vars[10], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr88, 3)} | P-value: {round(p88, 4)} \n -----------------');

        #plot XXXXXXXXIX
        sns.lmplot(x = quant_vars[11], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr89, 3)} | P-value: {round(p89, 4)} \n -----------------');

        #plot XXXXXXXXX
        sns.lmplot(x = quant_vars[12], y = quant_vars[13], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr90, 3)} | P-value: {round(p90, 4)} \n -----------------');



#<^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^> CUSTOMIZED RETURNS ON STATS TESTS FOUND HERE <^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^>#




#return_chi2 defines one parameter, an observed cross-tabulation, runs the stats.chi2_contingency function and returns the test results in a readable format.
def return_chi2(observed):
    
    #run the test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    #print the rest
    print('Observed')
    print('--------')
    print(observed.values)
    print('===================')
    print('Expected')
    print('--------')
    print(expected.astype(int))
    print('===================')
    print(f'Degrees of Freedom: {degf}')
    print('===================')
    print('Chi^2 and P')
    print('-----------')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p = {p:.4f}')





### /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ FEATURE SELECTION FUNCTIONS HERE /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ ###



### Note: must define X_train and y_train prior to running feature selection functions
## note: also these output lists are ordered backward

#X_train = predictors or features (same thing if you got the right features)
#y_train = target
#k = number of features you want

#select_kbest defines 3 parameters, X_train (predictors), y_train (target variable) and k (number of features to spit), and returns a list of the best features my man
def select_kbest(X_train, y_train, k):

    #import feature selection tools
    from sklearn.feature_selection import SelectKBest, f_regression

    #create the selector
    f_select = SelectKBest(f_regression, k = k)

    #fit the selector
    f_select.fit(X_train, y_train)

    #create a boolean mask to show if feature was selected
    feat_mask = f_select.get_support()
    
    #create a list of the best features
    best_features = X_train.iloc[:,feat_mask].columns.to_list()

    #gimme gimme
    return best_features



#rfe defines 3 parameters, X_train (features), y_train (target variable) and k (number of features to bop), and returns a list of the best boppits m8
def rfe(X_train, y_train, k):

    #import feature selection tools
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    #crank it
    lm = LinearRegression()

    #pop it
    rfe = RFE(lm, k)
    
    #bop it
    rfe.fit(X_train, y_train)  
    
    #twist it
    feat_mask = rfe.support_
    
    #pull it 
    best_rfe = X_train.iloc[:,feat_mask].columns.tolist()
    
    #bop it
    return best_rfe
