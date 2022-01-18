import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


sns.set()

#plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    #plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');



def plot_variable_pairs(df, quant_vars):

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

