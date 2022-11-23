'''
Contains all methods to do analysis in this project.
'''

import numpy as np
import pandas as pd
import operator

from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

from lib.preprocessor import DataFrameImputer
from lib.preprocessor import Preprocessor

class Analyser:

    @staticmethod
    def get_corr_vars(dat, corr_val):
        '''
        Outputs a list of redundant vars that are correlated with others.
        '''

        # Computes correlation
        dat_cor = dat.corr()

        # Cherry picks the lower triangular, excludes diagonal
        dat_cor.loc[:, :] = np.tril(dat_cor, k = -1)

        # Stacks the data.frame
        dat_cor = dat_cor.stack()

        # Get list of correlated vars
        corr_pairs = dat_cor[dat_cor > corr_val].to_dict().keys()
        chosen_vars = [i[0] for i in corr_pairs]
        chosen_vars.extend([i[1] for i in corr_pairs if i[1] not in chosen_vars])

        redundant_vars = [var for var in [
            x for t in corr_pairs for x in t] if var not in chosen_vars]

        return redundant_vars

    @staticmethod
    def get_redundant_vars(dat):
        '''
        This method outputs a set of redundant variables.
        '''

        # Some vars that don't seem of good value
        redundant_vars = ['CONTROLN', 'ZIP']

        # Identifies numerical variables with variance zero < 0.1%
        #sel = feature_selection.VarianceThreshold(threshold = 0.001)
        #sel.fit_transform(dat)
        dat_var = dat.var()
        redundant_vars.extend(dat_var.index[dat_var < 0.001])

        # Identifies variables that are too sparse (less than 1%)
        idxs = dat.count() < int(dat.shape[0] * .01)
        redundant_vars.extend(dat.columns[idxs])

        # Identifies variables that are strongly correlated with others
        #redundant_vars.extend(Analyser.get_corr_vars(dat, corr_val = 0.9)) 
       

        return redundant_vars
    
    @staticmethod
    def get_high_cardinality_vars(dat):
        '''
        This method outputs a set of variables which have very high cardinality

        '''
        cardinality_vars=[]
        #Identifies categorical variables which have very high cardinality >60  
        #These are deleted so that the data does not explode when creating dummies
        dat_cardi = dat[dat.columns[dat.dtypes == 'object']].nunique()
        cardinality_vars.extend(dat_cardi.index[dat_cardi > 60])
        
        return cardinality_vars

    
    

    @staticmethod
    def get_nan_cols(dat):
        '''
        This method filters the columns which have high nan values >65% .
        '''
    
        #Removing columns with high  null values in the dataset
        nan_cols = dat.columns[dat.isnull().sum()/len(dat)*100 >65].to_list()
       
        
        return nan_cols
   


    