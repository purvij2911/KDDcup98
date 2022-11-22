'''
Contains all methods to do analysis in this project.
'''

import numpy as np
import pandas as pd
import operator

from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

from Lib.preprocessor import DataFrameImputer
from Lib.preprocessor import Preprocessor

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
   

    @staticmethod
    def get_important_vars(cfg , dat):
        '''
        This method does Feature Selection.
        '''

        # Balances the dataset
        idxs_pos = dat[cfg['target']] == 1
        pos = dat[idxs_pos]
        neg = dat[dat[cfg['target']] == 0][1:sum(idxs_pos)]

        # Concatenates pos and neg, it's already shuffled
        sub_dat = pos.append(neg, ignore_index = True)

        # Imputes the data and fills in the missing values
        sub_dat = Preprocessor.fill_nans(sub_dat)

        # Changes categorical vars to a numerical form
        X = pd.get_dummies(sub_dat)

        #### Correlation-based Feature Selection ####

        # Computes correlation between cfg['target'] and the predictors
        target_corr = X.corr()[cfg['target']].copy()
        target_corr.sort(ascending = False)

        # Sorts and picks the first x features
        # TODO: get optimal x value automatically
        tmp = abs(target_corr).copy()
        tmp.sort(ascending = False)
        important_vars = [tmp.index[0]]
        important_vars.extend(list(tmp.index[2:52])) # removes other target

        #### Variance-based Feature Selection ####

        #sel = VarianceThreshold(threshold = 0.005)
        #X_new = sel.fit_transform(X)

        #### Univariate Feature Selection ####

        #y = X.TARGET_B
        #X = X.drop("TARGET_B", axis = 1)

        #X_new = SelectKBest(chi2, k = 10).fit_transform(X.values, y.values)

        #### Tree-based Feature Selection ####

        #clf = ExtraTreesClassifier()
        #X_new = clf.fit(X.values, y.values).transform(X.values)

        #aux = dict(zip(X.columns, clf.feature_importances_))
        #important_vars = [i[0] for i in sorted(
        #    aux.items(), key = operator.itemgetter(0))]

        return important_vars
