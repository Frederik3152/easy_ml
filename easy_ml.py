#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing regression libraries
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.linear_model import Lars, LarsCV, LassoLars, LassoLarsIC, LassoLarsCV, SGDRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, RANSACRegressor, OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveRegressor, OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.dummy import DummyRegressor

#Importing classification libraries
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.dummy import DummyClassifier

#Miscellaneous libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
from prettytable import PrettyTable
import warnings

#Importing metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

warnings.filterwarnings('ignore')
class easy_ml:
    def __init__(self, features, target_var, standard_scaler = False, min_max_scaler = False):
        self.features = features
        self.target_var = target_var
        
        #Applying the StandardScaler to the feature data
        if standard_scaler:
            self.features = StandardScaler().fit_transform(self.features) 
        
        #Applying the MinMaxScaler to the feature data
        if min_max_scaler:
            self.features = MinMaxScaler().fit_transform(self.features) 
        
        #Using the constructor function to create the train / test split
        train_x, test_x, train_y, test_y = train_test_split(self.features, self.target_var, 
                                           test_size = 0.2, 
                                           random_state = 42)
        
        #Setting up the train / test values as class variables
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        
    def regression(self):
        #Creating a pipeline with each model that needs to be evaluated
        model_pipeline = []
        model_pipeline.append(LinearRegression())
        model_pipeline.append(Lasso())
        model_pipeline.append(Ridge())
        model_pipeline.append(RandomForestRegressor())
        model_pipeline.append(DecisionTreeRegressor())
        model_pipeline.append(ElasticNet())
        model_pipeline.append(ExtraTreesRegressor())
        model_pipeline.append(AdaBoostRegressor())
        model_pipeline.append(NuSVR())
        model_pipeline.append(GradientBoostingRegressor())
        model_pipeline.append(KNeighborsRegressor(n_neighbors=5))
        model_pipeline.append(HistGradientBoostingRegressor())
        model_pipeline.append(BaggingRegressor())
        model_pipeline.append(MLPRegressor())
        model_pipeline.append(HuberRegressor())
        model_pipeline.append(RidgeCV())
        model_pipeline.append(BayesianRidge())
        model_pipeline.append(LassoCV())
        model_pipeline.append(ElasticNetCV())              
        model_pipeline.append(LassoLarsCV())
        model_pipeline.append(LassoLarsIC())        
        model_pipeline.append(LarsCV())
        model_pipeline.append(Lars())
        model_pipeline.append(SGDRegressor())
        model_pipeline.append(RANSACRegressor())              
        model_pipeline.append(OrthogonalMatchingPursuitCV())
        model_pipeline.append(ExtraTreeRegressor())        
        model_pipeline.append(PassiveAggressiveRegressor())
        model_pipeline.append(GaussianProcessRegressor())           
        model_pipeline.append(OrthogonalMatchingPursuit())
        model_pipeline.append(LassoLars())           
        model_pipeline.append(DummyRegressor())
        model_pipeline.append(SVR(kernel='linear'))
        model_pipeline.append(SVR(kernel='poly'))
        model_pipeline.append(SVR(kernel='rbf'))

        #Creating arrays to store the different values of model.fit, model.predict and metrics
        fitted_model_pipeline = []
        model_prediction = []
        model_accuracy = []
        model_RMSE = []
        #Fitting the models to the data and applying the algorithms to predict
        i = 0
        for model in model_pipeline:
            fitted_model_pipeline.append(model_pipeline[i].fit(self.train_x, self.train_y))
            model_prediction.append(fitted_model_pipeline[i].predict(self.test_x))
            #Evaluating r2 score and root-mean-square-error of each model
            model_accuracy.append(round(r2_score(self.test_y, model_prediction[i]), 5))
            model_RMSE.append(round(mean_squared_error(self.test_y, model_prediction[i], squared=False), 5))
            i+= 1

        #Printing the metric arrays for each model using prettytable library
        t = PrettyTable()        
        i = 0
        t.field_names = ["Model", "R-squared", "RMSE"]
        for model in model_pipeline:
            #Converting each model to a string, to allow the 'sortby' function to work
            model_str = ' '.join(map(str, [model]))
            t.add_row([model_str, model_accuracy[i], model_RMSE[i]])
            i+= 1
        t.align = 'r'
        t.sortby = 'R-squared'
        t.reversesort=True
        print(t)
        
        #Determining the most accurate model according to the coefficient of determination 
        best_model = model_accuracy.index(max(model_accuracy))
        return deepcopy(fitted_model_pipeline[best_model])
   
    def classification(self):
        #Creating a pipeline with each model that needs to be evaluated
        model_pipeline = []
        model_pipeline.append(KNeighborsClassifier(n_neighbors=5))
        model_pipeline.append(LogisticRegression())
        model_pipeline.append(DecisionTreeClassifier())
        model_pipeline.append(RandomForestClassifier())
        model_pipeline.append(GaussianNB())
        model_pipeline.append(MultinomialNB())
        model_pipeline.append(ExtraTreesClassifier())
        model_pipeline.append(AdaBoostClassifier())
        model_pipeline.append(NuSVC())
        model_pipeline.append(GradientBoostingClassifier())
        model_pipeline.append(HistGradientBoostingClassifier())
        model_pipeline.append(BaggingClassifier())
        model_pipeline.append(MLPClassifier())
        model_pipeline.append(RidgeClassifier())
        model_pipeline.append(RidgeClassifierCV())
        model_pipeline.append(SGDClassifier())
        model_pipeline.append(Perceptron())
        model_pipeline.append(LogisticRegressionCV())         
        model_pipeline.append(PassiveAggressiveClassifier())
        model_pipeline.append(LabelPropagation())
        model_pipeline.append(LabelSpreading())
        model_pipeline.append(BernoulliNB())
        model_pipeline.append(NearestCentroid())             
        model_pipeline.append(ExtraTreeClassifier())          
        model_pipeline.append(DummyClassifier())    
        model_pipeline.append(SVC(kernel='linear'))
        model_pipeline.append(SVC(kernel='poly'))
        model_pipeline.append(SVC(kernel='rbf'))

       #Creating arrays to store the different values of model.fit, model.predict and metrics
        fitted_model_pipeline = []
        model_prediction = []
        model_accuracy = []
        model_b_accuracy = []
        model_roc_auc = []
        model_f1 = []
        #Fitting the models to the data and applying the algorithms to predict
        i = 0
        for model in model_pipeline:
            fitted_model_pipeline.append(model_pipeline[i].fit(self.train_x, self.train_y))
            model_prediction.append(fitted_model_pipeline[i].predict(self.test_x))
            #Evaluating accuracy, balanced accuracy, area-under-curve and f1 score of each model
            model_accuracy.append(round(accuracy_score(self.test_y, model_prediction[i]), 5))
            model_b_accuracy.append(round(balanced_accuracy_score(self.test_y, model_prediction[i]), 5))
            model_roc_auc.append(round(roc_auc_score(self.test_y, model_prediction[i]), 5))
            model_f1.append(round(f1_score(self.test_y, model_prediction[i]), 5))
            i+= 1

        #Printing the metric arrays for each model using prettytable library
        t = PrettyTable()        
        i = 0
        t.field_names = ['Model', 'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1-Score']
        for model in model_pipeline:
            #Converting each model to a string, to allow the 'sortby' function to work
            model_str = ' '.join(map(str, [model]))
            t.add_row([model_str, model_accuracy[i], model_b_accuracy[i], model_roc_auc[i], model_f1[i]])
            i+= 1
        t.align = 'r'
        t.sortby = 'Accuracy'
        t.reversesort=True
        print(t)
        
        #Determining the most accurate model according to the accuracy
        best_model = model_accuracy.index(max(model_accuracy))
        return deepcopy(fitted_model_pipeline[best_model]) 

