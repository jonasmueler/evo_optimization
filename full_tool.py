import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# use glob to get all the csv files 
def dataReader():
    """
    

    Returns
    -------
    data : pd.DataFrame
        data with missing values.

    """
    path = os.getcwd() + "/data"
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    
    # integrate read json here
  
    # loop over the list of csv files
    for f in csv_files:
      # read the csv file
      data = pd.read_csv(f, on_bad_lines="skip")
      
    return data 
      
#df = dataReader()
def healthCheck(df):
    """

    Parameters
    ----------
    path : string
        Path to file with csv data.

    Returns
    -------
    list
        Ascending variables and number of missing values.

    """
    
    missings = np.sum(df.isnull()).sort_values()
    missings_perc = np.sum(missings)/(len(df.iloc[1,:]) * len(df.iloc[:,1]))
    missings_perc = round(missings_perc * 100)
    print(f"Data contains {missings_perc}% missing values")
    missings = missings[~(missings==0)]
    
    ## missings list
    missings_list = missings.index.values
    print("\nList of ascending missing value amounts in data columns: \n")
    missingName = []
    missingsNumber = []
    for i in range(len(missings_list)):
        missingName.append(missings_list[i])
        missingsNumber.append(missings.values[i])
        print([missings_list[i], missings.values[i]])
    
    if len(missingName) <= 80:
        ## plot missings 
        plt.bar(missingName, missingsNumber,color = "darkred")
        plt.title("Number of missing values by variable", fontsize = 18)
        plt.ylim([0, len(df.iloc[:,0])]) ## force realistic axis!
        plt.text(0,len(df.iloc[:,0])- (len(df.iloc[:,0])/10), f"{missings_perc}% Missing Values", fontsize = 12)
        plt.ylabel('Number', fontsize=18)
        plt.xticks(rotation=75)
        
    elif len(missingName) >= 80:
        print("Number of variables is too large to display column names on x-axis.\n")
        ## plot missings 
        plt.bar(np.arange(0,len(missingName), 1),np.array(missingsNumber))
        plt.title("Number of missing values by variable", fontsize = 18)
        plt.ylabel('Number', fontsize=18)
        plt.ylim([0, len(df.iloc[:,0])]) ## force realistic axis!
        plt.text(0,len(df.iloc[:,0])- (len(df.iloc[:,0])/20), f"{missings_perc}% Missing Values", fontsize = 12)
        
        
    
    ## save to a pdf file 
    plt.savefig("missing_values.png", format="png", bbox_inches="tight")
    plt.show()
        
    
#healthCheck(df)

def recoder(column, predictors, inverse = False, transformers = None):
    """
    recodes string variables into factors for classification 

    Parameters
    ----------
    columns : np.array
    predictors : np.array
    
    Returns
    -------
    List of recoded data and label codings.

    """
    if inverse == False:
        transformers = []
        le = preprocessing.LabelEncoder() 
        le.fit(column)
        transformers.append(le)
        column = le.transform(column)
        # transform rest of data
        for i in range(len(predictors[0,:])):
            le = preprocessing.LabelEncoder() 
            le.fit(predictors[:,i])
            transformers.append(le)
            predictors[:,i] = le.transform(predictors[:,i])
            
        return [column, predictors, transformers]
    if inverse: 
        column = transformers[0].inverse_transform(column.astype(np.int64))
        for t in range(len(predictors[0,:])):
            le = transformers[t+1]
            col = predictors[:,t].astype(np.int64)
            predictors[:,t] = le.inverse_transform(col)
            
        return [column, predictors, None]

def correlationPlot(df, columnName):
    """
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Data.
    columnName : string
        String of column to be predicted.

    Returns
    -------
    Correlation plot.
    
    """
    column = df[columnName] # column  for training

    
    
    # create predictors
    predictors = df.drop([columnName], axis = 1)
    names = predictors.columns # names for plot
    toBeDeleted = []
    def helper(x):
        if x.isna().sum()/len(x) > 0.10:
            toBeDeleted.append(x.name)
        return toBeDeleted
            
    df.apply(helper, axis=0)
    df = df.drop(toBeDeleted, axis = 1)
    
    # impute missings in predictors 
    imputer = SimpleImputer(strategy="most_frequent") 
    imputer.fit(predictors)
    predictors = imputer.transform(predictors)
    
    # get indices of missings in column 
    idx = column[column.isnull()].index.tolist()
    
    # convert data to numpy 
    column = np.delete(np.array(column), idx)
    predictors = np.delete(np.array(predictors), idx, axis = 0)
    
    
    # recode data  into categorical   
    data = np.c_[column, predictors]
    result = recoder(data[:,0], data[:,1:])
    column = result[0]
    predictors = result[1]


    # get correlations to target value
    correlations = []
    for i in range(len(predictors[1,:])):
            cor = stats.spearmanr(column, predictors[:,i])[0] 
            correlations.append(cor)

    plt.bar(names, correlations)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(rotation=75)
    plt.title("Correlation to predictor variable: PITEM_REVISION_ID", fontsize = 12)
    
    ## save to a pdf file 
    plt.savefig("correlations.png", format="png", bbox_inches="tight")
    plt.show()
    
    

#correlationPlot(df, "PITEM_ID")

def imputer(df, columnName):
    """
    
    Parameters
    ----------
    df : pd.DataFrame
        Input Data.
    columnName : string
        String of column to be predicted.

    Returns
    -------
    data column with imputed values.
    
    """
    
    
    column = df[columnName] # column  for training
    columnOriginal = column # column used for imputation
    
    
    # create predictors
    predictors = df.drop([columnName], axis = 1)
    toBeDeleted = []
    def helper(x):
        if x.isna().sum()/len(x) > 0.10:
            toBeDeleted.append(x.name)
        return toBeDeleted
            
    df.apply(helper, axis=0)
    df = df.drop(toBeDeleted, axis = 1)
    
    # impute missings in predictors 
    imputer = SimpleImputer(strategy="most_frequent") 
    imputer.fit(predictors)
    predictors = imputer.transform(predictors)
    predictorsOriginal = predictors.copy()
    
    # for imputation 
    imputer = SimpleImputer(strategy="most_frequent") 
    imputer.fit(np.array(column).reshape(-1, 1))
    columnOriginal = imputer.transform(np.array(column).reshape(-1, 1))
    
    # get indices of missings in column 
    idx = column[column.isnull()].index.tolist()
    
    # convert data to numpy 
    column = np.delete(np.array(column), idx)
    predictors = np.delete(np.array(predictors), idx, axis = 0)
    
    
    # recode data  into categorical   
    data = np.c_[column, predictors]
    result = recoder(data[:,0], data[:,1:])
    column = result[0]
    predictors = result[1]
    predictors = np.array(predictors, dtype = "float64")
    
    
    # stopp if data only contains unique values as classes -> can not be predicted correctly
    ## heuristic: only train if length of unique classes is bigger than 4/5 of data length
    print(round(len(column)*0.2))
    if len(np.unique(column)) > round(len(column)*0.2):
        sys.exit("Not enough examples in classes to create classification model correctly")
        
    
    print(len(np.unique(column)))
    #train decision tree model
    #clf = ExtraTreesClassifier(n_estimators = 10, criterion = "entropy")
    #clf = MLPClassifier(hidden_layer_sizes = (50,50,50, 50, 50, 50, 50,50, ), alpha = 0.01, verbose = True)
    #scores = cross_val_score(clf, predictors, column, cv=5, n_jobs = -1, verbose = 2)
    #print(f"mean performance of model {np.mean(scores)}")
    
    # fit algorithm 
    clf = ExtraTreesClassifier(n_estimators = 1000, criterion = "entropy", verbose = 2)
    #clf = DecisionTreeClassifier()
    #clf = MLPClassifier(hidden_layer_sizes = (50,50,50, 50, 50, 50, 50,50, ), alpha = 0.1, 
    #                   max_iter=10000, verbose = 2)
    #clf.fit(predictors, column)
    evolutionaryOptimization(200, 500, predictors, column)
    
    ########
    # impute missing values
    data = np.c_[columnOriginal, predictorsOriginal]
    result = recoder(data[:,0], data[:,1:])
    columnOriginal = result[0]
    predictorsOriginal = result[1]
    transformersOriginal = result[2]
    #predictorsOriginal = np.array(predictorsOriginal, dtype = "float64")
    
    
    columnOriginal[idx] = clf.predict(predictorsOriginal[idx])
    
    

    # recode into original format 
    res = recoder(columnOriginal, predictorsOriginal, inverse = True, transformers = transformersOriginal)
    
    return res[0]

    
#print(imputer(df, "PITEM_ID"))


#### evolutionary optimization 

def createPopulation(n):
    """
    

    Parameters
    ----------
    n : int
        Number of Individuals in Population.

    Returns
    -------
    list of list 
        Model parameters

    """
    criterion = ["gini", "entropy"]#, "log_loss"]
    max_features = ["sqrt", "log2"]
    splitter = ["best", "random"]
    
    ## leaf minimma at default, optimize upper bounds of parameters, specify priors 
    population = []
    for i in range(n):
        ind = {
            "n_estimators": np.random.randint(1,10, 1)[0],
            "splitter": np.random.choice(splitter, size=1)[0],
            "criterion": np.random.choice(criterion, size=1)[0],
            "max_depth": np.random.randint(1,10, 1)[0],
            "max_features": np.random.choice(max_features, 1)[0],
            "ccp_alpha" : np.random.exponential(0.5,1)[0]
            }
        population.append(ind)
    
    print("population generation successful!")
        
    return population

#print(createPopulation(10)[1]["n_estimators"])
    

def tournament(pop,n, predictors, column, average):
    
    
    """
        Parameters
        ----------
        players : list of dict
            parameters of agents.
        average : list of float
                running list of mean improvement of permance over generations
    
        Returns
        -------
                dict.
                winning parameter set.
    """
      
    for z in range(n):
        # set up models
        agents = np.random.randint(1,n, 4)
        agent1 = DecisionTreeClassifier(#n_estimators = pop[agents[0]]["n_estimators"],
                                        splitter = pop[agents[0]]["splitter"], 
                                      criterion = pop[agents[0]]["criterion"], 
                                      max_depth =  pop[agents[0]]["max_depth"],
                                      max_features = pop[agents[0]]["max_features"],
                                      #ccp_alpha = pop[agents[0]]["ccp_alpha"],
                                      #n_jobs = -1)
                                      )
        
        agent2 = DecisionTreeClassifier(#n_estimators = pop[agents[1]]["n_estimators"], 
                                        splitter = pop[agents[1]]["splitter"],
                                      criterion = pop[agents[1]]["criterion"], 
                                      max_depth =  pop[agents[1]]["max_depth"],
                                      max_features = pop[agents[1]]["max_features"],
                                      #ccp_alpha = pop[agents[1]]["ccp_alpha"],
                                      #n_jobs = -1)
                                      )
                                                      
        agent3 = DecisionTreeClassifier(#n_estimators = pop[agents[2]]["n_estimators"], 
                                        splitter = pop[agents[2]]["splitter"],
                                      criterion = pop[agents[2]]["criterion"], 
                                      max_depth =  pop[agents[2]]["max_depth"],
                                      max_features = pop[agents[2]]["max_features"],
                                      #ccp_alpha = pop[agents[2]]["ccp_alpha"],
                                      #n_jobs = -1)
                                      )
        
        agent4 = DecisionTreeClassifier(#n_estimators = pop[agents[3]]["n_estimators"],
                                        splitter = pop[agents[3]]["splitter"],
                                      criterion = pop[agents[3]]["criterion"], 
                                      max_depth =  pop[agents[3]]["max_depth"],
                                      max_features = pop[agents[3]]["max_features"],
                                      #ccp_alpha = pop[agents[3]]["ccp_alpha"],
                                      #n_jobs = -1)
                                      )
        
        
        ## get train and testset
        criterion = np.random.uniform(0,1,1)[0]
        cut_off = round(criterion*len(column))
        
        pred_train = predictors[0:cut_off,:]
        col_train = column[0:cut_off]
        
        pred_test = predictors[cut_off:len(predictors),:]
        col_test = column[cut_off:len(predictors)]
        
        
        ## fit models
        res = []
        agent1.fit(pred_train, col_train)
        pred = agent1.predict(pred_test)
        acc = accuracy_score(col_test, pred)
        res.append(acc)
        
        agent2.fit(pred_train, col_train)
        pred = agent2.predict(pred_test)
        acc = accuracy_score(col_test, pred)
        res.append(acc)
        
        agent3.fit(pred_train, col_train)
        pred = agent3.predict(pred_test)
        acc = accuracy_score(col_test, pred)
        res.append(acc)
        
        agent4.fit(pred_train, col_train)
        pred = agent4.predict(pred_test)
        acc = accuracy_score(col_test, pred)
        res.append(acc)
        
        # max out 
        Max = np.argmax(res)
        average.append(res[Max])
        pop[z] = pop[agents[Max]]
        
        #print("tournament: ", z, "done")
    
    #print("Average Performance: ", np.mean(average))

    return [pop, average] 



def mutate(pop, pPopMut):
    """
    
    Parameters
    ----------
    pop : list of dict
        population to be mutated.

    Returns
    -------
        list of dict
        mutated population

    """
    idx = np.random.randint(1,len(pop), round(pPopMut*100))  
    criterion = ["gini", "entropy"]#, "log_loss"]
    max_features = ["sqrt", "log2"]
    
    for i in idx:
        gene = np.random.randint(1,5,1) # p(mutate) = 0.2
        if gene == 1:
            pop[i]["n_estimators"] = pop[i]["n_estimators"] + np.random.randint(-10,10,1)[0]
            
            if pop[i]["n_estimators"] <= 0:
               pop[i]["n_estimators"] = np.random.randint(1,100, 1)[0]
                
        
        if gene == 2:
            pop[i]["criterion"] = np.random.choice(criterion, size=1)[0]
        
        if gene == 3:
            pop[i]["max_depth"] = pop[i]["max_depth"] + np.random.randint(-10,10,1)[0]
            
            if pop[i]["max_depth"] <= 0:
               pop[i]["max_depth"] = np.random.randint(1,1000, 1)[0]
               
        if gene == 4:
            pop[i]["max_features"] =  np.random.choice(max_features, size=1)[0]
        
        if gene == 5:
            pop[i]["ccp_alpha"] = pop[i]["ccp_alpha"] + np.random.normal(0,0.05, 1)[0]
            if pop[i]["ccp_alpha"] <= 0: 
                np.random.exponential(0.5,1)[0]
                
            
    return pop
            
            
                
def evolutionaryOptimization(n, nGen, predictors, column):
    """
    

    Parameters
    ----------
    n : int
        size of population.
    nGen : int
        number of generations.
    predictors: np.array
        features of models
    column: np.array 
        to be predicted column

    Returns
    -------
        list of dict
        optimized model population

    """
    
    pop = createPopulation(200)
    average = []
    for i in range(nGen):
        res = tournament(pop, n, predictors, column, average)
        pop = res[0]
        average = res[1]
        print("generation: ", i, "done! ", "running average performance: ", np.mean(average))
        
        pop = mutate(pop, 0.1)
    
###############################################################################################################
## put all together in one function

def dataCleaner(selectedCol, helthCheck = False, correlation = False):
    """
    
    Parameters
    ----------
    df : pd.dataframe
        dataset with predictors and criterion.
    selectedCol : string
        column to be predicted.

    Returns
    -------
    Information plots and data with imputed values.

    """
    
    df = dataReader()
    if healthCheck:
        healthCheck(df)
    if correlation:
        correlationPlot(df, selectedCol)
    column = imputer(df, selectedCol)
    
    return column

print(dataCleaner("PITEM_REVISION_ID"))
    
    
    
    


    


