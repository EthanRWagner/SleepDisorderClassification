import numpy as np
import pandas as pd
import itertools as itt

def preprocessing(X):
    for ix in range(len(X)):
        # GENDER DOES NOT NEED PREPROCESSING
        # AGE
        X.loc[ix, 'Age'] = int((X.loc[ix,'Age']/10).astype(str)[0])*10
        
        # SLEEP DURATION
        sleep_dur = X.iloc[ix]['Sleep Duration']
        if sleep_dur%1 > 0.5:
            X.loc[ix, 'Sleep Duration'] = int((X.loc[ix, 'Sleep Duration'] // 1 + 1).astype(str)[0])
        else:
            X.loc[ix, 'Sleep Duration'] = int((X.loc[ix, 'Sleep Duration'] // 1).astype(str)[0])
            
        # QUALITY OF SLEEP DOES NOT NEED PREPROCESSING
        
        # PHYSICAL ACTIVITY LEVEL
        X.loc[ix, 'Physical Activity Level'] = int((X.loc[ix, 'Physical Activity Level']/10).astype(str)[0])*10
        
        # STRESS LEVEL DOES NOT NEED PREPROCESSING
        # BMI CATEGORY DOES NOT NEED PREPROCESSING
        # BLOOD PRESSURE
        bp = [int(x) for x in X.loc[ix, 'Blood Pressure'].split("/")]
        if (bp[0] < 120 and bp[1] < 80):
            X.loc[ix, 'Blood Pressure'] = "Normal"
        elif ((bp[0] >= 120 and bp[0] <= 129) and bp[1] < 80):
            X.loc[ix, 'Blood Pressure'] = "Elevated"
        elif ((bp[0] >= 130 and bp[0] <= 139) or (bp[1] >= 80 and bp[1] <= 89)):
            X.loc[ix, 'Blood Pressure'] = "Stage 1"
        elif ((bp[0] >= 140 and bp[0] <= 180) or (bp[1] >= 90 and bp[1] <= 120)):
            X.loc[ix, 'Blood Pressure'] = "Stage 2"
        elif (bp[0] > 180 or (bp[1] > 120)):
            X.loc[ix, 'Blood Pressure'] = "Crisis"
        else:
            X.loc[ix, 'Blood Pressure'] = "Unclassified"
        
        # HEART RATE
        hr = X.loc[ix, 'Heart Rate']
        if hr > 100:
            X.loc[ix, 'Heart Rate'] = "High"
        elif hr < 60:
            X.loc[ix, 'Heart Rate'] = "Low"
        else:
            X.loc[ix, 'Heart Rate'] = "Normal"
            
        # DAILY STEPS
        X.loc[ix, 'Daily Steps'] = int((X.loc[ix, 'Daily Steps']/1000).astype(str)[0])*1000
        
    # SLEEP DISORDER
    X["Sleep Disorder"] = X["Sleep Disorder"].fillna("None")
    
def variable_importance(X,t,seeds):
    importances = pd.Series(np.zeros((X.shape[1],)),index=X.columns)

    l_seeds = len(seeds)
    for seed in seeds:
        w,X_test,t_test,results = train(X,t,seed=seed)
        
        i = 0
        w_abs = np.sqrt(w**2)
        max_in_seed = max(w_abs)
        for weight_i in w_abs:
            importances.iloc[i] += weight_i / (max_in_seed*l_seeds)       
            i+=1
        
    return importances

def compute_priors(y):
    total_samples = len(y)
    class_counts = y.value_counts()
    priors = {}

    for class_label, count in class_counts.items():
        class_name = f"{y.name}={class_label}"
        prior = count / total_samples
        priors[class_name] = prior

    return priors

###Calculate a certain class conditional (in this case P(Female|Sleep Apnea)
def specific_class_conditional(x,xv,y,yv):
    prob = None
    total_samples_yv = y[y == yv].count()
    samples_xv_and_yv = x[(x == xv) & (y == yv)].count()
    prob = samples_xv_and_yv / total_samples_yv
    return prob

###Calculate all class conditionals
def class_conditional(X,y):
    probs = {}
    for feature in X.columns:
        for feature_value in X[feature].unique():
            for class_label in y.unique():
                key = f"{feature}={feature_value}|{y.name}={class_label}"
                prob = specific_class_conditional(X[feature], feature_value, y, class_label)
                probs[key] = prob
    return probs

### Calculate posterior for a given sample
def posteriors(probs, priors, x):
    post_probs = {}
    evidence = 0.0
    not_found = 0

    # Calculate the evidence
    for sleep_label in priors:
        prior = priors[sleep_label]
        likelihood = 1.0
        for feature_label in x.index:
            feature_value = x[feature_label]
            cond_key = f"{feature_label}={feature_value}|{sleep_label}"
            if cond_key in probs:
                likelihood *= probs[cond_key]
        evidence += likelihood * prior

    # Calculate the posterior probabilities
    for sleep_label in priors:
        key = f"{sleep_label}|"
        posterior = priors[sleep_label]
        for feature_label in x.index:
            feature_value = x[feature_label]
            key += f"{feature_label}={feature_value},"
            cond_key = f"{feature_label}={feature_value}|{sleep_label}"
            if cond_key in probs:
                posterior *= probs[cond_key]
            else:
                not_found = 1
        if not_found == 0:
            if evidence ==0:
                evidence = 1
            post_probs[key[:-1]] = posterior / evidence
        else:
            post_probs[key[:-1]] = 0.5

    return post_probs

def train_test_split(X, y, test_frac=0.5):
    np.random.seed(0)
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs, :]
    y = y.iloc[inxs]
    
    split_idx = int(np.round(len(y) * (1 - test_frac)))
    Xtrain = X.iloc[:split_idx, :]
    ytrain = y.iloc[:split_idx]
    Xtest = X.iloc[split_idx:, :]
    ytest = y.iloc[split_idx:]
    
    return Xtrain, ytrain, Xtest, ytest

def prediction_accuracy(Xtrain,ytrain,Xtest,ytest):
    accuracy = None
    
    #Calculate Priors
    priors = compute_priors(ytrain)
    
    #Calculate Likelihood
    probs=class_conditional(Xtrain,ytrain)
    
    #Begin Testing
    correct_predictions=0
    total_predictions=len(ytest)
    
    for i in range(len(Xtest)):
        x=Xtest.iloc[i]
        true_class= ytest.iloc[i]
        
        #Calculate posterior probabilities
        post_probs: Dict[str, float] = posteriors(probs, priors, x)
        
        #Get predicted value for class
        predicted_class_str = max(post_probs, key=post_probs.get)
        predicted_class = (predicted_class_str.split("|")[0].split("=")[1])
        
        if predicted_class == true_class:
            correct_predictions += 1
            
    accuracy = correct_predictions/ total_predictions
    return accuracy

def test_based_FIP(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = prediction_accuracy(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            
            #calculate the accuracy of the current permutation
            perm_accuracy= prediction_accuracy(Xtrain, ytrain, Xtest2, ytest)
            
            #calculate the importance of the feature
            feature_importance = orig_accuracy - perm_accuracy
            importances[col]+= feature_importance
            
        importances[col] = importances[col]/npermutations
    return importances

def optimization_with_ranking(rankings, X, y, test_frac=0.5):
    
    accuracies = {}
    
    for ix in range(len(rankings)):
        X_copy = X.copy()
        X_new = X_copy.drop(rankings[ix+1:], axis=1)
        # print(X_new.head())
        Xtrain, ytrain, Xtest, ytest = train_test_split(X_new, y, test_frac)
        accuracies[str(ix+1) + " top features"] = prediction_accuracy(Xtrain, ytrain, Xtest, ytest)
        
    return accuracies

def optimization_with_random(X, y, test_frac=0.5, iterations=100):
    np.random.seed(0)
    accuracies = {}
    features = X.columns
    nFeatures = len(features)
    
    tot_combs = []
    for it in range(nFeatures):
        combs = list(itt.combinations(features, it+1))
        tot_combs = tot_combs + combs
    
    for ix in range(iterations):
        np.random.shuffle(tot_combs)
        drop_set = list(tot_combs[0])
        # print(drop_set)
        # break
        X_copy = X.copy()
        X_new = X_copy.drop(drop_set, axis=1)
        key = ''
        for feature in X_new.columns:
            # print(drop_set)
            # print(feature)
            # break
            key = key + feature + ', '
        key = key[:-2]
        # print(X_new.head())
        Xtrain, ytrain, Xtest, ytest = train_test_split(X_new, y, test_frac)
        accuracies[key] = prediction_accuracy(Xtrain, ytrain, Xtest, ytest)
        
    return accuracies
        