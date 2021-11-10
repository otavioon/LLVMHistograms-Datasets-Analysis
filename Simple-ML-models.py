from sklearn import svm, datasets, linear_model, tree, neighbors

import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd

def main():

    datasets = []
    datasets.append( ("AnghaBestSeqsSBLP2021","OptZ-A") )
    datasets.append( ("AnghaLoops_2C","Loop-2") )
    datasets.append( ("AnghaLoops","Loop-4") )
    datasets.append( ("CodeNet-5","Algo-C") )
    datasets.append( ("CodeNetBestSeqsSBLP2021_merge_Nclasses","OptZ-C") )
    datasets.append( ("POJ-5","Algo-P") )
    datasets.append( ("POJBestSeqsSBLP2021_merge_2classes","OptZ-P-sp") )
    datasets.append( ("POJBestSeqsSBLP2021_merge_Nclasses","OptZ-P") )

    #evaluate_dataset("data/POJ-104.csv")
    #evaluate_dataset("data/POJ-104.csv", normalize=True)

    results = {}
    for ds_basename, ds_name in datasets:
        print("Fitting models for {} ({})".format(ds_basename, ds_name))
        results[ds_name] = evaluate_dataset(ds_name, "data/"+ds_basename+".training.csv", "data/"+ds_basename+".validation.csv", "data/"+ds_basename+".test.csv")

    print(results)

    for ds_name, ds_name_result in results.items():
        LR="?"
        if "LR" in ds_name_result: LR = ds_name_result["LR"]
        DT="?"
        if "DT" in ds_name_result: DT = ds_name_result["DT"]
        KNN="?"
        if "KNN" in ds_name_result: KNN = ds_name_result["KNN"]
        SVM="?"
        if "SVM" in ds_name_result: SVM = ds_name_result["SVM"]
        print("\pacc{"+ds_name+"}{Fig. ?}{"+"{:.1f}".format(LR)+"}{"+"{:.1f}".format(DT)+"}{"+"{:.1f}".format(KNN)+"}{"+"{:.1f}".format(SVM)+"}")
        


def eval_model(X,y,model):
    model_pred = model.predict(X)
    model_accuracy = accuracy_score(y, model_pred)
    model_f1 = f1_score(y, model_pred, average='weighted')
    return model_accuracy*100, model_f1*100


def train_model(X_train, X_val, y_train, y_val, model):
    #print("Evaluating SVC with: C={}, gamma={}, degree={}, kernel={}".format(C,gamma,degree,kernel))
    model = model.fit(X_train,y_train)
    acc, f1 = eval_model(X_val, y_val, model)
    return acc, f1, model

def search_best_SVM_model(X_train, X_val, y_train, y_val):
    best_params = None
    for kernel in ['poly', 'rbf']:
        for C in [0.1, 10.0, 100.0, 1000.0, 10000.0]:
            model = svm.SVC(kernel=kernel, gamma='scale', C=C, degree=3)
            acc, f1, model = train_model(X_train, X_val, y_train, y_val, model)
            if not best_params or best_params["Val acc"] < acc:
                best_params = { "Val acc": acc, "F1": f1, "params" : {"C":C, "kernel":kernel}, "model" : model}
    return best_params

def search_best_KNN_model(X_train, X_val, y_train, y_val):
    best_params = None
    for n_neighbors in [5, 10, 15, 20, 50]:
        for weights in ["uniform", "distance"]:
            # we create an instance of Neighbours Classifier and fit the data.
            model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            acc, f1, model = train_model(X_train, X_val, y_train, y_val, model)
            if not best_params or best_params["Val acc"] < acc:
                best_params = { "Val acc": acc, "F1": f1,  "params" : {"weights":weights, "n_neighbors":n_neighbors} , "model" : model}
    return best_params

def search_best_LR_model(X_train, X_val, y_train, y_val):
    best_params = None
    for penalty in ["l1", "elasticnet", "l2", "none"]:
        for solver in ["lbfgs"]:
        #for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
            for c in [5, 10, 20, 50, 70, 100]:
                for tol in [0.1, 0.5, 1.0]:
                    C=c/len(X_train.index)
                    model = linear_model.LogisticRegression(C=C, penalty="l1", solver="saga", tol=0.1)
                    acc, f1, model = train_model(X_train, X_val, y_train, y_val, model)
                    if not best_params or best_params["Val acc"] < acc:
                        best_params = { "Val acc": acc, "F1": f1,  "params" : {"C":C, "tol":tol, "penalty":penalty, "solver":solver} , "model" : model}
    return best_params

def search_best_SGDC_model(X_train, X_val, y_train, y_val):
    best_params = None
    for loss in ["log", "hinge", "modified_huber", "perceptron"]:
        for penalty in ["l1", "elasticnet", "l2"]:
            for max_iter in [50]:
                model = linear_model.SGDClassifier(loss=loss, penalty=penalty)
                acc, f1, model = train_model(X_train, X_val, y_train, y_val, model)
                if not best_params or best_params["Val acc"] < acc:
                    best_params = { "Val acc": acc, "F1": f1,  "params" : {"loss":loss, "penalty":penalty, "max_iter":max_iter} , "model" : model}
    return best_params


def evaluate_dataset(ds_name, train_filename, val_filename, test_filename, normalize=False):

    result = {}

    train_data = pd.read_csv(filepath_or_buffer=train_filename)
    X_train = train_data.iloc[0:,3:]
    y_train = train_data['class']
    val_data = pd.read_csv(filepath_or_buffer=val_filename)
    X_val = val_data.iloc[0:,3:]
    y_val = val_data['class']
    test_data = pd.read_csv(filepath_or_buffer=test_filename)
    X_test = test_data.iloc[0:,3:]
    y_test = test_data['class']

    # Normalize rows
    if normalize:
        X_train = X_train.div(X_train.sum(axis=1), axis=0)
        X_val   = X_val.div(X_val.sum(axis=1), axis=0)
        X_test  = X_test.div(X_test.sum(axis=1), axis=0)
        suffix = " (normalized)"
    else:
        suffix = "" 

    prefix = ds_name + ": "

    best_LR = search_best_LR_model(X_train, X_val, y_train, y_val)
    acc, f1 = eval_model(X_test, y_test, best_LR["model"])
    best_LR["Test acc"] = acc
    best_LR["Test f1"] = f1
    print(prefix+"Best LR model"+suffix+": Test acc:", acc, ":", best_LR)
    result["LR"] = acc

    acc, f1, dt_model = train_model(X_train, X_val, y_train, y_val, tree.DecisionTreeClassifier())
    best_DT = { "Val acc" : acc, "Val f1" : f1, "params" : "Default" , "model": dt_model}
    acc, f1 = eval_model(X_test, y_test, best_DT["model"])
    best_DT["Test acc"] = acc
    best_DT["Test f1"] = f1
    print(prefix+"Best DT model"+suffix+": Test acc:", acc, ":", best_DT)
    result["DT"] = acc

    #import graphviz
    #dot_data = tree.export_graphviz(dt_model, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("filename") 

    best_KNN = search_best_KNN_model(X_train, X_val, y_train, y_val)
    acc, f1  = eval_model(X_test, y_test, best_KNN["model"])
    best_KNN["Test acc"] = acc
    best_KNN["Test f1"] = f1
    print(prefix+"Best KNN model"+suffix+": Test acc:", acc, ":", best_KNN)
    result["KNN"] = acc

    best_SVM = search_best_SVM_model(X_train, X_val, y_train, y_val)
    acc, f1  = eval_model(X_test, y_test, best_SVM["model"])
    best_SVM["Test acc"] = acc
    best_SVM["Test f1"] = f1
    print(prefix+"Best SVM model"+suffix+": Test acc:", acc, ":", best_SVM)
    result["SVM"] = acc

    #best_SGDC = search_best_SGDC_model(X_train, X_val, y_train, y_val)
    #acc, _ = eval_model(X_train, X_test, y_train, y_test, best_SGDC["model"])
    #print(prefix+"Best SGDC model"+suffix+":", best_SGDC)

    return result

main()
