from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score,confusion_matrix

def train_test(X_train, X_test, y_train, y_test):

    #classifier
    clf1 = RandomForestClassifier(n_estimators = 20, max_depth = 10)
    
    print('Starting training')
    
    # alternative metric to optimize over grid parameters: AUC
    clf1.fit(X_train, y_train)
    predict_proba = clf1.predict_proba(X_test)[:,1]
    
    print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
