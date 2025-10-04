import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None):
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'LR': LogisticRegression(max_iter=1000),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }
    clf = classifiers.get(approach, LinearDiscriminantAnalysis())
    clf.fit(train_x, train_y)

    
    pred = clf.predict_proba(test_x) if output_probability else clf.predict(test_x)
    
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def CSP_LDA(Train,Test,approach):
    csp = mne.decoding.CSP(n_components=10)
    train_x = Train.data.astype(np.float64)
    train_y = Train.label

    test_x = Test.data.astype(np.float64)
    test_y = Test.label

    train_x_csp = csp.fit_transform(train_x, train_y)
    test_x_csp = csp.transform(test_x)

    pred, model = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
    score = np.round(accuracy_score(test_y, pred), 5)

    return score*100







