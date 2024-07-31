from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_model(model_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "Logistic Regression":
        model = LogisticRegression(multi_class='ovr')  # Specify multi_class parameter
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    # Handle binary and multiclass ROC AUC
    if len(y.unique()) == 2:
        # Binary classification
        metrics = classification_report(y_test, y_pred, output_dict=True)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob[:, 1])
    else:
        # Multiclass classification
        metrics = classification_report(y_test, y_pred, output_dict=True)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

    return model, metrics
