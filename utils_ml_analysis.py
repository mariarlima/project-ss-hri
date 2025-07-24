import os
import spacy
import pandas as pd
from ideadensity import depid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from scipy.stats import loguniform, uniform, randint
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from scipy.stats import loguniform, sem, t


# Function to clean and tokenize using spaCy
def clean_and_tokenize_spacy(transcript, lang='en'):
    # Load the spaCy English and Spanish models
    nlp_en = spacy.load("en_core_web_sm")
    doc = nlp_en(transcript)

    # Extract tokens (words) excluding punctuation and whitespace
    words = [token.text.lower() for token in doc if token.is_alpha]
    return words, doc

def calculate_depid(text):
    text = "This is a test of DEPID-R. This is a test of DEPID-R"
    density, word_count, dependencies = depid(text, is_depid_r=True)
    return density

# Function to calculate proportion of consecutive duplicate words using spaCy
def calculate_consecutive_duplicates_spacy(words):
    duplicate_count = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            duplicate_count += 1
    return duplicate_count / len(words) if len(words) > 0 else 0

def calculate_ling_nlp(transcript, lang, alpha=0.165):
    from collections import Counter
    import math
    # Clean and tokenize the transcript using spaCy
    words, doc = clean_and_tokenize_spacy(transcript, lang)

    # Total number of words (tokens)
    N = len(words)
    
    # Frequency of each unique word
    word_freq = Counter(words)
    
    # Total number of unique words (types)
    V = len(word_freq)
    
    # Number of hapax legomena (words that occur only once)
    V1 = sum(1 for count in word_freq.values() if count == 1)

    # Calculate Brunet's Index
    brunet_index = N ** (V ** (-alpha)) if N > 0 and V > 0 else 0
    brunet_index = round(brunet_index, 2)
    
    # Calculate Honoré's Statistic
    honors_statistic = (100 * math.log(N)) / (1 - (V1 / V)) if V > V1 and N > 0 else None
    honors_statistic = round(honors_statistic, 2) if honors_statistic else None
    
    # Calculate Type-Token Ratio (TTR) corrected for text length (CTTR)
    cttr = V / math.sqrt(2 * N) if N > 0 else 0
    cttr = round(cttr, 2)

    depid = calculate_depid(doc)
    depid = round(depid, 5)
    
    # Calculate proportion of consecutive duplicate words
    duplicate_proportion = calculate_consecutive_duplicates_spacy(words)
    duplicate_proportion = round(duplicate_proportion, 2)

    return brunet_index, honors_statistic, cttr, depid, duplicate_proportion


def create_models():
    """
    Function to create models
    """
    # TODO: Define models with some parameters (note hyperparameters will be tuned later)
    # Example of output: return {'lr': lr, 'svm': svm, 'rf': rf, 'nn': nn, 'xgboost': xgb_model}


def create_param_grids():
    """
    Create randomized hyperparameter grids for model selection.
    Returns:
        dict: Mapping of model identifiers to their parameter grids.
    e.g., 
    # Logistic Regression
    lr_params = {
        'C': loguniform(1e-5, 1e2),  # Inverse regularization strength
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']  # Compatible with both penalties
    }
    """
    
    # TODO: Define hyperparameter grids for each model you defined in create_models()
    # Example of output: return {'lr': lr_params, 'svm': svm_params, 'rf': rf_params, 'nn': nn_params, 'xgboost': xgb_params}


def crossval(model_name, model, params, X, y, feature_set):
    """
    Perform 10-fold CV with hyperparameter tuning and save the best model.
    Returns mean and std for multiple metrics.
    """

    # TODO: Define model_name_mapping = {} with the model names and their abbreviations
    model_name_mapping = {
    
    }

    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    scoring = {
        'sensitivity': make_scorer(recall_score),  # Sensitivity is the same as recall
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc',
        'accuracy': make_scorer(accuracy_score)
    }

    random_search = RandomizedSearchCV(
        model, params, 
        n_iter=50, cv=10, verbose=1, 
        random_state=42, n_jobs=-1, 
        scoring=scoring, refit='accuracy'  # Use accuracy or any metric of choice for refitting
    )
    random_search.fit(X, y)

    # Save the best model
    abbreviation = model_name_mapping.get(model_name, model_name.lower())
    filename = f"10fcv_{abbreviation}.pkl"
    joblib.dump(random_search.best_estimator_, f"./data/{feature_set}/{filename}")

    # Extract scores
    metrics = ['sensitivity', 'specificity', 'roc_auc', 'accuracy']
    scores = {'Model': model_name}
    
    for metric in metrics:
        best_index = random_search.best_index_
        mean = random_search.cv_results_[f'mean_test_{metric}'][best_index] * 100
        std = random_search.cv_results_[f'std_test_{metric}'][best_index] * 100
        scores[metric.capitalize()] = f"{mean:.1f} ({std:.1f})"
    
    return scores

def load_best_params(feature_set):
    """
    Load the best trained models from disk for a given feature set.
    Returns a dictionary of model names to fitted estimators.
    """

    # TODO: Define model_name_mapping = {} with the model names and their abbreviations
    model_name_mapping = {
    
    }

    best_hyperparams = {}

    # Load each saved model into the best_models dictionary
    for model_name, abbreviation in model_name_mapping.items():
        file_path = os.path.join("./data", feature_set, f"10fcv_{abbreviation}.pkl")
        if os.path.exists(file_path):
            print(file_path)
            best_hyperparams[model_name] = joblib.load(file_path)
    
    return best_hyperparams


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate the model on the given test set.
    Returns recall, specificity, ROC AUC, accuracy, and predicted probabilities.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    roc_auc = roc_auc_score(y_test, y_proba)
    return recall, specificity, roc_auc, accuracy, y_proba


def fit_and_evaluate_bootstrap_(best_hyperparams, X_train, y_train, X_test, y_test, n_repeats=10, confidence=0.95):
    """
    Evaluate each best model on unseen data with bootstrapped training samples.
    Returns evaluation metrics with confidence intervals and predicted probabilities.
    """

    evaluation_results = []

    # Store predicted probabilities across all bootstrap runs for each model
    bootstrap_probabilities = {model_name: [] for model_name in best_hyperparams.keys()}

    # Define metric names for reporting
    metric_names = ['Recall', 'Specificity', 'ROC-AUC', 'Accuracy']
    
    # Wrap each model with a StandardScaler pipeline
    model_dict = {
        model_name: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
        for model_name, clf in best_hyperparams.items()
    } 
    
    # Iterate over each model for evaluation
    for model_name, clf in model_dict.items():
        
        bootstrap_metrics = {'Recall': [], 'Specificity': [], 'ROC-AUC': [], 'Accuracy': []}

        for i in range(n_repeats):
            # Bootstrap resample from the training set
            X_train_r, y_train_r = resample(X_train, y_train, n_samples=len(X_train), random_state=i)

            clf[-1].random_state = i
 
            clf.fit(X_train_r, y_train_r)

            # Evaluate the model on the original test set
            r = evaluate_on_test(clf, X_test, y_test)
            scores, probabilities = r[:-1], r[-1]
            i += 1

            # Updated to ensure order 
            scores_dict = dict(zip(metric_names, scores))
            for metric, score in scores_dict.items():
                bootstrap_metrics[metric].append(score)

            # Store the probabilities for this iteration
            bootstrap_probabilities[model_name].append(probabilities)

        # Calculate the 95% CI for the mean of the metrics for this model
        metrics_ci = {}
        for metric, scores in bootstrap_metrics.items():
            mean_score = np.mean(scores)
            se = sem(scores)
            ci = se * t.ppf((1 + confidence) / 2, len(scores) - 1)
            metrics_ci[metric] = (mean_score, mean_score - ci, mean_score + ci)

        # Prepare the result dictionary
        result = {'Model': model_name}
        for metric, (mean, lower_ci, upper_ci) in metrics_ci.items():
            result[f'{metric} Mean'] = mean
            result[f'{metric} Lower CI'] = lower_ci
            result[f'{metric} Upper CI'] = upper_ci
            
        evaluation_results.append(result)

    return evaluation_results, bootstrap_probabilities


def extract_results_classif_test(df):
    recall_result = f"{round(df['Recall Mean'].values[0], 2)} ({round(df['Recall Lower CI'].values[0], 2)} - {round(df['Recall Upper CI'].values[0], 2)})"
    spec_result = f"{round(df['Specificity Mean'].values[0], 2)} ({round(df['Specificity Lower CI'].values[0], 2)} - {round(df['Specificity Upper CI'].values[0], 2)})"
    auc_result = f"{round(df['ROC-AUC Mean'].values[0], 2)} ({round(df['ROC-AUC Lower CI'].values[0], 2)} - {round(df['ROC-AUC Upper CI'].values[0], 2)})"
    acc_result = f"{round(df['Accuracy Mean'].values[0], 2)} ({round(df['Accuracy Lower CI'].values[0], 2)} - {round(df['Accuracy Upper CI'].values[0], 2)})"
    return recall_result, spec_result, auc_result, acc_result