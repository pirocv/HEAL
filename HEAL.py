import time
import pandas as pd
import numpy as np

import argparse
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed, dump

import os

def run_single_trial(trial_data, total_folds):
    # Unpack data
    trial_idx, (train_ix, test_ix), X, y, scoring, lambda_candi, splits = trial_data

    trial_start_time = time.time()
    print(f"Processing fold {trial_idx} / {total_folds} with {splits} CV")

    # Extract train and test data
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

    # Setup inner CV for tuning hyperparameters
    inner_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=trial_idx)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear'))
    param_grid = {'logisticregression__C': 1 / np.array(lambda_candi)}

    # Fit the train data
    clf = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring=scoring, n_jobs=2, verbose=0)
    clf.fit(X_train, y_train)

    best_lambda = 1 / clf.best_estimator_.named_steps['logisticregression'].C

    # Get scores on test data
    predicted_probs = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    predicted_labels = (predicted_probs > 0.5).astype(int)
    ids = X_test.index  # Assuming X_test has an index that can be used as an identifier
    # Assert that no ids in the test set are in the train set
    assert len(set(train_ix).intersection(set(test_ix))) == 0
    predictions_with_ids = []
    for id, prob, label in zip(ids, predicted_probs, y_test):
        predictions_with_ids.append((id, prob, label, trial_idx))

    roc_auc = roc_auc_score(y_test, predicted_probs)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    auprc = average_precision_score(y_test, predicted_probs)

    scores = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'prc_auc': auprc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Show how much time we have left
    trial_end_time = time.time()  # End timing the current trial
    trial_duration = trial_end_time - trial_start_time

    print(f"    Fold {trial_idx} completed in {trial_duration:.2f} seconds")

    # Return scores, features of best model as well as predictions and best parameter
    return scores, predictions_with_ids, best_lambda

def nested_cross_val_with_features(merged_filtered, scoring, lambda_candi, splits, trials, tts, random_state):
    total_start_time = time.time()

    # Run a repeated stratified K folds for n trials and k folds (this will be the total of tests on different splits)
    all_tests = []
    total_cases = 0
    total_controls = 0
    for idx in range(trials):
        trial_data = merged_filtered.copy()
        X = trial_data.iloc[:,:-1]
        y = trial_data.iloc[:,-1]

        total_cases = y[y == 1].shape[0]
        total_controls = y[y == 0].shape[0]

        if tts:
            # Do a train_test_split instead of the StratifiedKFold
            X_train, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=(random_state + idx), stratify=y)
            train_ix = [X.index.get_loc(label) for label in X_train.index]
            test_ix = [X.index.get_loc(label) for label in X_test.index]
            cv_split = (train_ix, test_ix)
            trial_data = [(idx+1, cv_split, X, y, scoring, lambda_candi, splits)]
        else:
            outer_cv = StratifiedKFold(n_splits=splits, random_state=(random_state + idx), shuffle=True)
            trial_data = [(trial_idx+(idx*splits), cv_split, X, y, scoring, lambda_candi, splits)
                        for trial_idx, cv_split in enumerate(outer_cv.split(X, y), start=1)]

        all_tests.extend(trial_data)

    scores = []
    predictions_with_ids = []
    best_lambdas = []

    print(f"Running with Total cases: {total_cases}, Total controls: {total_controls}")
    total_folds = splits * trials
    if tts:
        total_folds = trials
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_trial)(data, total_folds) for data in all_tests
    )

    # Process results
    for trial_scores, trial_predictions_with_ids, trial_best_lambda in results:
        scores.append(trial_scores)
        predictions_with_ids.extend(trial_predictions_with_ids)
        best_lambdas.append(trial_best_lambda)

    # Scores and Predictions processing
    # Calculate mean and std for each metric
    metric_summaries = {metric: f"{np.mean([s[metric] for s in scores]):.3f} ± {np.std([s[metric] for s in scores]):.3f}" for metric in scores[0].keys()}
    all_cases = [total_cases] * len(scores)
    all_controls = [total_controls] * len(scores)
    scores_df = pd.DataFrame({**{metric: [score[metric] for score in scores] for metric in scores[0].keys()}, 'Cases': all_cases, 'Controls': all_controls})
    predictions_df = pd.DataFrame(predictions_with_ids, columns=['ID', 'PredictedProb', 'TrueLabel', 'Trial'])

    end_time = time.time()
    elapsed_time = end_time - total_start_time
    print(f"Nested CV completed in {elapsed_time:.2f} seconds")
    print(f"with Total cases: {total_cases}, Total controls: {total_controls}")

    return scores_df, predictions_df, best_lambdas, metric_summaries


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run HEAL with params')
    parser.add_argument('--file_path', type=str, required=True, help='Full path to file')
    parser.add_argument('--output', type=str, required=False, help='Output path', default=os.getcwd())
    parser.add_argument('--splits', type=int, required=False, help='Number of splits for CV (default 5)', default=5)
    parser.add_argument('--trials', type=int, required=False, help='Number of trials to run (default 1)', default=1)
    parser.add_argument('--l1', type=float, required=False, help='Lower bound of lambda candidates', default=1.0)
    parser.add_argument('--l2', type=float, required=False, help='Upper bound of lambda candidates', default=40.0)
    parser.add_argument('--lfidelity', type=int, required=False, help='Fidelity of linspace of lambda candidates', default=5)
    parser.add_argument('--scoring', type=str, required=False, help='Scoring metric to maximize (default: roc_auc)', default='roc_auc')
    parser.add_argument('--random_state', type=int, required=False, help='Random state to start from (default 42)', default=42)
    parser.add_argument('--tts', type=bool, required=False, help='Use train_test_split instead of StratifiedKFold for outer CV (default False)', default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Fix the file, load it
    merged_filtered = pd.read_csv(args.file_path, index_col=0, sep='\t')
    X = merged_filtered.iloc[:,:-1]
    y = merged_filtered.iloc[:,-1]

    # Fix the file name
    file_name = re.split(r'\.csv|\.tsv', args.file_path.split('/')[-1])[0]

    lambda_candi = np.linspace(args.l1, args.l2, args.lfidelity)

    # Main analysis
    print(f"Will perform Nested Cross val on {file_name} with {args.trials} repeats and {args.splits} outer CV and {args.splits} inner CV for {args.lfidelity} candidates resulting in {args.trials*args.splits*args.splits*args.lfidelity} total fits for {lambda_candi} lambda range")
    scores, predictions, best_lambdas, metric_summaries = nested_cross_val_with_features(
        merged_filtered,
        args.scoring,
        lambda_candi,
        splits=args.splits,
        trials=args.trials,
        tts=args.tts,
        random_state=args.random_state
    )

    # Try the final model on the full data
    final_best_lambda = np.mean(best_lambdas)
    final_model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=1/final_best_lambda, solver='liblinear'))
    final_model.fit(X, y)

    final_model_coefs = final_model.named_steps['logisticregression'].coef_[0]
    final_model_features = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': final_model_coefs
    })
    final_model_features.sort_values(by='Coefficient', ascending=False, inplace=True)

    # Store it
    ## Create top folder with configuration
    suffix = f'_{args.trials}trials_{args.splits}CV_{args.l1}to{args.l2}w{args.lfidelity}lambdarange_{args.tts}tts'
    path = args.output + '/' + file_name + suffix
    if not os.path.exists(path):
        os.makedirs(path)
    
    scores.to_csv(path + '/scores.csv')
    predictions.to_csv(path + '/predictions.csv', index=None)
    final_model_features.to_csv(path + '/final_model_features.csv', index=None)

    total_folds = args.splits * args.trials
    if args.tts:
        total_folds = args.trials

    # Create a summary pandas frame
    summary = pd.DataFrame({
        'File': [file_name],
        'Iterations': [args.trials],
        'Lambda range': [f'{args.l1} to {args.l2} with {args.lfidelity} fidelity'],
        'Splits': [args.splits],
        'Total folds': [total_folds],
        'Final Best Lambda': [final_best_lambda],
        **{f'{metric.capitalize()} (mean ± std)': [metric_summaries[metric]] for metric in metric_summaries.keys()}
    })

    # Nice print
    print("\n\n")
    [print(f'{metric.capitalize()}: {metric_summaries[metric]}') for metric in metric_summaries.keys()]
    print(f"\nFinal Best Lambda: {final_best_lambda:.2f}")
    # Store the summary
    summary.to_csv(args.output + f'/{file_name}{suffix}_model_summary.csv', index=None)

    ## Store the final model
    dump(final_model, f'{path}/final_model.joblib')


if __name__ == '__main__':
    main()

