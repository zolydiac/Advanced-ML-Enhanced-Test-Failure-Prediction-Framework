"""
Advanced ML-Enhanced Test Failure Prediction Framework
Research-Grade Prototype for University of Tokyo Masters Application

Author: Harrison King
Institution: University of Kent Canterbury
Research Area: Leveraging Machine Learning to Improve Automated Software Testing

This project explores how machine learning can help us predict when automated tests
might fail, reducing the time developers spend fixing broken test suites. The approach
combines traditional ML techniques with deep learning to analyze patterns in test
execution history and code changes.

The main idea: instead of just running tests and hoping they pass, we can use historical
data to predict which tests are likely to fail and why. This helps teams focus their
maintenance efforts where they're needed most.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import re
import warnings
from scipy import stats
from collections import defaultdict
warnings.filterwarnings('ignore')

# Setting seeds for reproducible results - important for research integrity
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class GitHubDataCollector:
    """
    Collects real data from GitHub repositories to understand how code changes
    affect test stability. This gives us authentic patterns instead of just
    synthetic data.

    The key insight: tests don't fail randomly - they fail in patterns related
    to code changes, developer behavior, and timing.
    """

    def __init__(self, github_token=None):
        self.github_token = github_token
        self.headers = {'Authorization': f'token {github_token}'} if github_token else {}
        self.rate_limit_remaining = 5000

    def fetch_repository_commits(self, repo_url, days_back=365):
        """
        Grab commit history from a GitHub repo. Each commit tells us something
        about how the codebase is evolving, which helps predict test stability.

        We're looking for patterns like:
        - Frequent commits often correlate with test instability
        - Certain authors might write more fragile tests
        - Time of day/week affects test reliability
        """

        # Parse the GitHub URL to get owner and repo name
        repo_parts = repo_url.replace('https://github.com/', '').split('/')
        owner, repo = repo_parts[0], repo_parts[1]

        # Only look at recent commits - older ones are less relevant
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits"

        commits = []
        page = 1

        # GitHub paginates results, so we need to fetch multiple pages
        while page <= 5:  # Limiting to 5 pages to avoid hitting API limits
            params = {
                'since': since_date,
                'page': page,
                'per_page': 100
            }
            try:
                response = requests.get(api_url, headers=self.headers, params=params, timeout=10)
                if response.status_code != 200:
                    break

                page_commits = response.json()
                if not page_commits:
                    break

                commits.extend(page_commits)
                page += 1

            except Exception as e:
                print(f"Had trouble fetching from GitHub API: {e}")
                break

        return commits

    def analyze_commit_patterns(self, commits):
        """
        Extract meaningful insights from commit data that might predict test failures.

        My hypothesis: certain types of commits (bug fixes, refactors, test changes)
        have different impacts on test stability. Let's capture those patterns.
        """
        commit_data = []

        for commit in commits:
            try:
                commit_info = {
                    'sha': commit['sha'][:8],  # Just the short hash for readability
                    'timestamp': pd.to_datetime(commit['commit']['author']['date']),
                    'message': commit['commit']['message'],
                    'author': commit['commit']['author']['name'],
                    # Note: Getting file change counts would require additional API calls
                    'files_changed': 0,
                    'additions': 0,
                    'deletions': 0
                }
                # Analyze commit messages for patterns that matter to testing
                message_lower = commit_info['message'].lower()

                # Test-related commits might indicate ongoing test maintenance issues
                commit_info['is_test_related'] = any(keyword in message_lower for keyword in
                                                     ['test', 'spec', 'junit', 'selenium', 'cypress', 'jest'])

                # Bug fix commits often correlate with test failures
                commit_info['is_bugfix'] = any(keyword in message_lower for keyword in
                                               ['fix', 'bug', 'error', 'issue', 'patch'])

                # Refactoring can break tests even when functionality stays the same
                commit_info['is_refactor'] = any(keyword in message_lower for keyword in
                                                 ['refactor', 'cleanup', 'improve', 'optimize'])
            commit_data.append(commit_info)

            except KeyError:
            # Some commits might have malformed data, just skip them
            continue

    return pd.DataFrame(commit_data)


class TestFeatureEngineering:
    """
    Transforms raw test data into features that machine learning models can use effectively.

    The art of feature engineering: taking messy real-world data and finding the signals
    that actually matter for prediction. This is where domain expertise really helps.
    """

    def __init__(self):
        self.temporal_window = 30  # Look at the last 30 days of activity
        self.scaler = StandardScaler()

    def create_temporal_features(self, df):
        """
        Time matters a lot in software testing. Tests that fail on Monday morning
        might be different from tests that fail on Friday afternoon.

        We're capturing patterns like:
        - Monday deployments often have more issues
        - Weekend deployments are riskier (less oversight)
        - Tests run at different times have different failure patterns
        """
        df_enhanced = df.copy()
        df_enhanced['timestamp'] = pd.to_datetime(df_enhanced.get('timestamp', datetime.now()))

        # Basic time features that often matter for test stability
        df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
        df_enhanced['hour_of_day'] = df_enhanced['timestamp'].dt.hour
        df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)

        # Rolling window features - looking at recent trends
        # The idea: a test that's been failing recently is more likely to fail again
        for window in [7, 14, 30]:
            df_enhanced[f'failure_rate_{window}d'] = (
                df_enhanced.get('test_failed', 0)
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # Code change velocity - how fast is the code changing?
        # Faster changes often mean higher test instability
        df_enhanced['code_churn_velocity'] = (
                df_enhanced.get('code_changes_last_week', 0) / 7.0
        )

        return df_enhanced

    def extract_test_complexity_metrics(self, test_code):
        """
        Analyze the actual test code to understand its complexity.
        More complex tests tend to be more fragile.

        This is based on software metrics research - things like cyclomatic
        complexity have been shown to correlate with code quality issues.
        """
        if not test_code:
            return {
                'cyclomatic_complexity': 1,
                'lines_of_code': 0,
                'assertion_count': 0,
                'xpath_complexity': 0,
                'wait_statements': 0
            }
        lines = test_code.split('\n')
        # Approximate McCabe complexity - counts decision points in code
        # More decision points = more ways for test to go wrong
        decision_keywords = ['if', 'elif', 'while', 'for', 'try', 'except', 'case']
        complexity = 1  # Start at 1 for the base path
        for line in lines:
            for keyword in decision_keywords:
                complexity += line.count(keyword)
                # Test-specific metrics that matter for UI testing
                assertion_count = sum(line.count('assert') for line in lines)
                xpath_complexity = sum(len(re.findall(r'//\w+', line)) for line in lines)
                wait_statements = sum(line.count('wait') + line.count('sleep') for line in lines)

                return {
                    'cyclomatic_complexity': complexity,
                    'lines_of_code': len([l for l in lines if l.strip()]),
                    'assertion_count': assertion_count,
                    'xpath_complexity': xpath_complexity,
                    'wait_statements': wait_statements
                }


class DeepTestPredictor(nn.Module):
    """
    A neural network designed specifically for test failure prediction.

    Why deep learning? Traditional ML models struggle with complex interactions
    between features. Neural networks can capture subtle patterns that might
    indicate a test is about to become unstable.

    The architecture: multiple hidden layers with dropout to prevent overfitting.
    We're essentially learning a function that maps test characteristics to
    failure probability.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout_rate=0.3):
        super(DeepTestPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Building the network layer by layer
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            # Linear transformation
            layers.append(nn.Linear(current_dim, hidden_dim))
            # Batch normalization helps with training stability
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation - simple but effective
            layers.append(nn.ReLU())
            # Dropout prevents overfitting to training data
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Final output layer - two classes (pass/fail)
        layers.append(nn.Linear(hidden_dim, 2))
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMTestPredictor(nn.Module):
    """
    LSTM model for capturing sequential patterns in test executions.

    The insight: test failures aren't independent events. If a test failed yesterday,
    that affects the probability it fails today. LSTMs are good at learning these
    temporal dependencies.

    Think of it like predicting weather - past conditions help predict future ones.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMTestPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer - this is where the temporal magic happens
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Classification head - turns LSTM output into pass/fail prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the final hidden state for classification
        last_hidden = hn[-1]
        output = self.classifier(last_hidden)
        return output


class EnsemblePredictor:
    """
    Combines multiple different models to get better predictions than any single model.

    The ensemble approach: different models capture different aspects of the problem.
    Random Forest might catch feature interactions, Neural Networks might find
    non-linear patterns, and Gradient Boosting might handle outliers well.

    Together, they're more robust than any individual approach.
    """

    def __init__(self):
        # Diverse set of models - each has different strengths
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,  # Enough trees for good performance
                max_depth=15,  # Deep enough to capture complexity
                random_state=42,  # Reproducible results
                class_weight='balanced'  # Handle imbalanced data
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'deep_nn': None,  # Will be initialized during training
            'lstm': None  # Will be initialized during training
        }

        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        self.is_trained = False
        self.performance_metrics = {}

    def train_sklearn_models(self, X, y):
        """
        Train the traditional machine learning models with proper validation.

        Using cross-validation here because it gives us more reliable performance
        estimates than a single train/test split. The idea is to train and test
        on different data splits multiple times.
        """
        try:
            # Preprocessing: scale features and select the most informative ones
            X_scaled = self.scaler.fit_transform(X)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)

            cv_scores = {}

            for name, model in self.models.items():
                if model is None:  # Skip the deep learning models for now
                    continue

                try:
                    # 5-fold cross-validation - standard in ML research
                    scores = cross_val_score(
                        model, X_selected, y,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring='roc_auc'  # AUC is good for imbalanced classification
                    )

                    cv_scores[name] = {
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'scores': scores.tolist()
                    }

                    # Train on the full dataset for final model
                    model.fit(X_selected, y)

                except Exception as e:
                    print(f"Issue training {name}: {e}")
                    # Provide reasonable defaults if training fails
                    cv_scores[name] = {
                        'mean_score': 0.5,
                        'std_score': 0.0,
                        'scores': [0.5] * 5
                    }

            return cv_scores
        except Exception as e:
            print(f"Problem with traditional model training: {e}")
            return {
                'random_forest': {'mean_score': 0.5, 'std_score': 0.0, 'scores': [0.5] * 5},
                'gradient_boosting': {'mean_score': 0.5, 'std_score': 0.0, 'scores': [0.5] * 5}
            }

    def train_deep_models(self, X, y):
        """
        Train the neural network models. This is where things get interesting -
        deep learning can capture patterns that traditional ML might miss.

        The training process: show the network examples of tests and their outcomes,
        and let it learn the underlying patterns through backpropagation.
        """
        try:
            # Use the same preprocessing as the sklearn models
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)

            # Convert to PyTorch tensors - deep learning frameworks love tensors
            X_tensor = torch.FloatTensor(X_selected)
            y_tensor = torch.LongTensor(y)

            # Create data loader for batch processing
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            deep_scores = {}

            # Initialize and train the neural network
            deep_nn = DeepTestPredictor(input_dim=X_selected.shape[1])
            optimizer = optim.Adam(deep_nn.parameters(), lr=0.001)  # Adam is usually a good choice
            criterion = nn.CrossEntropyLoss()

            # Training loop - this is where the learning happens
            deep_nn.train()
            for epoch in range(50):  # 50 epochs is reasonable for this size problem
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()  # Reset gradients
                    outputs = deep_nn(batch_X)  # Forward pass
                    loss = criterion(outputs, batch_y)  # Calculate loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update weights

            self.models['deep_nn'] = deep_nn

            # Evaluate the trained model
            deep_nn.eval()
            with torch.no_grad():
                predictions = deep_nn(X_tensor)
                predicted_classes = torch.argmax(predictions, dim=1)
                accuracy = (predicted_classes == y_tensor).float().mean().item()
                deep_scores['deep_nn'] = {'mean_score': accuracy, 'std_score': 0.0}

        except Exception as e:
            print(f"Issue with deep learning model training: {e}")
            deep_scores = {'deep_nn': {'mean_score': 0.5, 'std_score': 0.0}}

        return deep_scores

    def statistical_comparison(self, results):
        """
        Compare model performances using proper statistical tests.

        It's not enough to just look at average scores - we need to know if
        differences are statistically significant. The Wilcoxon test helps
        us determine if one model is genuinely better than another.
        """
        model_names = list(results.keys())
        comparisons = {}

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1:]:
                if 'scores' in results[model1] and 'scores' in results[model2]:
                    scores1 = results[model1]['scores']
                    scores2 = results[model2]['scores']

                    # Wilcoxon signed-rank test - standard for comparing paired samples
                    try:
                        statistic, p_value = stats.wilcoxon(scores1, scores2)

                        comparisons[f"{model1}_vs_{model2}"] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05,  # Standard significance threshold
                            'effect_size': np.mean(scores1) - np.mean(scores2)
                        }
                    except:
                        # Sometimes the test fails (e.g., identical scores)
                        comparisons[f"{model1}_vs_{model2}"] = {
                            'statistic': 0,
                            'p_value': 1.0,
                            'significant': False,
                            'effect_size': 0
                        }

            return comparisons

    def train(self, X, y):
        """
        Main training function that orchestrates everything.

        This is where all the pieces come together: data preprocessing,
        model training, and evaluation.
        """
        print("Training ensemble of models with cross-validation...")

        # Train traditional ML models
        traditional_results = self.train_sklearn_models(X, y)

        # Train deep learning models
        deep_results = self.train_deep_models(X, y)

        # Combine all results
        all_results = {**traditional_results, **deep_results}

        # Statistical comparison between models
        statistical_tests = self.statistical_comparison(traditional_results)

        self.is_trained = True
        self.performance_metrics = {
            'model_performance': all_results,
            'statistical_comparisons': statistical_tests,
            'feature_importance': dict(zip(
                range(self.feature_selector.k),
                self.feature_selector.scores_[:self.feature_selector.k]
            ))
        }

        return self.performance_metrics


class TestFailurePredictionFramework:
    """
    The main framework that brings everything together.

    This is the user-facing interface that coordinates data collection,
    feature engineering, model training, and evaluation. Think of it as
    the conductor of our machine learning orchestra.
    """

    def __init__(self, github_token=None):
        self.github_collector = GitHubDataCollector(github_token)
        self.feature_engineer = TestFeatureEngineering()
        self.ensemble_predictor = EnsemblePredictor()

        self.experimental_data = None
        self.results_cache = {}

    def generate_realistic_dataset(self, n_samples=2000):
        """
        Create a synthetic dataset that mirrors real-world testing patterns.

        Why synthetic data? Real test failure data is often proprietary and hard
        to get. But we can simulate realistic patterns based on what we know
        about how tests fail in practice.

        The key is making the data realistic enough that models trained on it
        would work on real data.
        """
        np.random.seed(42)  # Ensures reproducible results

        print("Creating realistic synthetic dataset for model training...")

        data = []
        base_timestamp = datetime.now() - timedelta(days=365)

        # Different types of tests have different failure characteristics
        test_categories = {
            'unit': {'weight': 0.6, 'base_failure_rate': 0.05},  # Unit tests are usually stable
            'integration': {'weight': 0.25, 'base_failure_rate': 0.12},  # Integration tests more flaky
            'e2e': {'weight': 0.15, 'base_failure_rate': 0.18}  # End-to-end tests most fragile
        }
        for i in range(n_samples):
            # Choose test category based on realistic distributions
            category = np.random.choice(
                list(test_categories.keys()),
                p=[cat['weight'] for cat in test_categories.values()]
            )

            # Generate realistic timestamps with patterns
            # (more activity during weekdays, etc.)
            days_offset = np.random.exponential(30)
            if np.random.random() < 0.3:  # Weekend effect - different patterns
                days_offset += np.random.uniform(0, 2)

            timestamp = base_timestamp + timedelta(days=days_offset)

            # Generate features based on what we know affects test stability
            test_record = {
                'test_id': f"{category}_test_{i:04d}",
                'category': category,
                'timestamp': timestamp,

                # Code change patterns - based on empirical software engineering research
                'commits_last_7d': np.random.poisson(2.5),
                'lines_changed_ratio': np.random.beta(2, 8),  # Most changes are small
                'authors_involved': min(np.random.poisson(1.8) + 1, 10),

                # Test complexity - more complex tests tend to be more fragile
                'cyclomatic_complexity': np.random.gamma(2, 3),
                'assertion_density': np.random.exponential(0.3),
                'xpath_complexity': np.random.gamma(1.5, 2) if category == 'e2e' else 0,

                # Historical patterns - past behavior predicts future behavior
                'previous_30d_failure_rate': np.random.beta(2, 15),
                'flakiness_score': np.random.beta(1.5, 8),
                'maintenance_debt': np.random.exponential(0.4),

                # Environmental factors that affect test stability
                'parallel_execution': np.random.choice([0, 1], p=[0.3, 0.7]),
                'browser_type': np.random.choice(['chrome', 'firefox', 'edge'],
                                                 p=[0.6, 0.3, 0.1]) if category == 'e2e' else None,
                'test_data_dependency': np.random.choice([0, 1], p=[0.7, 0.3]),

                # Temporal patterns - time of day/week matters
                'time_since_last_run': np.random.exponential(24),  # hours
                'day_of_week': timestamp.weekday(),
                'is_holiday_period': int(timestamp.month in [12, 1, 7, 8]),
            }

            # Generate realistic failure probability based on multiple factors
            base_failure_prob = test_categories[category]['base_failure_rate']

            # Risk factors that increase failure probability
            risk_multiplier = 1.0
            if test_record['commits_last_7d'] > 5:
                risk_multiplier += 0.5  # Lots of recent changes
            if test_record['flakiness_score'] > 0.3:
                risk_multiplier += 0.3  # History of flakiness
            if test_record['cyclomatic_complexity'] > 10:
                risk_multiplier += 0.4  # Complex test logic
            if test_record['previous_30d_failure_rate'] > 0.2:
                risk_multiplier += 0.2  # Recent failures
            if test_record['maintenance_debt'] > 1.0:
                risk_multiplier += 0.3  # Needs maintenance
            if test_record['is_holiday_period']:
                risk_multiplier += 0.15  # Holiday deployments are riskier

            # Monday morning effect - more issues after weekends
            if test_record['day_of_week'] == 0:
                risk_multiplier += 0.2

            failure_probability = min(base_failure_prob * risk_multiplier, 0.85)
            test_record['test_failed'] = int(np.random.random() < failure_probability)

            data.append(test_record)

        df = pd.DataFrame(data)

        # Apply feature engineering to create additional predictive features
        df = self.feature_engineer.create_temporal_features(df)

        print(f"Generated {len(df)} test samples with {len(df.columns)} features")
        print(f"Overall failure rate: {df['test_failed'].mean():.1%}")
        print(f"Distribution by test type: {df['category'].value_counts().to_dict()}")

        return df

    def run_comprehensive_evaluation(self):
        """
        Execute a complete experimental evaluation following academic standards.

        This is the main research contribution: a thorough evaluation that shows
        our approach works and is statistically sound.
        """
        if self.experimental_data is None:
            self.experimental_data = self.generate_realistic_dataset(2500)

        print("\nRunning comprehensive experimental evaluation...")
        # Prepare the data for machine learning
        feature_columns = [col for col in self.experimental_data.columns
                           if col not in ['test_id', 'category', 'timestamp', 'test_failed', 'browser_type']]
        X = self.experimental_data[feature_columns].fillna(0).values
        y = self.experimental_data['test_failed'].values