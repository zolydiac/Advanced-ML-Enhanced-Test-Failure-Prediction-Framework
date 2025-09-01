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