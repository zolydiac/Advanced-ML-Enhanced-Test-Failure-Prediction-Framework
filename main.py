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