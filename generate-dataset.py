import numpy as np
import pandas as pd
from scipy.stats import truncnorm


N_USERS = 50_000
np.random.seed(42)

def generate_categorical_feature(n, categories, probs, shuffle=True, dtype=None):
    """
    Generate a categorical feature with custom categories and distributions
    
    Args:
        n (int): Number of samples to generate
        categories (list): List of categories
        probs (list, optional): Probabilities for each category, should sum to 1
                                If None, equal distribution is used
        shuffle (bool): If True, shuffle the final array
        dtype: Data type for output (e.g., np.int8 for encoded values)
    
    Returns:
        np.ndarray: Generated categorical feature.
    """

    # Validate inputs
    if n < 1:
        raise ValueError("Number of samples must be positive")

    if not categories:
        raise ValueError("Categories list is empty")

    if not probs:
        raise ValueError("Probabilities list is empty")
    
    if np.sum(probs) != 1.0 and probs is not None:
        raise ValueError(f"Probabilities must sum to 1 for categories {categories}")

    if probs is not None and len(categories) != len(probs):
        raise ValueError(f"Categories and probabilities must have the same length.")

    probs = np.array(probs)
    k = len(categories)
    
    if probs is None:  # Equal split if not provided
        probs = [1.0 / k] * k
    
    counts = (np.array(probs) * n).astype(int)
    
    # Adjust last category to fix rounding issues
    counts[-1] += n - counts.sum()
    
    feature = np.concatenate([
        np.full(counts[i], categories[i], dtype=object) for i in range(k)
    ])
    
    if shuffle:
        np.random.shuffle(feature)
    
    if dtype is not None:
        feature = feature.astype(dtype)
    
    return feature

def truncated_normal(n, mean, std, low, high):
    """
    Generate n samples from a truncated normal distribution

    Args:
        n (int): Number of samples to generate
        mean (float): Mean of the distribution
        std (float): Standard deviation of the distribution
        low (float): Lower bound for truncation
        high (float): Upper bound for truncation
    """

    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)

def generate_ages(n, weights=(0.8, 0.2)):
    """
    Generate age distribution for users based on specified weights.

    Args:
        n (int): Number of samples to generate
        weights (tuple): Weights for different age groups
    """

    n_main = int(n * weights[0])
    n_older = n - n_main

    # Main group: ages 18-40, peak at 28, std=6 for a broad peak
    ages_main = truncated_normal(n_main, mean=34, std=9, low=18, high=55)

    # Older group: ages 41-80, slow falloff, mean=50, std=8
    ages_older = truncated_normal(n_older, mean=58, std=12, low=45, high=75)

    ages = np.concatenate([ages_main, ages_older])
    np.random.shuffle(ages)
    return ages.astype(int)


def generate_daily_watch_hours(ages):
    """
    Generate realistic daily watch hours for a streaming platform,
    based on age (younger users watch more).

    Args:
        ages (np.ndarray): Array of user ages.

    Returns:
        np.ndarray: Daily watch hours per user.
    """
    watch_hours = np.zeros_like(ages, dtype=float)

    # Age groups
    young_mask = ages < 25
    adult_mask = (ages >= 25) & (ages < 40)
    middle_mask = (ages >= 40) & (ages < 60)
    senior_mask = ages >= 60

    # Generate watch hours for each group
    watch_hours[young_mask] = truncated_normal(
        n=young_mask.sum(), mean=3.5, std=1.0, low=1, high=6
    )
    watch_hours[adult_mask] = truncated_normal(
        n=adult_mask.sum(), mean=2.5, std=0.8, low=1, high=4
    )
    watch_hours[middle_mask] = truncated_normal(
        n=middle_mask.sum(), mean=1.8, std=0.7, low=0.5, high=3
    )
    watch_hours[senior_mask] = truncated_normal(
        n=senior_mask.sum(), mean=1.0, std=0.5, low=0.2, high=2
    )

    # Small random noise for realism
    watch_hours += np.random.normal(0, 0.2, size=len(ages))
    watch_hours = watch_hours.round(2)
    watch_hours = np.clip(watch_hours, 0, None)  # no negative hours

    return watch_hours


genders_encoded = generate_categorical_feature(
    n=N_USERS,
    categories=["Male", "Female", "Other"],
    probs=[0.6, 0.35, 0.05],
    shuffle=True,
    dtype=None
)


ages = generate_ages(N_USERS)
daily_watch_hours = generate_daily_watch_hours(ages)

df = pd.DataFrame({
    "customer_id": np.arange(1, N_USERS + 1),
    "gender": genders_encoded,
    "age": ages,
    "daily_watch_hours": daily_watch_hours
})

df.to_csv("generated-dataset.csv", index=False)