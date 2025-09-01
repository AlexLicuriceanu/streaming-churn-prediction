import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import string
import seaborn as sns
import matplotlib.pyplot as plt

N_USERS = 50_000
np.random.seed(42)

def generate_customer_id(n=0):
    """
    Generate unique customer IDs.

    Args:
        n (int): Number of customer IDs to generate.

    Returns:
        np.ndarray: Array of customer IDs.
    """
    
    chars = string.ascii_uppercase + string.digits
    base = len(chars)
    max_ids = base ** 8

    if n > max_ids:
        raise ValueError(f"Cannot generate more than {max_ids} unique 8-character IDs.")
    
    rng = np.random.default_rng()
    
    numbers = rng.choice(max_ids, size=n, replace=False)

    def to_base36(num):
        s = ""

        for _ in range(8):
            s = chars[num % base] + s
            num //= base
        return s
    
    ids = [to_base36(num) for num in numbers]

    return np.array(ids)


def generate_categorical_feature(n=0, categories=[], probs=None, shuffle=True, dtype=None):
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
    Generate right-skewed daily watch hours for streaming platform users.
    Younger users tend to watch more, with a long tail for heavy viewers.
    """
    watch_hours = np.zeros_like(ages, dtype=float)

    # Age groups
    young_mask = ages < 25
    adult_mask = (ages >= 25) & (ages < 40)
    middle_mask = (ages >= 40) & (ages < 60)
    senior_mask = ages >= 60

    # Log-normal parameters (mean and sigma in log-space)
    # Younger users: heavier tail
    watch_hours[young_mask] = np.random.lognormal(mean=1.2, sigma=0.7, size=young_mask.sum())
    watch_hours[adult_mask] = np.random.lognormal(mean=0.75, sigma=0.6, size=adult_mask.sum())
    watch_hours[middle_mask] = np.random.lognormal(mean=0.5, sigma=0.5, size=middle_mask.sum())
    watch_hours[senior_mask] = np.random.lognormal(mean=0.3, sigma=0.4, size=senior_mask.sum())

    # Clip to realistic maximum (e.g., no one watches more than 12 hours/day)
    watch_hours = np.clip(watch_hours, 0, 8)

    # Optional rounding
    watch_hours = watch_hours.round(2)

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
    "customer_id": generate_customer_id(N_USERS),
    "gender": genders_encoded,
    "age": ages,
    "daily_watch_hours": daily_watch_hours
})

df.to_csv("generated-dataset.csv", index=False)

# Plot correlation matrix
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()