import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import string
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

    ages_main = truncated_normal(n_main, mean=34, std=9, low=18, high=55)
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

    # Clip to realistic maximum
    watch_hours = np.clip(watch_hours, 0, 8)

    # Optional rounding
    watch_hours = watch_hours.round(2)

    return watch_hours


def generate_subscription_types(genders, ages, daily_watch_hours):
    """
    Fully vectorized generation of subscription types based on gender, age, and daily watch hours.

    Args:
        genders (np.ndarray or pd.Series): Array of genders.
        ages (np.ndarray): Array of user ages.
        daily_watch_hours (np.ndarray): Array of daily watch hours.

    Returns:
        np.ndarray: Array of subscription types ("Premium", "Standard", "Basic").
    """

    genders = np.array(genders)
    ages = np.array(ages)
    watch_hours = np.array(daily_watch_hours)
    n = len(genders)

    categories = np.array(["Premium", "Standard", "Basic"])

    # Base probabilities per gender
    base_probs_dict = {
        "Male": np.array([0.25, 0.53, 0.22]),
        "Female": np.array([0.1, 0.25, 0.65]),
        "Other": np.array([0.15, 0.35, 0.5])
    }

    # Map base probabilities to each user
    base_probs = np.zeros((n, 3))
    for gender, probs in base_probs_dict.items():
        mask = genders == gender
        base_probs[mask] = probs

    # Age factor: younger users more likely premium
    age_factor = np.clip((50 - ages) / 50, 0, 1)

    # Watch hours factor: higher daily hours == higher premium
    watch_factor = np.clip(watch_hours / 12, 0, 1)

    # Combined adjustment
    combined_factor = 0.7 * age_factor + 0.3 * watch_factor
    adjustment = combined_factor * 0.2  # max +20% boost to premium

    # Apply adjustment: boost premium, reduce basic
    base_probs[:, 0] += adjustment
    base_probs[:, 2] -= adjustment

    # Clip to [0,1] and normalize
    base_probs = np.clip(base_probs, 0, 1)
    base_probs /= base_probs.sum(axis=1, keepdims=True)

    # Create cumulative sum for inverse transform sampling
    cum_probs = np.cumsum(base_probs, axis=1)
    rand_vals = np.random.rand(n, 1)
    subscription_idx = (rand_vals < cum_probs).argmax(axis=1)

    subscription = categories[subscription_idx]

    return subscription


def generate_profiles(subscription_types, ages):
    """
    Generate number of profiles created per user account.
    
    Args:
        subscription_types (np.ndarray): Array of subscription types ("Premium", "Standard", "Basic")
        ages (np.ndarray): User ages
    
    Returns:
        np.ndarray: Number of profiles per account
    """
    n = len(subscription_types)
    profiles = np.zeros(n, dtype=int)

    # Base probabilities by subscription type
    probs_map = {
        "Basic":    [0.85, 0.12, 0.03, 0.0, 0.0],  # mostly 1 profile
        "Standard": [0.25, 0.45, 0.20, 0.08, 0.02],  # mix of 2–3
        "Premium":  [0.10, 0.25, 0.30, 0.20, 0.15]   # more likely 3–5
    }

    # Age effect: younger account owners slightly more likely to have fewer profiles
    age_factor = np.clip((ages - 30) / 40, 0, 1)  # older → bigger household
    for i, sub in enumerate(subscription_types):
        probs = np.array(probs_map[sub], dtype=float)

        # Adjust probabilities by age: older users → more profiles
        probs = probs * (1 - 0.2 * (1 - age_factor[i]))
        probs /= probs.sum()

        profiles[i] = np.random.choice([1, 2, 3, 4, 5], p=probs)

    return profiles


genders = generate_categorical_feature(
    n=N_USERS,
    categories=["Male", "Female", "Other"],
    probs=[0.6, 0.35, 0.05],
    shuffle=True,
    dtype=None
)

ages = generate_ages(N_USERS)
daily_watch_hours = generate_daily_watch_hours(ages)
subscription_types = generate_subscription_types(genders, ages, daily_watch_hours)
profiles = generate_profiles(subscription_types, ages)

df = pd.DataFrame({
    "customer_id": generate_customer_id(N_USERS),
    "gender": genders,
    "age": ages,
    "daily_watch_hours": daily_watch_hours,
    "subscription_type": subscription_types,
    "profiles": profiles
})

df.to_csv("generated-dataset.csv", index=False)

# Plot correlation matrix
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

corr = df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

def cramers_v(x, y):
    """
    Calculate Cramér's V for categorical-categorical association.
    
    Args:
        x, y: Categorical variables (pd.Series or np.ndarray)
        
    Returns:
        float: Cramér's V statistic
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    k = min(confusion_matrix.shape)  # smaller of #rows or #columns
    return np.sqrt(chi2 / (n * (k - 1)))

# Example usage:
v = cramers_v(df["gender"], df["subscription_type"])
print("Cramér's V:", v)