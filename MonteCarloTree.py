import numpy as np
import matplotlib.pyplot as plt


def geometric_brownian_motion(S0, mu, sigma, T, n, num_paths=100000, rand_seed=None):
    """
    Generates paths of geometric Brownian motion.

    Parameters:
    S0 : float
        Initial price of the underlying asset.
    mu : float
        Drift (average rate of return) of the underlying asset.
    sigma : float
        Volatility (standard deviation of returns) of the underlying asset.
    T : float
        Time horizon in years.
    n : int
        Number of time steps.
    num_paths : int, optional
        Number of simulated paths.
    rand_seed : int or None, optional
        Random seed for reproducibility.

    Returns:
    ndarray
        2D array of shape (n+1, num_paths) containing simulated price paths.
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    dt = T / n
    t = np.linspace(0, T, n + 1)
    Z = np.random.standard_normal(size=(n, num_paths))
    S = np.zeros((n + 1, num_paths))
    S[0] = S0

    for i in range(1, n + 1):
        S[i] = S[i - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i - 1]
        )

    return S


def european_call_option_price(S0, K, r, sigma, T, n, num_paths=100000, rand_seed=None):
    """
    Estimates the price of a European call option using Monte Carlo simulation.

    Parameters:
    S0 : float
        Initial price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (standard deviation of returns) of the underlying asset.
    T : float
        Time to expiration in years.
    n : int
        Number of time steps for simulation.
    num_paths : int, optional
        Number of simulated paths.
    rand_seed : int or None, optional
        Random seed for reproducibility.

    Returns:
    float
        Estimated price of the European call option.
    """
    S = geometric_brownian_motion(S0, r, sigma, T, n, num_paths, rand_seed)
    ST = S[-1]  # Terminal prices
    payoff = np.maximum(ST - K, 0)  # Option payoff at expiration
    option_price = np.exp(-r * T) * np.mean(payoff)  # Discounted expected payoff
    return option_price


# Example usage:
S0 = 100  # Initial price of the underlying asset
K = 105  # Strike price of the option
r = 0.05  # Annual risk-free interest rate
sigma = 0.2  # Annual volatility of the underlying asset
T = 1  # Time to expiration in years
n = 252  # Number of time steps (daily observations for 1 year)
option_price = european_call_option_price(S0, K, r, sigma, T, n)

# Generate simulated price paths
num_paths = 10  # Number of paths to visualize
simulated_paths = geometric_brownian_motion(S0, r, sigma, T, n, num_paths)

# Plot simulated price paths
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(np.linspace(0, T, n + 1), simulated_paths[:, i], linewidth=1)

plt.title("Simulated Price Paths (Geometric Brownian Motion)")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.grid(True)
plt.show()

print("Estimated price of the European call option:", option_price)
