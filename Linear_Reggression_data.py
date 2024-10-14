"""In this example we simulate data when there is NO LATENT CONFOUNDING variable:
- The dataset include 3 variables: 2 continuous: X1, X2 ~ N(0,1), one binary treatment T and the continuous outcome Y
- We assume that we have available experimental data and observational data with 2 known confounding variables X1,X2
- Observational causal Graph:
                            T -> Y
                            X1 -> Y
                            X2 -> Y
                        Confounders:
                            X1 -> T
                            X2 -> T

Notes: Change the random.seed() for creating different datasets. If random seed is the same for experimental and observational data the X1 and X2 variable will be the same"""
import numpy as np
import pandas as pd

def simulate_experimental_data(n_samples):
    # Step 1: Set random seed for reproducibility
    np.random.seed(42)

    # Step 2: Generate features
    X1 = np.random.randn(n_samples)  # Feature 1 from N(0,1)
    X2 = np.random.randn(n_samples)  # Feature 2 from N(0,1)
    T = np.random.binomial(1, 0.5, n_samples)  # Generate T(binary) with probability 0.5(RCT)

    X = np.column_stack([X1, X2, T])  # Combine features into one matrix

    # Step 3: Set true coefficients and intercept
    beta_0 = 1  # Intercept
    beta_1 = 3  # Coefficient for X1
    beta_2 = -2  # Coefficient for X2
    beta_binary = 4  # Coefficient for T

    # Step 4: Generate random noise (error term)
    noise = np.random.normal(0, 1, n_samples)  # Noise with mean 0 and std deviation 1

    # Step 5: Compute the target variable y
    Y = beta_0 + beta_1 * X1 + beta_2 * X2 + beta_binary * T + noise

    # Step 6: Create a DataFrame
    data = pd.DataFrame({'T': T,'X1': X1, 'X2': X2, 'Y': Y})

    return data

def simulate_observational_data(n_samples):

    # Step 1: Set random seed for reproducibility
    np.random.seed(42)

    # Step 2: Generate features
    X1 = np.random.randn(n_samples)  # Feature 1 from N(0,1)
    X2 = np.random.randn(n_samples)  # Feature 2 from N(0,1)

    noise = np.random.normal(0, 1, n_samples)  # Noise with mean 0 and std deviation 1
    #generate T_new confounded from X1, X2 using logistic regression model
    b_X1 = 2
    b_X2 = 1.5
    logit_T_new = b_X1 * X1 + b_X2 * X2 + noise
    prob = 1 / (1 + np.exp(-logit_T_new))

    T_new = np.random.binomial(1, prob.flatten())

    X = np.column_stack([X1, X2, T_new])  # Combine features into one matrix

    # Step 3: Set true coefficients and intercept
    beta_0 = 1  # Intercept
    beta_1 = 3  # Coefficient for X1
    beta_2 = -2  # Coefficient for X2
    beta_binary = 4  # Coefficient for T


    # Step 5: Compute the target variable y
    Y = beta_0 + beta_1 * X1 + beta_2 * X2 + beta_binary * T_new + noise

    # Step 6: Create a DataFrame
    data = pd.DataFrame({'T': T_new, 'X1': X1, 'X2': X2, 'Y': Y})
    return data


n_samples = 1000
data_exp = simulate_experimental_data(n_samples)
data_obs = simulate_observational_data(n_samples)

print('Experimental data:\n', data_exp)
print('Observational dataL\n', data_obs)
