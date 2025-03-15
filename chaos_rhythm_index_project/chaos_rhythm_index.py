import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system parameters
sigma, beta, rho = 10, 8/3, 28

# Define the Lorenz system (without control) for baseline comparison
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Compute the original CRI
def compute_cri(x, y, lambda_val):
    return np.abs(x + lambda_val * y)

# Compute the enhanced CRI
def compute_enhanced_cri(x, y, z, dxdt, lambda_val, gamma, eta):
    return np.sqrt(x**2 + lambda_val * y**2 + gamma * z**2) + eta * np.abs(dxdt)

# Adaptive Sliding Mode Control using Enhanced CRI with tanh to reduce chattering
def adaptive_control(s, cri, k, alpha, epsilon=0.1):
    return -k * np.tanh(s / epsilon) - alpha * cri

# Define the controlled Lorenz system with adaptive SMC
def controlled_lorenz(t, state, sigma, beta, rho, lambda_val, gamma, eta, k, alpha):
    x, y, z = state
    # Calculate dx/dt from the uncontrolled Lorenz system for CRI enhancement
    dxdt_uncontrolled = sigma * (y - x)
    s = x + lambda_val * y  # Sliding surface
    # Compute CRI values
    cri_original = compute_cri(x, y, lambda_val)
    cri_enhanced = compute_enhanced_cri(x, y, z, dxdt_uncontrolled, lambda_val, gamma, eta)
    # Compute control input
    u = adaptive_control(s, cri_enhanced, k, alpha)
    dxdt = sigma * (y - x) + u
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Simulation settings
lambda_val = 2        # Sliding surface parameter
gamma = 1.5           # Weight for z(t) in enhanced CRI
eta = 0.05            # Weight for derivative of x(t) in enhanced CRI
k = 0.7               # Control gain
alpha = 0.01          # Adaptive factor
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)
initial_state = [-8, 8, 27]

# Solve the controlled system
solution = solve_ivp(controlled_lorenz, t_span, initial_state, t_eval=t_eval,
                     args=(sigma, beta, rho, lambda_val, gamma, eta, k, alpha))

# Extract state variables and compute CRI values
t = solution.t
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]
dxdt = sigma * (y - x)  # Approximation for dx/dt

CRI_original = compute_cri(x, y, lambda_val)
CRI_enhanced = compute_enhanced_cri(x, y, z, dxdt, lambda_val, gamma, eta)

# Compute control signal history based on sliding surface
s_values = x + lambda_val * y
u_values = -k * np.tanh(s_values / 0.1) - alpha * CRI_enhanced

# Plotting the results
plt.figure(figsize=(18, 8))

# Time series of x(t)
plt.subplot(2, 2, 1)
plt.plot(t, x, 'r', linewidth=1)
plt.title("Time Series of x(t)", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("x(t)", fontsize=12)
plt.grid(True)

# Comparison of original CRI and Enhanced CRI
plt.subplot(2, 2, 2)
plt.plot(t, CRI_original, 'g', label="Original CRI", linewidth=1)
plt.plot(t, CRI_enhanced, 'm--', label="Enhanced CRI", linewidth=1)
plt.title("CRI Comparison", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("CRI", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Control signal u(t)
plt.subplot(2, 2, 3)
plt.plot(t, u_values, 'b', linewidth=1)
plt.title("Control Signal u(t)", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("u(t)", fontsize=12)
plt.grid(True)

# Combined view: x(t) and Enhanced CRI
plt.subplot(2, 2, 4)
plt.plot(t, x, 'r', label="x(t)", linewidth=1)
plt.plot(t, CRI_enhanced, 'm--', label="Enhanced CRI", linewidth=1)
plt.title("x(t) vs. Enhanced CRI", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.suptitle("Enhanced CRI and Chaos Twin Detection in the Lorenz System", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("enhanced_cri_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
