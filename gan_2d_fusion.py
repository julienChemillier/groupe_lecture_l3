import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------------------------------
# ---------------------Parameters----------------------
# -----------------------------------------------------

# The matrix M from the calculus
M = np.array([
    [0, 0, -1, 0],
    [0, 0, 0, -1],
    [2, 0, -2, 0],
    [0, 2, 0, -2]
])

alpha_d = 1.0
alpha_g = 1.0
alpha = 1.0  # Common alpha for the randomized system
x_star, y_star = 0.0, 0.0  # Values that both networks are aiming for

# Initial conditions randomly taken in [-3, 3]
theta0 = np.random.uniform(-3, 3, size=4)
print(f"Conditions initiales aléatoires: x_d={theta0[0]:.2f}, y_d={theta0[1]:.2f}, x_g={theta0[2]:.2f}, y_g={theta0[3]:.2f}")


# -----------------------------------------------------
# ----------------------Systems------------------------
# -----------------------------------------------------

def dtheta_dt_deterministic(t, theta):
    ''' System of differential equation from the report without random part '''
    
    # The constant term which represent theta*
    constant_term = np.array([x_star, y_star, 0, 0])
    
    # The final differential equation
    dtheta = alpha * (M @ theta + constant_term)
    return dtheta

def random_noise(mu=0, sigma=0.3):
    ''' Generates a random noise vector using a Gaussian distribution '''
    # Take the noise from a gaussian distribution
    epsilon = np.random.normal(mu, sigma)

    # The final noise drawn randomly
    return np.array([epsilon, epsilon, 2 * epsilon, epsilon])

def dtheta_dt_random(t, theta):
    ''' System of differential equation from the report with a random part '''
    
    # The constant term from your calculus
    constant_term = np.array([x_star, y_star, 0, 0])
    
    # The random term from your calculus
    random_term = random_noise(sigma=0.3)
    
    # The final differential equation
    dtheta = alpha * (M @ theta + constant_term + random_term)
    
    return dtheta


# -----------------------------------------------------
# ----------------------Systems------------------------
# -----------------------------------------------------

# Time step
t_span = (0, 5)
t_eval = np.linspace(*t_span, 1000)

# Resolution of both systems
sol_det = solve_ivp(dtheta_dt_deterministic, t_span, theta0, t_eval=t_eval)
sol_rand = solve_ivp(dtheta_dt_random, t_span, theta0, t_eval=t_eval)

# Separation of the coordinates
# Deterministic
x_d_vals_det, y_d_vals_det = sol_det.y[0], sol_det.y[1]
x_g_vals_det, y_g_vals_det = sol_det.y[2], sol_det.y[3]
# Randomized
x_d_vals_rand, y_d_vals_rand = sol_rand.y[0], sol_rand.y[1]
x_g_vals_rand, y_g_vals_rand = sol_rand.y[2], sol_rand.y[3]

# Animation of the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_title('Comparaison des trajectoires (Déterministe vs. Aléatoire avec bruit Gaussien)', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

# Initialize the target and the lines for the animation
ax.scatter(x_star, y_star, color='red', marker='x', s=200, zorder=5, label='Cible')
line_d_det, = ax.plot([], [], 'b-', lw=1.5, alpha=0.5, label='Discriminateur (Déterministe)')
line_g_det, = ax.plot([], [], 'orange', lw=1.5, alpha=0.5, label='Générateur (Déterministe)')
line_d_rand, = ax.plot([], [], 'b--', linestyle='dotted', lw=2, alpha=0.9, label='Discriminateur (Aléatoire)')
line_g_rand, = ax.plot([], [], 'orange', linestyle='dotted', lw=2, alpha=0.9, label='Générateur (Aléatoire)')

# Making sur all lines are visible in the plot
all_x_vals = np.concatenate([x_d_vals_det, x_g_vals_det, x_d_vals_rand, x_g_vals_rand])
all_y_vals = np.concatenate([y_d_vals_det, y_g_vals_det, y_d_vals_rand, y_g_vals_rand])
ax.set_xlim(np.min(all_x_vals) - 0.5, np.max(all_x_vals) + 0.5)
ax.set_ylim(np.min(all_y_vals) - 0.5, np.max(all_y_vals) + 0.5)

ax.legend(fontsize=10)

# Animation loop
for i in range(1, len(t_eval)):
    # Deterministic lines
    line_d_det.set_data(x_d_vals_det[:i], y_d_vals_det[:i])
    line_g_det.set_data(x_g_vals_det[:i], y_g_vals_det[:i])
    
    # Random lines
    line_d_rand.set_data(x_d_vals_rand[:i], y_d_vals_rand[:i])
    line_g_rand.set_data(x_g_vals_rand[:i], y_g_vals_rand[:i])
    
    # Small pause for the animation
    plt.pause(0.005)

plt.show()