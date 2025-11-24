import numpy as np
import kmeans
import common
import naive_em
import em
import config
from config import IMAGES_OUTPUT_DIR
X = np.loadtxt(config.DATA_DIR / "toy_data.txt")

# Store best solutions for each K
best_solutions = {}
lowest_costs = {}

# Try K = [1, 2, 3, 4]
for K in [1, 2, 3, 4]:
    best_cost = float('inf')
    best_mixture = None
    best_post = None
    
    # Run 5 times with different seeds
    for seed in [0, 1, 2, 3, 4]:
        # Initialize mixture
        mixture, post = common.init(X, K, seed)
        
        # Run k-means
        mixture, post, cost = kmeans.run(X, mixture, post)
        
        # Track the best solution (lowest cost)
        if cost < best_cost:
            best_cost = cost
            best_mixture = mixture
            best_post = post
    
    # Store the best solution for this K
    best_solutions[K] = (best_mixture, best_post)
    lowest_costs[K] = best_cost
    
    # Plot and save the best solution
    common.plot(X, best_mixture, best_post, f"K-Means (K={K})", save_path=f"{IMAGES_OUTPUT_DIR}\kmeans_k{K}.png")
    
    print(f"K={K}: Lowest cost = {best_cost:.4f}")

# Print summary
print("\nSummary of lowest costs:")
for K in [1, 2, 3, 4]:
    print(f"Cost|K={K} = {lowest_costs[K]:.4f}")
