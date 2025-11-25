#!/usr/bin/env python3
"""
Script to compare K-means and EM algorithms for K=1,2,3,4
Generates plots for both algorithms to compare their solutions
"""

import numpy as np
import kmeans
import common
import naive_em
import config
from config import IMAGES_OUTPUT_DIR

# Load the toy data
X = np.loadtxt(config.DATA_DIR / "toy_data.txt")

print("Comparing K-means and EM algorithms for K=1,2,3,4")
print("=" * 60)

# Compare algorithms for each K
for K in [1, 2, 3, 4]:
    print(f"\n--- K = {K} ---")
    
    # Use the same seed for fair comparison
    seed = 0
    
    # Initialize with the same starting point
    mixture_init, post_init = common.init(X, K, seed)
    
    # Run K-means (reinitialize to avoid modifying original)
    mixture_kmeans_init, post_kmeans_init = common.init(X, K, seed)
    mixture_kmeans, post_kmeans, cost_kmeans = kmeans.run(X, mixture_kmeans_init, post_kmeans_init)
    
    # Run EM (naive_em) (reinitialize to avoid modifying original)
    mixture_em_init, post_em_init = common.init(X, K, seed)
    mixture_em, post_em, log_likelihood_em = naive_em.run(X, mixture_em_init, post_em_init)
    
    # Print results
    print(f"K-means cost (distortion): {cost_kmeans:.4f}")
    print(f"EM log-likelihood: {log_likelihood_em:.4f}")
    
    # Generate plots
    kmeans_title = f"K-Means (K={K})"
    em_title = f"EM Algorithm (K={K})"
    
    kmeans_path = f"{IMAGES_OUTPUT_DIR}/comparison_kmeans_k{K}.png"
    em_path = f"{IMAGES_OUTPUT_DIR}/comparison_em_k{K}.png"
    
    # Save plots
    common.plot(X, mixture_kmeans, post_kmeans, kmeans_title, save_path=kmeans_path)
    common.plot(X, mixture_em, post_em, em_title, save_path=em_path)
    
    print(f"Saved: {kmeans_path}")
    print(f"Saved: {em_path}")
    
    # Compare cluster centers
    print("K-means cluster centers:")
    for j in range(K):
        print(f"  Cluster {j+1}: μ=({mixture_kmeans.mu[j,0]:.2f}, {mixture_kmeans.mu[j,1]:.2f}), σ={np.sqrt(mixture_kmeans.var[j]):.2f}")
    
    print("EM cluster centers:")
    for j in range(K):
        print(f"  Cluster {j+1}: μ=({mixture_em.mu[j,0]:.2f}, {mixture_em.mu[j,1]:.2f}), σ={np.sqrt(mixture_em.var[j]):.2f}")

print("\nComparison complete! Check the images-output folder for plots.")
