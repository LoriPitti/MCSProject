#"PLOTS GRAPHS OF ITERATION PROGRESS, TIME, AND RELATIVE ERROR WITH RESPECT TO THE TOLERANCES

import matplotlib.pyplot as plt
import numpy as np
from utils.iterResult import IterationResult
from scipy.sparse import csr_matrix
import os


def plot_result(tols: np.ndarray, results: list[IterationResult], name, function):
    iterations = [res.iterations for res in results]
    rel_errors = [res.rel_error for res in results]
    times_taken = [res.time_taken for res in results]

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

    #create folder 
    folder = f"plots/{name}"
    os.makedirs(folder, exist_ok=True)

    # Plot 1 - Iterations
    axs[0].plot(tols, iterations, marker='o', color=np.random.rand(3,))
    axs[0].set_xscale('log')
    axs[0].set_title(f'{function}: Iterations vs Tolerance')
    axs[0].set_xlabel('Tolerance')
    axs[0].set_ylabel('Iterations')
    axs[0].grid(True)

    # Plot 2 - Relative Error
    axs[1].plot(tols, rel_errors, marker='o', color=np.random.rand(3,))
    axs[1].set_xscale('log')
    axs[1].set_title(f'{function}: Relative Error vs Tolerance')
    axs[1].set_xlabel('Tolerance')
    axs[1].set_ylabel('Relative Error')
    axs[1].grid(True)

    # Plot 3 - Time Taken
    axs[2].plot(tols, times_taken, marker='o', color=np.random.rand(3,))
    axs[2].set_xscale('log')
    axs[2].set_title(f'{function}: Time Taken vs Tolerance')
    axs[2].set_xlabel('Tolerance')
    axs[2].set_ylabel('Time Taken (s)')
    axs[2].grid(True)


    plt.tight_layout()  

    save_path = os.path.join(folder, f"{name}_{function}_tollerace.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
   # plt.show()

def plot_matrix_heatmap(A: csr_matrix, name):
    #create folder 
    folder = f"plots/{name}"
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=0.5)
    plt.title("Spy plot - Posizione dei valori non nulli in A")

    save_path = os.path.join(folder, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_difference(x_k, name, x):
    diff  = np.abs(x_k - x)

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(diff)), diff, label="Differenza assoluta")
    plt.xlabel('Indice')
    plt.ylabel('Differenza')
    plt.title("Differenza tra Soluzione Approssimata e Soluzione Esatta")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gradient_norm(grad_norms, name, i:float):
    #create folder 
    folder = f"plots/{name}/gradient_norm"
    os.makedirs(folder, exist_ok=True)

    plt.plot(grad_norms)
    plt.yscale('log')  # scala logaritmica per evidenziare il calo
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title(f'Convergence of the Gradient Method, tol: {i}')
    plt.grid(True)

    save_path = os.path.join(folder, f"{name}_conv_gradient_{i}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

