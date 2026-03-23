import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import argparse
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wasserstein_distance

def parse_distribution_file(file_path):
    """
    Parse a distribution file with format 'x: y' where x is the value and y is the count.
    Returns a numpy array of values with repetitions based on counts.
    """
    values = []
    counts = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                value, count = line.split(':')
                value = int(value.strip())
                count = int(count.strip())
                values.append(value)
                counts.append(count)
            except ValueError:
                print(f"Warning: Skipping invalid line: {line}")
    
    # Create array with repetitions based on counts
    result = np.array([], dtype=int)
    for val, count in zip(values, counts):
        result = np.append(result, np.full(count, val))
    
    return result

def find_distribution_pairs(distribution_dir, distribution_type):
    """
    Find pairs of ground truth and prediction files in the distribution directory.
    Returns a dictionary mapping sample sizes to (gt_file, pr_file) tuples.
    """
    pairs = {}
    
    # Get all files in the directory
    files = os.listdir(distribution_dir)
    
    # Find ground truth files based on distribution type
    if distribution_type == "triples":
        gt_files = [f for f in files if f.endswith('_triples_gt.txt')]
    else:  # tokens
        gt_files = [f for f in files if f.endswith('_tokens_gt.txt')]
    
    for gt_file in gt_files:
        # Extract sample size from filename (e.g., "500_triples_gt.txt" -> 500)
        match = re.match(r'(\d+)_(?:triples|tokens)_gt\.txt', gt_file)
        if match:
            sample_size = int(match.group(1))
            pr_file = f"{sample_size}_{distribution_type}_pr.txt"
            
            # Check if corresponding prediction file exists
            if pr_file in files:
                pairs[sample_size] = (
                    os.path.join(distribution_dir, gt_file),
                    os.path.join(distribution_dir, pr_file)
                )
    
    return pairs

def plot_distributions(distributions, sample_sizes, output_dir, distribution_type):
    """
    Generate and save distribution plots with subplots for each sample size.
    
    Args:
        distributions: Dictionary mapping sample sizes to (ground_truth, predictions) tuples
        sample_sizes: List of sample sizes in ascending order
        output_dir: Directory to save the output plot
        distribution_type: Either "triples" or "tokens"
    """
    # Set style with a more professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (12, 10),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create custom color palette
    main_color = '#1E3A8A'  # Deep blue
    secondary_color = '#f7ab44' if distribution_type == "triples" else '#cb519f'  # Orange for triples, Pink for tokens
    
    # Calculate the number of rows and columns for the subplot grid
    n_samples = len(sample_sizes)
    n_cols = min(3, n_samples)
    n_rows = int(np.ceil(n_samples/n_cols))
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes array for easier indexing if it's a 2D array
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols == 1:
        axes = [axes]
    
    # Find the maximum value across all distributions to set consistent x-axis limits
    max_value = 0
    for size in sample_sizes:
        ground_truth, predictions = distributions[size]
        max_value = max(max_value, np.max(ground_truth), np.max(predictions))
    
    # Create handles for the legend
    handles = []
    labels = []
    
    # Plot each sample size in its own subplot
    for idx, size in enumerate(sample_sizes):
        ax = axes[idx]
        ground_truth, predictions = distributions[size]
        
        # Plot ground truth and predictions with enhanced styling
        gt_line = sns.kdeplot(
            ground_truth, 
            color=main_color, 
            linewidth=2.5, 
            label='Ground Truth', 
            ax=ax, 
            fill=True,
            alpha=0.2
        )
        pr_line = sns.kdeplot(
            predictions, 
            color=secondary_color, 
            linewidth=2.5, 
            label='Predictions', 
            ax=ax, 
            fill=True,
            alpha=0.2
        )
        
        # Store handles and labels for the first plot only
        if idx == 0:
            handles = [
                plt.Line2D([0], [0], color=main_color, linewidth=2.5),
                plt.Line2D([0], [0], color=secondary_color, linewidth=2.5)
            ]
            labels = ['Ground Truth', 'Predictions']
        
        # Enhanced styling for each subplot
        ax.set_title(f'Sample Size: {size}', fontsize=14, fontweight='bold')
        
        # Only add x-axis label to bottom row subplots
        row = idx // n_cols
        if row == n_rows - 1:  # Bottom row
            ax.set_xlabel(f'{distribution_type.capitalize()[:-1]} Count per Knowledge Graph', fontsize=12)
        else:
            ax.set_xlabel('')  # Remove x-label for non-bottom rows
            
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#D1D5DB')
    
    # Hide empty subplots if any
    for idx in range(len(sample_sizes), len(axes)):
        axes[idx].set_visible(False)
    
    # Add single legend to the figure with enhanced styling - now positioned at upper right
    fig.legend(
        handles, 
        labels, 
        loc='upper right', 
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    # Add a title for the entire figure
    fig.suptitle(
        f'{distribution_type.capitalize()} Distribution Comparison', 
        fontsize=18, 
        fontweight='bold',
        y=0.98
    )
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    
    # Save the plot with high quality
    output_path = os.path.join(output_dir, f'{distribution_type}_distribution_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Enhanced distribution plots saved to {output_path}")
    return output_path

def plot_distribution_convergence(distributions, sample_sizes, output_dir, distribution_type):
    """
    Create a visualization showing how distributions converge with increasing sample size.
    """
    # Set style with a more professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (12, 10),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create custom color palette
    main_color = '#1E3A8A'  # Deep blue
    secondary_color = '#f7ab44' if distribution_type == "triples" else '#cb519f'  # Orange for triples, Pink for tokens
    
    # Calculate the number of rows and columns for the subplot grid
    n_samples = len(sample_sizes)
    n_cols = min(3, n_samples)
    n_rows = int(np.ceil(n_samples/n_cols))
    
    # Create figure with subplots - Modified to arrange horizontally
    fig = plt.figure(figsize=(16, 8))
    
    # Create grid with Wasserstein plot on left, distributions on right
    gs = fig.add_gridspec(n_rows, n_cols + 2, width_ratios=[2, 0.1] + [1] * n_cols)
    
    # Main convergence plot (left side, spans all rows)
    ax_convergence = fig.add_subplot(gs[:, 0])
    
    # Calculate Wasserstein distances for convergence plot
    wasserstein_distances = []
    for size in sample_sizes:
        ground_truth, predictions = distributions[size]
        w_dist = wasserstein_distance(ground_truth, predictions)
        wasserstein_distances.append(w_dist)
    
    # Create gradient color for convergence plot
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4C78A8', '#72B7B2', '#54A24B'])
    norm = plt.Normalize(min(wasserstein_distances), max(wasserstein_distances))
    
    # Plot convergence with gradient colors and improved styling
    for i in range(len(sample_sizes) - 1):
        ax_convergence.plot(
            sample_sizes[i:i+2], 
            wasserstein_distances[i:i+2], 
            '-', 
            color=cmap(norm(wasserstein_distances[i])), 
            linewidth=3
        )
    
    # Add points with white edge for better visibility
    ax_convergence.scatter(
        sample_sizes, 
        wasserstein_distances, 
        s=120, 
        c=[cmap(norm(w)) for w in wasserstein_distances],
        edgecolor='white', 
        linewidth=2, 
        zorder=5
    )
    
    # Improve convergence plot styling
    ax_convergence.set_xlabel('Sample Size', fontsize=18, fontweight='bold', labelpad=15)
    ax_convergence.set_ylabel('Wasserstein Distance', fontsize=18, fontweight='bold', labelpad=15)
    ax_convergence.tick_params(axis='both', labelsize=16)  # Increase tick label font size
    ax_convergence.grid(True, linestyle='--', alpha=0.7)
    ax_convergence.spines['top'].set_visible(False)
    ax_convergence.spines['right'].set_visible(False)
    
    # Add annotations to points
    for i, (x, y) in enumerate(zip(sample_sizes, wasserstein_distances)):
        ax_convergence.annotate(
            f"{y:.2f}",  # Changed from .4f to .2f for 2 decimal places
            (x, y), 
            textcoords="offset points",
            xytext=(0, 10), 
            ha='center',
            fontsize=14,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8)
        )
    
    # Create handles for the legend
    handles = [
        plt.Line2D([0], [0], color=main_color, linewidth=2.5),
        plt.Line2D([0], [0], color=secondary_color, linewidth=2.5)
    ]
    labels = ['Ground Truth', 'Predictions']
    
    # Create subplot axes for distributions on the right side
    distribution_axes = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Plot individual distributions with enhanced styling
    for idx, size in enumerate(sample_sizes):
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot - now on the right side (starting from column 2)
        ax = fig.add_subplot(gs[row, col + 2])
        distribution_axes[row][col] = ax
        
        ground_truth, predictions = distributions[size]
        
        # Use KDE plots with enhanced styling
        sns.kdeplot(
            ground_truth, 
            color=main_color, 
            linewidth=2.5, 
            label='Ground Truth', 
            ax=ax,
            fill=True,
            alpha=0.2
        )
        sns.kdeplot(
            predictions, 
            color=secondary_color, 
            linewidth=2.5, 
            label='Predictions', 
            ax=ax,
            fill=True,
            alpha=0.2
        )
        
        # Add sample size as a tiny legend at the top right corner
        ax.text(
            0.95, 0.95,  # Position at top right
            f'N={size}',  # Shortened label
            transform=ax.transAxes,  # Use axes coordinates
            fontsize=12,  # Increased from 9
            fontweight='bold',  # Changed from 'normal' for better visibility
            ha='right',
            va='top',
            bbox=dict(
                boxstyle="round,pad=0.4",  # Slightly larger padding
                fc='white',
                ec="#4B5563",
                alpha=0.9  # Slightly more opaque
            )
        )
        
        # Enhanced x-axis labeling strategy:
        # Remove individual x-labels from all subplots since we have a common figure label
        ax.set_xlabel('')  # No individual x-labels
        
        # Remove x-tick labels for non-bottom rows for cleaner look
        if row != n_rows - 1:  # Non-bottom rows
            ax.tick_params(axis='x', labelbottom=False)
        
        # For subplots that are not in the first column of distributions, remove y-tick labels
        if col > 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=0)
        else:
            # For the first column, increase y-axis tick label font size
            ax.tick_params(axis='y', labelsize=14)
        
        # Increase x-axis tick label font size for all distribution subplots
        ax.tick_params(axis='x', labelsize=14)
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#D1D5DB')

    # Calculate global maximum y limit across all subplots
    global_max_y = 0
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = distribution_axes[row_idx][col_idx]
            if ax is not None:
                current_ylim = ax.get_ylim()
                current_max = current_ylim[1]
                if current_max > global_max_y:
                    global_max_y = current_max

    # Apply uniform y limits to all subplots
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = distribution_axes[row_idx][col_idx]
            if ax is not None:
                ax.set_ylim(0, global_max_y)

    # Set y-axis labels for first column only (one label per row)
    for row in range(n_rows):
        if row < len(distribution_axes) and 0 < len(distribution_axes[row]):
            if distribution_axes[row][0] is not None:
                distribution_axes[row][0].set_ylabel('Density', fontsize=18, fontweight='bold')  # Increased from 12 to 16
    
    # Add a common x-axis label for all distribution subplots at the bottom
    # This creates a unified label that applies to all subplots
    fig.text(
        0.75,  # x position (center of right subplot area)
        0.01,  # y position (bottom of figure) - moved down to increase space
        f'{distribution_type.capitalize()[:-1]} Count per Knowledge Graph',
        ha='center',
        va='bottom',
        fontsize=18,  # Increased from 16 to 18
        fontweight='bold'
    )
    
    # Add legend to the figure
    fig.legend(
        handles, 
        labels, 
        loc='upper center',
        fontsize=18,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        ncol=2,
        bbox_to_anchor=(0.7, 0.95)
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Reduce spacing between subplots and add bottom margin for the common x-label
    fig.subplots_adjust(hspace=0.15, wspace=0.4, top=0.85, bottom=0.08)
    
    # Save the plot with high quality
    output_path = os.path.join(output_dir, f'{distribution_type}_convergence_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Enhanced convergence analysis plots saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Visualize distribution comparisons')
    parser.add_argument('--distribution_dir', type=str, default='distribution', 
                        help='Directory containing distribution files')
    parser.add_argument('--output_dir', type=str, default='plots', 
                        help='Directory to save output plots')
    parser.add_argument('--distribution_type', type=str, choices=['triples', 'tokens'], default='triples',
                        help='Type of distribution to visualize (triples or tokens)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all distribution pairs
    pairs = find_distribution_pairs(args.distribution_dir, args.distribution_type)
    
    # Create distributions dictionary
    distributions = {size: (parse_distribution_file(gt_file), parse_distribution_file(pr_file)) for size, (gt_file, pr_file) in pairs.items()}
    
    # Plot distributions
    plot_distributions(distributions, sorted(distributions.keys()), args.output_dir, args.distribution_type)
    
    # Plot distribution convergence
    plot_distribution_convergence(distributions, sorted(distributions.keys()), args.output_dir, args.distribution_type)

if __name__ == "__main__":
    main()