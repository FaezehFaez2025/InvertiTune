import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects

def plot_f1_difference(output_dir='plots'):
    """
    Create an attractive plot showing the difference in BERTScore F1 between 
    One-KGC and Original Qwen 1.5B as sample size increases.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    sample_sizes = [2000, 4000, 6000, 8000, 10000, 12000]
    f1_differences = [11.10, 12.32, 12.53, 13.00, 13.13, 13.02]
    
    # Set style for a professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot data with single color
    ax.plot(
        sample_sizes, 
        f1_differences, 
        linestyle='-', 
        linewidth=3.5, 
        color='#3366cc',
        solid_capstyle='round'
    )
    
    # Add points with stylish markers
    scatter = ax.scatter(
        sample_sizes, 
        f1_differences, 
        s=140, 
        c='#3366cc',
        alpha=0.9,
        edgecolor='white', 
        linewidth=2, 
        zorder=5
    )
    
    # Add elegant annotations to each point
    for i, (x, y) in enumerate(zip(sample_sizes, f1_differences)):
        txt = ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 14),
            ha='center',
            va='bottom',
            fontsize=22,
            fontweight='bold',
            color='#303030',
            bbox=dict(
                boxstyle="round,rounding_size=0.5",
                fc='white',
                ec="#bbbbbb",
                alpha=0.9
            )
        )
        # Add subtle text shadow for depth
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=2, foreground='white')
        ])
    
    # Set labels with more professional styling
    ax.set_xlabel('Sample Size', fontsize=26, fontweight='bold', labelpad=12)
    ax.set_ylabel('F1 Improvement (%)', fontsize=26, fontweight='bold', labelpad=12)
    
    # Set appropriate axis limits with aesthetic padding
    ax.set_xlim(min(sample_sizes) - 900, max(sample_sizes) + 900)
    y_min = max(0, min(f1_differences) - 1.0)  # Ensure we start from zero or higher
    y_max = max(f1_differences) + 1.2
    ax.set_ylim(y_min, y_max)
    
    # Enhanced grid with better styling
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, color='#cccccc')
    
    # Add x-axis ticks for each sample size with improved formatting
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([f"{size:,}" for size in sample_sizes])
    
    # Set y-axis ticks with appropriate intervals and styling
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#555555')
    
    # Adjust layout for aesthetics
    plt.tight_layout()
    
    # Save with high quality
    output_path = os.path.join(output_dir, 'f1_difference_by_sample_size.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    
    # Also save as PNG for easy viewing
    png_path = os.path.join(output_dir, 'f1_difference_by_sample_size.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    
    plt.close()
    
    print(f"Enhanced F1 difference plots saved to:\n- {output_path}\n- {png_path}")
    return output_path

if __name__ == "__main__":
    plot_f1_difference()