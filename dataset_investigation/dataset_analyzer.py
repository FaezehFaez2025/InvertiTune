#!/usr/bin/env python3
"""
Dataset Analyzer for investigating statistics of different datasets.
Supports multiple report types and is easily extendable.
Enhanced with beautiful visualizations.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

# Set up beautiful plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DatasetAnalyzer:
    """
    Main class for analyzing datasets with different report types.
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """
        Initialize the dataset analyzer.
        
        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = Path(datasets_dir)
        self.report_generators = {
            'triple_count_distribution': TripleDistributionReport(),
            'cross_dataset_evaluation': CrossDatasetEvaluationReport(),
            'token_count_distribution': TokenCountDistributionReport(),
            'named_entity_distribution': NamedEntityDistributionReport(),
            # Add more report types here as they are implemented
        }
        
    def load_dataset(self, dataset_name: str, partition: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from JSON file.
        
        Args:
            dataset_name: Name of the dataset folder
            partition: Either 'train' or 'test'
            
        Returns:
            List of data samples
        """
        file_path = self.datasets_dir / dataset_name / f"T2G_{partition}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]
    
    def get_available_partitions(self, dataset_name: str) -> List[str]:
        """Get available partitions for a dataset."""
        dataset_path = self.datasets_dir / dataset_name
        partitions = []
        
        for file_path in dataset_path.glob("T2G_*.json"):
            partition = file_path.stem.replace("T2G_", "")
            partitions.append(partition)
        
        return partitions
    
    def generate_report(self, dataset_name: str, partition: str, report_type: str, 
                       output_dir: str = "reports", **kwargs) -> str:
        """
        Generate a specific report for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            partition: Dataset partition (train/test)
            report_type: Type of report to generate
            output_dir: Directory to save reports
            **kwargs: Additional arguments for the report generator
            
        Returns:
            Path to the generated report
        """
        if report_type not in self.report_generators:
            raise ValueError(f"Unknown report type: {report_type}. Available: {list(self.report_generators.keys())}")
        
        # Create output directory
        if report_type == 'cross_dataset_evaluation':
            train_dataset = kwargs.get('train_dataset')
            test_dataset = kwargs.get('test_dataset')
            output_path = Path(output_dir) / "cross_dataset_evaluation" / f"{train_dataset}_{test_dataset}"
        else:
            # Load dataset for regular reports
            data = self.load_dataset(dataset_name, partition)
            output_path = Path(output_dir) / dataset_name / partition
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_generator = self.report_generators[report_type]
        
        if report_type == 'cross_dataset_evaluation':
            # For cross-dataset evaluation, pass empty data and use kwargs
            report_path = report_generator.generate([], "", "", output_path, **kwargs)
        else:
            # For regular reports, pass the loaded data
            report_path = report_generator.generate(data, dataset_name, partition, output_path, **kwargs)
        
        return report_path
    
    def list_available_reports(self) -> List[str]:
        """List all available report types."""
        return list(self.report_generators.keys())


class BaseReportGenerator:
    """Base class for all report generators."""
    
    def generate(self, data: List[Dict[str, Any]], dataset_name: str, partition: str, 
                output_path: Path, **kwargs) -> str:
        """
        Generate a report. Must be implemented by subclasses.
        
        Args:
            data: Dataset data
            dataset_name: Name of the dataset
            partition: Dataset partition
            output_path: Path to save the report
            **kwargs: Additional arguments
            
        Returns:
            Path to the generated report
        """
        raise NotImplementedError


class TripleDistributionReport(BaseReportGenerator):
    """Report generator for triple distribution analysis."""
    
    def generate(self, data: List[Dict[str, Any]], dataset_name: str, partition: str, 
                output_path: Path, **kwargs) -> str:
        """
        Generate a triple distribution report.
        
        Args:
            data: Dataset data
            dataset_name: Name of the dataset
            partition: Dataset partition
            output_path: Path to save the report
            **kwargs: Additional arguments
            
        Returns:
            Path to the generated report
        """
        # Extract triple counts
        triple_counts = []
        for sample in data:
            if 'output' in sample:
                # Parse the output field to get triples
                triples = self._parse_triples(sample['output'])
                triple_counts.append(len(triples))
        
        # Create statistics
        stats = {
            'total_samples': len(data),
            'samples_with_triples': len(triple_counts),
            'min_triples': min(triple_counts) if triple_counts else 0,
            'max_triples': max(triple_counts) if triple_counts else 0,
            'mean_triples': np.mean(triple_counts) if triple_counts else 0,
            'median_triples': np.median(triple_counts) if triple_counts else 0,
            'std_triples': np.std(triple_counts) if triple_counts else 0
        }
        
        # Create visualizations
        self._create_enhanced_distribution_plot(triple_counts, dataset_name, partition, output_path, stats)
        self._create_statistics_table(stats, dataset_name, partition, output_path)
        
        # Save detailed statistics
        report_path = output_path / f"triple_distribution_report_{dataset_name}_{partition}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_name': dataset_name,
                'partition': partition,
                'statistics': stats,
                'triple_counts': triple_counts
            }, f, indent=2)
        
        return str(report_path)
    
    def _parse_triples(self, output: str) -> List[List[str]]:
        """
        Parse triples from the output field.
        
        Args:
            output: String representation of triples
            
        Returns:
            List of triples as [subject, relation, object]
        """
        try:
            # Handle both string and list formats
            if isinstance(output, str):
                # Remove any extra whitespace and parse
                output = output.strip()
                triples = json.loads(output)
            else:
                triples = output
            
            # Ensure it's a list of triples
            if isinstance(triples, list):
                return triples
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse triples from output: {output[:100]}...")
            return []
    
    def _create_enhanced_distribution_plot(self, triple_counts: List[int], dataset_name: str, 
                                         partition: str, output_path: Path, stats: Dict):
        """Create an enhanced, beautiful single distribution plot of triple counts."""
        # Create figure with single plot
        plt.figure(figsize=(14, 10))
        
        # Define beautiful color palette
        primary_color = '#a8c3e5'      # Modern blue
        secondary_color = '#8B5CF6'    # Purple
        accent_color = '#F59E0B'       # Amber
        complement_color = '#EF4444'   # Red
        gradient_color = '#06B6D4'     # Cyan
        
        # Create bins with optimal number
        n_bins = min(max(20, len(set(triple_counts)) // 2), 50)
        
        # Create histogram with uniform color for all bars
        plt.hist(triple_counts, bins=n_bins, alpha=0.8, 
                color=primary_color, edgecolor='white', 
                linewidth=1.2, density=True)
        
        # Add smooth KDE curve
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(triple_counts)
            x_range = np.linspace(min(triple_counts) - 1, max(triple_counts) + 1, 300)
            kde_values = kde(x_range)
            plt.plot(x_range, kde_values, color=secondary_color, linewidth=4, 
                    label='Density Curve', alpha=0.9, zorder=5)
            
            # Add subtle fill under the curve
            plt.fill_between(x_range, kde_values, alpha=0.2, color=secondary_color, zorder=1)
            
        except ImportError:
            pass
        
        # Add statistical reference lines with enhanced styling
        mean_val = stats['mean_triples']
        median_val = stats['median_triples']
        
        plt.axvline(mean_val, color=accent_color, linestyle='--', linewidth=3, 
                   label=f'Mean: {mean_val:.1f}', alpha=0.9, zorder=4)
        plt.axvline(median_val, color=complement_color, linestyle='-.', linewidth=3, 
                   label=f'Median: {median_val:.1f}', alpha=0.9, zorder=4)
        
        # Add percentile shading with better visibility
        p25 = np.percentile(triple_counts, 25)
        p75 = np.percentile(triple_counts, 75)
        plt.axvspan(p25, p75, alpha=0.4, color='#10B981', 
                   label=f'IQR ({p25:.1f} - {p75:.1f})', zorder=2,
                   linewidth=2, edgecolor='#059669')
        
        # Enhanced labels and title
        plt.xlabel('Number of Triples per Sample', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        plt.ylabel('Density', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        
        # Beautiful title with dataset info
        plt.title(f'Triple Count Distribution Analysis\n{dataset_name} Dataset ({partition.title()} Partition)', 
                 fontsize=22, fontweight='bold', color='#111827', pad=25)
        
        # Enhanced grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#6b7280')
        plt.gca().set_axisbelow(True)
        
        # Beautiful legend
        legend = plt.legend(loc='upper right', frameon=True, fancybox=True, 
                           shadow=True, fontsize=20, title='Statistics')
        legend.get_frame().set_facecolor('#f9fafb')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#d1d5db')
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(22)
        
        # Add beautiful statistics box
        stats_text = (f'Samples: {stats["total_samples"]:,}\n'
                     f'Range: {stats["min_triples"]} - {stats["max_triples"]}\n'
                     f'Std Dev: {stats["std_triples"]:.2f}')
        
        # Calculate position: right edge minus margin for consistent spacing
        right_margin = 0.035
        stats_x_pos = 1.0 - right_margin
        stats_y_pos = 0.70  # Position below legend
        
        plt.text(stats_x_pos, stats_y_pos, stats_text, transform=plt.gca().transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.2', facecolor='#f3f4f6', 
                         edgecolor='#9ca3af', alpha=0.95, linewidth=1.5))
        
        # Set background color and spine styling
        plt.gca().set_facecolor('#fefefe')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('#9ca3af')
            spine.set_linewidth(1.2)
        
        # Improve tick styling
        plt.gca().tick_params(axis='both', which='major', labelsize=22, 
                             colors='#374151', length=6, width=1.2)
        plt.gca().tick_params(axis='both', which='minor', length=3, width=1)
        
        # Add subtle margin
        plt.margins(x=0.02, y=0.05)
        
        # Tight layout for perfect spacing
        plt.tight_layout(pad=2.0)
        
        # Save plot with high quality
        plot_path = output_path / f"triple_distribution_{dataset_name}_{partition}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        print(f"Enhanced distribution plot saved to: {plot_path}")
    
    def _create_statistics_table(self, stats: Dict[str, Any], dataset_name: str, 
                               partition: str, output_path: Path):
        """Create a statistics table."""
        # Create DataFrame for better formatting
        df = pd.DataFrame([stats])
        
        # Save as CSV
        csv_path = output_path / f"triple_statistics_{dataset_name}_{partition}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as formatted text
        txt_path = output_path / f"triple_statistics_{dataset_name}_{partition}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Triple Distribution Statistics\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Partition: {partition}\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
        
        print(f"Statistics saved to: {csv_path} and {txt_path}")


class CrossDatasetEvaluationReport(BaseReportGenerator):
    """Report generator for cross-dataset evaluation analysis."""
    
    def generate(self, data: List[Dict[str, Any]], dataset_name: str, partition: str, 
                output_path: Path, **kwargs) -> str:
        """
        Generate a cross-dataset evaluation report.
        
        Args:
            data: Dataset data (not used for cross-dataset evaluation)
            dataset_name: Name of the dataset (not used for cross-dataset evaluation)
            partition: Dataset partition (not used for cross-dataset evaluation)
            output_path: Path to save the report
            **kwargs: Additional arguments including train_dataset and test_dataset
            
        Returns:
            Path to the generated report
        """
        # Extract train and test dataset names from kwargs
        train_dataset = kwargs.get('train_dataset')
        test_dataset = kwargs.get('test_dataset')
        
        if not train_dataset or not test_dataset:
            raise ValueError("Both train_dataset and test_dataset must be provided for cross-dataset evaluation")
        
        # Find the prediction file
        prediction_file = self._find_prediction_file(train_dataset, test_dataset)
        if not prediction_file:
            raise FileNotFoundError(f"Prediction file not found for train_dataset={train_dataset}, test_dataset={test_dataset}")
        
        # Load and parse the prediction file
        triple_counts = self._load_prediction_file(prediction_file)
        
        # Create statistics
        stats = {
            'total_samples': len(triple_counts),
            'samples_with_triples': len(triple_counts),
            'min_triples': min(triple_counts) if triple_counts else 0,
            'max_triples': max(triple_counts) if triple_counts else 0,
            'mean_triples': np.mean(triple_counts) if triple_counts else 0,
            'median_triples': np.median(triple_counts) if triple_counts else 0,
            'std_triples': np.std(triple_counts) if triple_counts else 0
        }
        
        # Create visualizations with modified title
        self._create_enhanced_distribution_plot(triple_counts, train_dataset, test_dataset, output_path, stats)
        self._create_statistics_table(stats, train_dataset, test_dataset, output_path)
        
        # Save detailed statistics
        report_path = output_path / f"cross_dataset_evaluation_{train_dataset}_{test_dataset}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'statistics': stats,
                'triple_counts': triple_counts
            }, f, indent=2)
        
        return str(report_path)
    
    def _find_prediction_file(self, train_dataset: str, test_dataset: str) -> Optional[Path]:
        """Find the prediction file for cross-dataset evaluation."""
        # Look for the file in the cross_dataset_evaluation directory
        base_path = Path("cross_dataset_evaluation")
        
        # Try different possible paths
        possible_paths = [
            base_path / "test_set" / test_dataset / "train_set" / train_dataset / "aggregated_finetuned_1.5B_improved_prediction_triplets.txt",
            base_path / test_dataset / train_dataset / "aggregated_finetuned_1.5B_improved_prediction_triplets.txt",
            base_path / f"{train_dataset}_{test_dataset}" / "aggregated_finetuned_1.5B_improved_prediction_triplets.txt"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_prediction_file(self, file_path: Path) -> List[int]:
        """Load and parse the prediction file."""
        triple_counts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # Parse the line as a list of triples
                        triples = json.loads(line)
                        if isinstance(triples, list):
                            triple_counts.append(len(triples))
                        else:
                            triple_counts.append(0)
                    except (json.JSONDecodeError, TypeError):
                        print(f"Warning: Could not parse line: {line[:100]}...")
                        triple_counts.append(0)
        
        return triple_counts
    
    def _create_enhanced_distribution_plot(self, triple_counts: List[int], train_dataset: str, 
                                         test_dataset: str, output_path: Path, stats: Dict):
        """Create an enhanced distribution plot for cross-dataset evaluation."""
        # Create figure with single plot
        plt.figure(figsize=(14, 10))
        
        # Define beautiful color palette
        primary_color = '#fba3df'
        secondary_color = '#e05bb6'
        accent_color = '#F59E0B'       # Amber
        complement_color = '#EF4444'   # Red
        gradient_color = '#06B6D4'     # Cyan
        
        # Create bins with optimal number
        n_bins = min(max(20, len(set(triple_counts)) // 2), 50)
        
        # Create histogram with uniform color for all bars
        plt.hist(triple_counts, bins=n_bins, alpha=0.8, 
                color=primary_color, edgecolor='white', 
                linewidth=1.2, density=True)
        
        # Add smooth KDE curve
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(triple_counts)
            x_range = np.linspace(min(triple_counts) - 1, max(triple_counts) + 1, 300)
            kde_values = kde(x_range)
            plt.plot(x_range, kde_values, color=secondary_color, linewidth=4, 
                    label='Density Curve', alpha=0.9, zorder=5)
            
            # Add subtle fill under the curve
            plt.fill_between(x_range, kde_values, alpha=0.2, color=secondary_color, zorder=1)
            
        except ImportError:
            pass
        
        # Add statistical reference lines with enhanced styling
        mean_val = stats['mean_triples']
        median_val = stats['median_triples']
        
        plt.axvline(mean_val, color=accent_color, linestyle='--', linewidth=3, 
                   label=f'Mean: {mean_val:.1f}', alpha=0.9, zorder=4)
        plt.axvline(median_val, color=complement_color, linestyle='-.', linewidth=3, 
                   label=f'Median: {median_val:.1f}', alpha=0.9, zorder=4)
        
        # Add percentile shading with better visibility
        p25 = np.percentile(triple_counts, 25)
        p75 = np.percentile(triple_counts, 75)
        plt.axvspan(p25, p75, alpha=0.4, color='#10B981', 
                   label=f'IQR ({p25:.1f} - {p75:.1f})', zorder=2,
                   linewidth=2, edgecolor='#059669')
        
        # Enhanced labels and title
        plt.xlabel('Number of Triples per Sample', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        plt.ylabel('Density', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        
        # Modified title for cross-dataset evaluation
        plt.title(f'Triple Count Distribution Analysis\nTrained on {train_dataset} Dataset, Tested on {test_dataset} Dataset', 
                 fontsize=22, fontweight='bold', color='#111827', pad=25)
        
        # Enhanced grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#6b7280')
        plt.gca().set_axisbelow(True)
        
        # Beautiful legend
        legend = plt.legend(loc='upper right', frameon=True, fancybox=True, 
                           shadow=True, fontsize=20, title='Statistics')
        legend.get_frame().set_facecolor('#f9fafb')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#d1d5db')
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(22)
        
        # Add beautiful statistics box
        stats_text = (f'Samples: {stats["total_samples"]:,}\n'
                     f'Range: {stats["min_triples"]} - {stats["max_triples"]}\n'
                     f'Std Dev: {stats["std_triples"]:.2f}')
        
        # Calculate position: right edge minus margin for consistent spacing
        right_margin = 0.035
        stats_x_pos = 1.0 - right_margin
        stats_y_pos = 0.70  # Position below legend
        
        plt.text(stats_x_pos, stats_y_pos, stats_text, transform=plt.gca().transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.2', facecolor='#f3f4f6', 
                         edgecolor='#9ca3af', alpha=0.95, linewidth=1.5))
        
        # Set background color and spine styling
        plt.gca().set_facecolor('#fefefe')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('#9ca3af')
            spine.set_linewidth(1.2)
        
        # Improve tick styling
        plt.gca().tick_params(axis='both', which='major', labelsize=22, 
                             colors='#374151', length=6, width=1.2)
        plt.gca().tick_params(axis='both', which='minor', length=3, width=1)
        
        # Add subtle margin
        plt.margins(x=0.02, y=0.05)
        
        # Tight layout for perfect spacing
        plt.tight_layout(pad=2.0)
        
        # Save plot with high quality
        plot_path = output_path / f"cross_dataset_evaluation_{train_dataset}_{test_dataset}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        print(f"Cross-dataset evaluation plot saved to: {plot_path}")
    
    def _create_statistics_table(self, stats: Dict[str, Any], train_dataset: str, 
                               test_dataset: str, output_path: Path):
        """Create a statistics table for cross-dataset evaluation."""
        # Create DataFrame for better formatting
        df = pd.DataFrame([stats])
        
        # Save as CSV
        csv_path = output_path / f"cross_dataset_evaluation_{train_dataset}_{test_dataset}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as formatted text
        txt_path = output_path / f"cross_dataset_evaluation_{train_dataset}_{test_dataset}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Cross-Dataset Evaluation Statistics\n")
            f.write(f"Train Dataset: {train_dataset}\n")
            f.write(f"Test Dataset: {test_dataset}\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
        
        print(f"Cross-dataset evaluation statistics saved to: {csv_path} and {txt_path}")


class TokenCountDistributionReport(BaseReportGenerator):
    """Report generator for token count distribution analysis."""
    
    def generate(self, data: List[Dict[str, Any]], dataset_name: str, partition: str, 
                output_path: Path, **kwargs) -> str:
        """
        Generate a token count distribution report.
        
        Args:
            data: Dataset data
            dataset_name: Name of the dataset
            partition: Dataset partition
            output_path: Path to save the report
            **kwargs: Additional arguments
            
        Returns:
            Path to the generated report
        """
        # Extract token counts from input field
        token_counts = []
        for sample in data:
            if 'input' in sample:
                # Count tokens in the input text (simple word-based tokenization)
                input_text = sample['input']
                tokens = input_text.split()  # Simple whitespace-based tokenization
                token_counts.append(len(tokens))
        
        # Create statistics
        stats = {
            'total_samples': len(data),
            'samples_with_input': len(token_counts),
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'mean_tokens': np.mean(token_counts) if token_counts else 0,
            'median_tokens': np.median(token_counts) if token_counts else 0,
            'std_tokens': np.std(token_counts) if token_counts else 0
        }
        
        # Create visualizations
        self._create_enhanced_distribution_plot(token_counts, dataset_name, partition, output_path, stats)
        self._create_statistics_table(stats, dataset_name, partition, output_path)
        
        # Save detailed statistics
        report_path = output_path / f"token_count_distribution_report_{dataset_name}_{partition}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_name': dataset_name,
                'partition': partition,
                'statistics': stats,
                'token_counts': token_counts
            }, f, indent=2)
        
        return str(report_path)
    
    def _create_enhanced_distribution_plot(self, token_counts: List[int], dataset_name: str, 
                                          partition: str, output_path: Path, stats: Dict):
        """Create an enhanced distribution plot of token counts."""
        # Create figure with single plot
        plt.figure(figsize=(14, 10))
        
        # Define beautiful color palette
        primary_color = '#FB7185'      # Coral/Peach
        secondary_color = '#E11D48'    # Darker coral
        accent_color = '#F59E0B'       # Amber
        complement_color = '#EF4444'   # Red
        gradient_color = '#06B6D4'     # Cyan
        
        # Create bins with optimal number
        n_bins = min(max(20, len(set(token_counts)) // 2), 50)
        
        # Create histogram with uniform color for all bars
        plt.hist(token_counts, bins=n_bins, alpha=0.8, 
                color=primary_color, edgecolor='white', 
                linewidth=1.2, density=True)
        
        # Add smooth KDE curve
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(token_counts)
            x_range = np.linspace(min(token_counts) - 1, max(token_counts) + 1, 300)
            kde_values = kde(x_range)
            plt.plot(x_range, kde_values, color=secondary_color, linewidth=4, 
                    label='Density Curve', alpha=0.9, zorder=5)
            
            # Add subtle fill under the curve
            plt.fill_between(x_range, kde_values, alpha=0.2, color=secondary_color, zorder=1)
            
        except ImportError:
            pass
        
        # Add statistical reference lines with enhanced styling
        mean_val = stats['mean_tokens']
        median_val = stats['median_tokens']
        
        plt.axvline(mean_val, color=accent_color, linestyle='--', linewidth=3, 
                   label=f'Mean: {mean_val:.1f}', alpha=0.9, zorder=4)
        plt.axvline(median_val, color=complement_color, linestyle='-.', linewidth=3, 
                   label=f'Median: {median_val:.1f}', alpha=0.9, zorder=4)
        
        # Add percentile shading with better visibility
        p25 = np.percentile(token_counts, 25)
        p75 = np.percentile(token_counts, 75)
        plt.axvspan(p25, p75, alpha=0.4, color='#10B981', 
                   label=f'IQR ({p25:.1f} - {p75:.1f})', zorder=2,
                   linewidth=2, edgecolor='#059669')
        
        # Enhanced labels and title
        plt.xlabel('Number of Tokens per Sample', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        plt.ylabel('Density', fontsize=22, fontweight='bold', 
                  color='#1f2937', labelpad=15)
        
        # Beautiful title with dataset info
        plt.title(f'Token Count Distribution Analysis\n{dataset_name} Dataset ({partition.title()} Partition)', 
                 fontsize=22, fontweight='bold', color='#111827', pad=25)
        
        # Enhanced grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#6b7280')
        plt.gca().set_axisbelow(True)
        
        # Beautiful legend
        legend = plt.legend(loc='upper right', frameon=True, fancybox=True, 
                           shadow=True, fontsize=20, title='Statistics')
        legend.get_frame().set_facecolor('#f9fafb')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#d1d5db')
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(22)
        
        # Add beautiful statistics box
        stats_text = (f'Samples: {stats["total_samples"]:,}\n'
                     f'Range: {stats["min_tokens"]} - {stats["max_tokens"]}\n'
                     f'Std Dev: {stats["std_tokens"]:.2f}')
        
        # Calculate position: right edge minus margin for consistent spacing
        right_margin = 0.035
        stats_x_pos = 1.0 - right_margin
        stats_y_pos = 0.70  # Position below legend
        
        plt.text(stats_x_pos, stats_y_pos, stats_text, transform=plt.gca().transAxes, 
                fontsize=20, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.2', facecolor='#f3f4f6', 
                         edgecolor='#9ca3af', alpha=0.95, linewidth=1.5))
        
        # Set background color and spine styling
        plt.gca().set_facecolor('#fefefe')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('#9ca3af')
            spine.set_linewidth(1.2)
        
        # Improve tick styling
        plt.gca().tick_params(axis='both', which='major', labelsize=22, 
                             colors='#374151', length=6, width=1.2)
        plt.gca().tick_params(axis='both', which='minor', length=3, width=1)
        
        # Add subtle margin
        plt.margins(x=0.02, y=0.05)
        
        # Tight layout for perfect spacing
        plt.tight_layout(pad=2.0)
        
        # Save plot with high quality
        plot_path = output_path / f"token_count_distribution_{dataset_name}_{partition}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        print(f"Token count distribution plot saved to: {plot_path}")
    
    def _create_statistics_table(self, stats: Dict[str, Any], dataset_name: str, 
                               partition: str, output_path: Path):
        """Create a statistics table for token count distribution."""
        # Create DataFrame for better formatting
        df = pd.DataFrame([stats])
        
        # Save as CSV
        csv_path = output_path / f"token_count_statistics_{dataset_name}_{partition}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as formatted text
        txt_path = output_path / f"token_count_statistics_{dataset_name}_{partition}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Token Count Distribution Statistics\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Partition: {partition}\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
        
        print(f"Token count statistics saved to: {csv_path} and {txt_path}")


class NamedEntityDistributionReport(BaseReportGenerator):
    """Report generator for named entity distribution analysis."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy model for NER."""
        try:
            import spacy
            self._spacy = spacy  # keep handle for use in generate
            try:
                self.nlp = spacy.load(model_name)
            except OSError as e:
                # Try to download once; if it fails, raise a clear error
                print(f"spaCy model '{model_name}' not found. Attempting to download...")
                import subprocess
                try:
                    subprocess.run(
                        ["python", "-m", "spacy", "download", model_name],
                        check=True
                    )
                    self.nlp = spacy.load(model_name)
                except Exception:
                    raise OSError(
                        f"Could not load or download spaCy model '{model_name}'. "
                        f"Install manually: python -m spacy download {model_name}"
                    ) from e
        except ImportError:
            self.nlp = None
            print("Warning: spaCy not available. Install with: pip install spacy")

    def generate(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        partition: str,
        output_path: Path,
        save_per_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a named entity distribution report.

        Computes per-sample:
          - entity_count
          - token_count (spaCy tokens, excluding spaces)
          - sentence_count (spaCy sentence splitter)
          - ents_per_token
          - ents_per_sentence
        And corpus-level entity-type counts & proportions.
        """
        if self.nlp is None:
            raise ImportError("spaCy is required for named entity analysis. Install with: pip install spacy")

        # -------- Collect texts safely
        texts: List[str] = [
            s["input"] for s in data
            if isinstance(s, dict) and isinstance(s.get("input", None), str)
        ]

        entity_counts: List[int] = []
        token_counts: List[int] = []
        sent_counts: List[int] = []
        ents_per_token: List[float] = []
        ents_per_sentence: List[float] = []
        entity_types: Dict[str, int] = {}

        # -------- Fast spaCy processing
        # Use simple sentencizer if model lacks parser (rare, but possible)
        if not self.nlp.has_pipe("senter") and not self.nlp.has_pipe("parser"):
            self.nlp.add_pipe("sentencizer")

        for doc in self.nlp.pipe(texts, batch_size=64, n_process=1):
            ents = list(doc.ents)
            e_cnt = len(ents)
            t_cnt = sum(1 for t in doc if not t.is_space)
            s_cnt = sum(1 for _ in doc.sents) or 1  # avoid zero if sentencizer fails

            entity_counts.append(e_cnt)
            token_counts.append(t_cnt)
            sent_counts.append(s_cnt)
            ents_per_token.append(e_cnt / max(t_cnt, 1))
            ents_per_sentence.append(e_cnt / max(s_cnt, 1))

            for ent in ents:
                label = ent.label_
                entity_types[label] = entity_types.get(label, 0) + 1

        total_samples = len(texts)
        total_entities = int(sum(entity_counts))
        num_with_ents = int(sum(1 for c in entity_counts if c > 0))

        # Type proportions
        type_props = {
            k: (v / total_entities) if total_entities > 0 else 0.0
            for k, v in sorted(entity_types.items(), key=lambda kv: kv[1], reverse=True)
        }

        # -------- Aggregate stats (convert to plain Python types for JSON)
        def _agg(v: List[float]):
            if not v:
                return 0, 0, 0.0, 0.0, 0.0
            return (
                int(min(v)),
                int(max(v)),
                float(np.mean(v)),
                float(np.median(v)),
                float(np.std(v)),
            )

        min_e, max_e, mean_e, med_e, std_e = _agg(entity_counts)
        _, _, mean_tok, _, _ = _agg(token_counts)
        _, _, mean_sent, _, _ = _agg(sent_counts)
        _, _, mean_ept, med_ept, std_ept = _agg(ents_per_token)
        _, _, mean_eps, med_eps, std_eps = _agg(ents_per_sentence)

        stats = {
            "total_samples": int(total_samples),
            "samples_with_entities": int(num_with_ents),
            "min_entities": int(min_e),
            "max_entities": int(max_e),
            "mean_entities": float(mean_e),
            "median_entities": float(med_e),
            "std_entities": float(std_e),
            "total_entities": int(total_entities),
            "unique_entity_types": int(len(entity_types)),
            "mean_tokens": float(mean_tok),
            "mean_sentences": float(mean_sent),
            "mean_ents_per_token": float(mean_ept),
            "median_ents_per_token": float(med_ept),
            "std_ents_per_token": float(std_ept),
            "mean_ents_per_sentence": float(mean_eps),
            "median_ents_per_sentence": float(med_eps),
            "std_ents_per_sentence": float(std_eps),
        }

        # -------- Visualizations
        self._plot_distribution(
            values=entity_counts,
            xlabel="Number of Named Entities per Sample",
            title=f"Named Entity Distribution Analysis\n{dataset_name} Dataset ({partition.title()} Partition)",
            stats_display={
                "Samples": f"{total_samples:,}",
                "Range": f"{min_e} - {max_e}",
                "Std Dev": f"{std_e:.2f}",
            },
            output_path=output_path / f"named_entity_distribution_{dataset_name}_{partition}.png",
        )

        self._plot_distribution(
            values=ents_per_token,
            xlabel="Entities per Token",
            title=f"Entity Density (per Token)\n{dataset_name} Dataset ({partition.title()} Partition)",
            stats_display={
                "Samples": f"{total_samples:,}",
                "Mean": f"{mean_ept:.4f}",
                "Std Dev": f"{std_ept:.4f}",
            },
            output_path=output_path / f"named_entity_density_per_token_{dataset_name}_{partition}.png",
        )

        self._plot_distribution(
            values=ents_per_sentence,
            xlabel="Entities per Sentence",
            title=f"Entity Density (per Sentence)\n{dataset_name} Dataset ({partition.title()} Partition)",
            stats_display={
                "Samples": f"{total_samples:,}",
                "Mean": f"{mean_eps:.3f}",
                "Std Dev": f"{std_eps:.3f}",
            },
            output_path=output_path / f"named_entity_density_per_sentence_{dataset_name}_{partition}.png",
        )

        self._create_entity_type_plot(
            entity_types=entity_types,
            type_props=type_props,
            dataset_name=dataset_name,
            partition=partition,
            output_path=output_path,
        )

        self._create_statistics_table(stats, dataset_name, partition, output_path)

        # Optional per-sample CSV (heavy; opt-in)
        if save_per_sample:
            per_sample_df = pd.DataFrame({
                "entity_count": entity_counts,
                "token_count": token_counts,
                "sentence_count": sent_counts,
                "entities_per_token": ents_per_token,
                "entities_per_sentence": ents_per_sentence,
            })
            per_sample_df.to_csv(
                output_path / f"named_entity_per_sample_{dataset_name}_{partition}.csv",
                index=False
            )

        # -------- Save JSON report
        report_path = output_path / f"named_entity_distribution_report_{dataset_name}_{partition}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "partition": partition,
                    "statistics": stats,
                    "entity_type_counts": entity_types,
                    "entity_type_proportions": type_props,
                },
                f,
                indent=2,
            )

        return str(report_path)

    # ---------------- internal helpers ---------------- #

    def _plot_distribution(
        self,
        values: List[float],
        xlabel: str,
        title: str,
        stats_display: Dict[str, str],
        output_path: Path,
    ):
        """Generic pretty histogram + gated KDE with robust defaults."""
        if not values:
            return

        plt.figure(figsize=(14, 10))

        primary_color = "#10B981"      # Emerald
        secondary_color = "#059669"    # Darker emerald
        accent_color = "#F59E0B"       # Amber
        complement_color = "#EF4444"   # Red

        # Binning: at least 5 bins; handle degenerate cases
        unique_vals = len(set(values))
        n_bins = max(5, min(50, max(20, unique_vals // 2)))
        plt.hist(
            values,
            bins=n_bins,
            alpha=0.8,
            color=primary_color,
            edgecolor="white",
            linewidth=1.2,
            density=True,
        )

        # KDE only if we have enough variation
        try:
            if unique_vals >= 2 and len(values) >= 5:
                from scipy.stats import gaussian_kde
                x_min, x_max = float(min(values)), float(max(values))
                # pad a little
                x_min, x_max = x_min - 0.05 * (x_max - x_min + 1e-9), x_max + 0.05 * (x_max - x_min + 1e-9)
                x_range = np.linspace(x_min, x_max, 300)
                kde = gaussian_kde(values)
                kde_values = kde(x_range)
                plt.plot(x_range, kde_values, color=secondary_color, linewidth=4, label="Density Curve", alpha=0.9)
                plt.fill_between(x_range, kde_values, alpha=0.2, color=secondary_color)
        except Exception:
            pass

        # Mean/median & IQR if defined
        mean_val = float(np.mean(values))
        med_val = float(np.median(values))
        plt.axvline(mean_val, color=accent_color, linestyle="--", linewidth=3, label=f"Mean: {mean_val:.3g}", alpha=0.9)
        plt.axvline(med_val, color=complement_color, linestyle="-.", linewidth=3, label=f"Median: {med_val:.3g}", alpha=0.9)

        if unique_vals >= 2:
            p25 = float(np.percentile(values, 25))
            p75 = float(np.percentile(values, 75))
            plt.axvspan(p25, p75, alpha=0.35, color="#8B5CF6", label=f"IQR ({p25:.3g} - {p75:.3g})", linewidth=2, edgecolor="#7C3AED")

        plt.xlabel(xlabel, fontsize=16, fontweight="bold", color="#1f2937", labelpad=15)
        plt.ylabel("Density", fontsize=22, fontweight="bold", color="#1f2937", labelpad=15)
        plt.title(title, fontsize=20, fontweight="bold", color="#111827", pad=25)

        legend = plt.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=20, title="Statistics")
        legend.get_frame().set_facecolor("#f9fafb")
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor("#d1d5db")
        legend.get_title().set_fontweight("bold")

        # stats box
        stats_text = "\n".join(f"{k}: {v}" for k, v in stats_display.items())
        plt.text(
            0.78, 0.70, stats_text, transform=plt.gca().transAxes, fontsize=20, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#f3f4f6", edgecolor="#9ca3af", alpha=0.95, linewidth=1.5),
        )

        plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="#6b7280")
        plt.gca().set_axisbelow(True)
        plt.gca().set_facecolor("#fefefe")
        for spine in plt.gca().spines.values():
            spine.set_edgecolor("#9ca3af")
            spine.set_linewidth(1.2)
        plt.gca().tick_params(axis="both", which="major", labelsize=18, colors="#374151", length=6, width=1.2)
        plt.gca().tick_params(axis="both", which="minor", length=3, width=1)
        plt.margins(x=0.02, y=0.05)
        plt.tight_layout(pad=2.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.3)
        plt.close()
        print(f"Plot saved to: {output_path}")

    def _create_entity_type_plot(
        self,
        entity_types: Dict[str, int],
        type_props: Dict[str, float],
        dataset_name: str,
        partition: str,
        output_path: Path,
    ):
        """Create bar plots of entity type counts and proportions."""
        if not entity_types:
            return

        # Counts
        plt.figure(figsize=(14, 8))
        sorted_counts = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_counts]
        counts = [v for _, v in sorted_counts]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        bars = plt.bar(range(len(labels)), counts, color=colors, alpha=0.85)
        for bar, c in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, str(c), ha="center", va="bottom", fontweight="bold", fontsize=9)

        plt.xlabel("Entity Types", fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")
        plt.title(f"Named Entity Types (Counts)\n{dataset_name} Dataset ({partition.title()} Partition)", fontsize=16, fontweight="bold", pad=20)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path_counts = output_path / f"entity_types_counts_{dataset_name}_{partition}.png"
        plt.savefig(path_counts, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Entity type counts plot saved to: {path_counts}")

        # Proportions
        plt.figure(figsize=(14, 8))
        props = [type_props.get(lbl, 0.0) for lbl in labels]
        bars = plt.bar(range(len(labels)), props, color=colors, alpha=0.85)
        for bar, p in zip(bars, props):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{p:.2%}", ha="center", va="bottom", fontweight="bold", fontsize=9)

        plt.xlabel("Entity Types", fontsize=14, fontweight="bold")
        plt.ylabel("Proportion", fontsize=14, fontweight="bold")
        plt.title(f"Named Entity Types (Proportions)\n{dataset_name} Dataset ({partition.title()} Partition)", fontsize=16, fontweight="bold", pad=20)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path_props = output_path / f"entity_types_proportions_{dataset_name}_{partition}.png"
        plt.savefig(path_props, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Entity type proportions plot saved to: {path_props}")

    def _create_statistics_table(self, stats: Dict[str, Any], dataset_name: str, partition: str, output_path: Path):
        """Create a statistics table for named entity distribution."""
        df = pd.DataFrame([stats])
        csv_path = output_path / f"named_entity_statistics_{dataset_name}_{partition}.csv"
        df.to_csv(csv_path, index=False)

        txt_path = output_path / f"named_entity_statistics_{dataset_name}_{partition}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Named Entity Distribution Statistics\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Partition: {partition}\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")

        print(f"Named entity statistics saved to: {csv_path} and {txt_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Dataset Analyzer")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--partition", type=str, choices=['train', 'test'], help="Dataset partition")
    parser.add_argument("--report_type", type=str, required=True, help="Type of report to generate")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--train_dataset", type=str, help="Training dataset name (for cross-dataset evaluation)")
    parser.add_argument("--test_dataset", type=str, help="Test dataset name (for cross-dataset evaluation)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Generate report
    try:
        if args.report_type == 'cross_dataset_evaluation':
            if not args.train_dataset or not args.test_dataset:
                print("Error: Both --train_dataset and --test_dataset are required for cross-dataset evaluation")
                return
            
            report_path = analyzer.generate_report(
                dataset_name="",  # Not used for cross-dataset evaluation
                partition="",     # Not used for cross-dataset evaluation
                report_type=args.report_type,
                output_dir=args.output_dir,
                train_dataset=args.train_dataset,
                test_dataset=args.test_dataset
            )
        else:
            if not args.dataset or not args.partition:
                print("Error: Both --dataset and --partition are required for regular reports")
                return
            
            report_path = analyzer.generate_report(
                args.dataset, 
                args.partition, 
                args.report_type, 
                args.output_dir
            )
        
        print(f"Report generated successfully: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")


if __name__ == "__main__":
    main()