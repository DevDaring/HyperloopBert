import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Standardized color palette to ensure consistency across all stages
# Avoid generic colors, use a premium modern aesthetic.
PALETTE = {
    'VanillaBERT': '#1f77b4',       # Muted Blue
    'LoopedBERT': '#ff7f0e',        # Safety Orange
    'ALBERTLoopedBERT': '#2ca02c',  # Cooked Asparagus Green
    'HyperloopBERT': '#d62728',     # Brick Red
    'EarlyMergeHyperloopBERT': '#9467bd' # Muted Purple
}

LINE_STYLES = {
    'VanillaBERT': '-',
    'LoopedBERT': '--',
    'ALBERTLoopedBERT': '-.',
    'HyperloopBERT': '-',
    'EarlyMergeHyperloopBERT': ':'
}

MARKERS = {
    'VanillaBERT': 'o',
    'LoopedBERT': 's',
    'ALBERTLoopedBERT': '^',
    'HyperloopBERT': 'D',
    'EarlyMergeHyperloopBERT': 'X'
}

def setup_plot_style():
    """Configure matplotlib/seaborn for a modern, clean aesthetic."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })

def plot_iso_loss_bias(df: pd.DataFrame, dataset_name: str, metric_col: str, 
                       size: str, output_path: str):
    """
    Plot Bias Metric vs Validation Loss (Iso-loss plot).
    This is the headline figure for the paper.
    X-axis is reversed (lower loss is better, reading left-to-right).
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for size and exclude token-budget markers if they don't have a specific band
    plot_df = df[(df['Model_Size'] == size) & (df['Band'].notna())].copy()
    if plot_df.empty:
        return
        
    # Convert Validation_Loss to float if it isn't already
    plot_df['Validation_Loss'] = pd.to_numeric(plot_df['Validation_Loss'], errors='coerce')
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors='coerce')
    plot_df = plot_df.dropna(subset=['Validation_Loss', metric_col])
    
    architectures = plot_df['Architecture'].unique()
    
    for arch in architectures:
        arch_df = plot_df[plot_df['Architecture'] == arch]
        
        # Aggregate across seeds if multiple seeds exist
        agg_df = arch_df.groupby('Validation_Loss').agg({
            metric_col: ['mean', 'std']
        }).reset_index()
        
        # Flatten columns
        agg_df.columns = ['Validation_Loss', 'Mean', 'Std']
        
        # Sort by validation loss for smooth line plotting
        agg_df = agg_df.sort_values('Validation_Loss')
        
        color = PALETTE.get(arch, '#333333')
        marker = MARKERS.get(arch, 'o')
        linestyle = LINE_STYLES.get(arch, '-')
        
        # Plot the mean line and scatter points
        ax.plot(agg_df['Validation_Loss'], agg_df['Mean'], 
                label=arch, color=color, linestyle=linestyle, marker=marker)
                
        # Add shaded error bars if std > 0 and we have multiple seeds
        if agg_df['Std'].sum() > 0:
            ax.fill_between(agg_df['Validation_Loss'], 
                            agg_df['Mean'] - agg_df['Std'],
                            agg_df['Mean'] + agg_df['Std'],
                            color=color, alpha=0.2)
                            
    ax.set_title(f'Stereotype Preference vs Validation Loss\n({dataset_name}, {size.capitalize()} Scale)')
    ax.set_xlabel('Validation Loss (Lower is Better ->)')
    ax.set_ylabel(metric_col.replace('_', ' '))
    
    # Invert X axis so lower loss is on the right
    ax.invert_xaxis()
    
    ax.legend(title='Architecture', loc='best')
    sns.despine()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def plot_token_budget_bias(df: pd.DataFrame, dataset_name: str, metric_col: str, 
                           size: str, output_path: str):
    """
    Plot Bias Metric vs Token Budget (Secondary comparison).
    Explicitly labeled as a secondary endpoint.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_df = df[(df['Model_Size'] == size) & (df['Token_Marker'].notna())].copy()
    if plot_df.empty:
        return
        
    # Convert token marker to millions for cleaner x-axis
    plot_df['Tokens_M'] = pd.to_numeric(plot_df['Token_Marker'], errors='coerce') / 1e6
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors='coerce')
    plot_df = plot_df.dropna(subset=['Tokens_M', metric_col])
    
    architectures = plot_df['Architecture'].unique()
    
    for arch in architectures:
        arch_df = plot_df[plot_df['Architecture'] == arch]
        
        agg_df = arch_df.groupby('Tokens_M').agg({
            metric_col: ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['Tokens_M', 'Mean', 'Std']
        agg_df = agg_df.sort_values('Tokens_M')
        
        color = PALETTE.get(arch, '#333333')
        marker = MARKERS.get(arch, 'o')
        linestyle = LINE_STYLES.get(arch, '-')
        
        ax.plot(agg_df['Tokens_M'], agg_df['Mean'], 
                label=arch, color=color, linestyle=linestyle, marker=marker)
                
        if agg_df['Std'].sum() > 0:
            ax.fill_between(agg_df['Tokens_M'], 
                            agg_df['Mean'] - agg_df['Std'],
                            agg_df['Mean'] + agg_df['Std'],
                            color=color, alpha=0.2)
                            
    ax.set_title(f'[SECONDARY ENDPOINT] Bias vs Token Budget\n({dataset_name}, {size.capitalize()} Scale)')
    ax.set_xlabel('Tokens Processed (Millions)')
    ax.set_ylabel(metric_col.replace('_', ' '))
    
    ax.legend(title='Architecture', loc='best')
    sns.despine()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def plot_loop_trajectory(df: pd.DataFrame, dataset_name: str, output_path: str):
    """
    Plot bias metric across loop depth.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df.empty or 'Loop_Depth' not in df.columns:
        return
        
    architectures = df['Architecture'].unique()
    
    for arch in architectures:
        arch_df = df[df['Architecture'] == arch]
        
        agg_df = arch_df.groupby('Loop_Depth').agg({
            'Mean_Preference_Rate': 'mean',
            'Std_Preference_Rate': 'mean' # Or recalculate std across seeds
        }).reset_index()
        
        agg_df = agg_df.sort_values('Loop_Depth')
        
        color = PALETTE.get(arch, '#333333')
        marker = MARKERS.get(arch, 'o')
        linestyle = LINE_STYLES.get(arch, '-')
        
        ax.plot(agg_df['Loop_Depth'], agg_df['Mean_Preference_Rate'], 
                label=arch, color=color, linestyle=linestyle, marker=marker)
                
    ax.set_title(f'Bias Trajectory Across Loop Depth\n({dataset_name})')
    ax.set_xlabel('Depth (Loop Index)')
    ax.set_ylabel('Mean Preference Rate')
    
    # Ensure x-axis shows integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax.legend(title='Architecture', loc='best')
    sns.despine()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def plot_cka_heatmap(cka_matrix: np.ndarray, architecture: str, output_path: str):
    """
    Plot Center Kernel Alignment (CKA) similarity matrix.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cka_matrix, annot=True, cmap='viridis', vmin=0, vmax=1, 
                fmt='.2f', cbar_kws={'label': 'CKA Similarity'}, ax=ax)
                
    ax.set_title(f'Representation Similarity (CKA) - {architecture}')
    ax.set_xlabel('Loop Index (Depth)')
    ax.set_ylabel('Loop Index (Depth)')
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def plot_pareto_front(df: pd.DataFrame, metric_col: str, output_path: str):
    """
    Plot Pareto front of GLUE Average vs Bias.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df.empty or 'GLUE_Average' not in df.columns or metric_col not in df.columns:
        return
        
    architectures = df['Architecture'].unique()
    
    for arch in architectures:
        arch_df = df[df['Architecture'] == arch]
        
        color = PALETTE.get(arch, '#333333')
        marker = MARKERS.get(arch, 'o')
        
        ax.scatter(arch_df['GLUE_Average'], arch_df[metric_col], 
                   label=arch, color=color, marker=marker, s=100)
                   
    ax.set_title('Pareto Frontier: Quality vs Bias')
    ax.set_xlabel('GLUE Average (Higher is Better)')
    ax.set_ylabel(metric_col.replace('_', ' ') + ' (Lower is Better)')
    
    ax.legend(title='Architecture', loc='best')
    sns.despine()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_stream_ablation(ablation_df, metric_col: str, band, output_path: str) -> None:
    """
    Line plot of stereotype preference rate vs stream count for HyperloopBERT.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Must contain columns: Stream_Count, metric_col (and optionally Band).
    metric_col : str
        Column name for the bias metric to plot on the y-axis.
    band : float or None
        The primary iso-loss band. If not None, filters ablation_df to that band.
    output_path : str
        File path for the saved figure.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    df = ablation_df.copy()
    if band is not None and 'Band' in df.columns:
        df = df[df['Band'] == band]

    if df.empty or 'Stream_Count' not in df.columns or metric_col not in df.columns:
        plt.close(fig)
        return

    summary = df.groupby('Stream_Count')[metric_col].agg(['mean', 'std']).reset_index()
    summary.columns = ['Stream_Count', 'mean', 'std']
    summary = summary.sort_values('Stream_Count')

    ax.errorbar(
        summary['Stream_Count'], summary['mean'],
        yerr=summary['std'], marker='o', linewidth=2,
        color=PALETTE.get('HyperloopBERT', '#e377c2'), capsize=4,
    )
    ax.set_title(f'Stream Count Ablation (Band {band})')
    ax.set_xlabel('Number of Streams')
    ax.set_ylabel(metric_col.replace('_', ' '))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    sns.despine()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_stream_disagreement(disagreement_df, output_path: str) -> None:
    """
    Scatter plot of stream disagreement vs bias effect size with a regression line.

    Parameters
    ----------
    disagreement_df : pd.DataFrame
        Must contain columns: Stream_Disagreement, Effect_Size.
    output_path : str
        File path for the saved figure.
    """
    import numpy as np
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    df = disagreement_df.dropna(subset=['Stream_Disagreement', 'Effect_Size'])
    if df.empty:
        plt.close(fig)
        return

    ax.scatter(df['Stream_Disagreement'], df['Effect_Size'], alpha=0.6,
               color=PALETTE.get('HyperloopBERT', '#e377c2'), s=60)

    z = np.polyfit(df['Stream_Disagreement'], df['Effect_Size'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['Stream_Disagreement'].min(), df['Stream_Disagreement'].max(), 100)
    ax.plot(x_range, p(x_range), color='black', linewidth=1.5, linestyle='--')

    if 'Pearson_R' in df.columns:
        r = df['Pearson_R'].mean()
        p_val = df['Pearson_P'].mean() if 'Pearson_P' in df.columns else float('nan')
        ax.annotate(f'r = {r:.3f} (p = {p_val:.3f})', xy=(0.05, 0.92),
                    xycoords='axes fraction', fontsize=10)

    ax.set_title('Stream Disagreement vs Bias Effect Size')
    ax.set_xlabel('Stream Disagreement')
    ax.set_ylabel('Bias Effect Size')
    sns.despine()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
