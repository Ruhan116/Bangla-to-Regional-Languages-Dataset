import matplotlib.pyplot as plt
import numpy as np

# Set style for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8

# Complete data from the table
data = {
    'BLEU': {
        'Sylheti': {
            'Zero Shot': {'Qwen 2.5-3B': 9.4, 'Gemma 3-1B': 12.64, 'GPT-4o': 22.9, 'Gemini 2.5': 20.97},
            'Few Shot': {'Qwen 2.5-3B': 8.57, 'Gemma 3-1B': 17.24, 'GPT-4o': 22.34, 'Gemini 2.5': 23.67},
            'CoT': {'Qwen 2.5-3B': 7.16, 'Gemma 3-1B': 23.57, 'GPT-4o': 21.8, 'Gemini 2.5': 21.53}
        },
        'Chittagonian': {
            'Zero Shot': {'Qwen 2.5-3B': 3.35, 'Gemma 3-1B': 6.93, 'GPT-4o': 9.08, 'Gemini 2.5': 14.66},
            'Few Shot': {'Qwen 2.5-3B': 3.36, 'Gemma 3-1B': 7.22, 'GPT-4o': 8.99, 'Gemini 2.5': 14.78},
            'CoT': {'Qwen 2.5-3B': 3.36, 'Gemma 3-1B': 7.34, 'GPT-4o': 24.67, 'Gemini 2.5': 9.29}
        }
    },
    'ROUGE-1': {
        'Sylheti': {
            'Zero Shot': {'Qwen 2.5-3B': 32.97, 'Gemma 3-1B': 33.84, 'GPT-4o': 46.36, 'Gemini 2.5': 46.33},
            'Few Shot': {'Qwen 2.5-3B': 31.38, 'Gemma 3-1B': 40.29, 'GPT-4o': 46.28, 'Gemini 2.5': 49.02},
            'CoT': {'Qwen 2.5-3B': 29.41, 'Gemma 3-1B': 47.24, 'GPT-4o': 45.81, 'Gemini 2.5': 46.84}
        },
        'Chittagonian': {
            'Zero Shot': {'Qwen 2.5-3B': 16.83, 'Gemma 3-1B': 24.04, 'GPT-4o': 27.74, 'Gemini 2.5': 37.12},
            'Few Shot': {'Qwen 2.5-3B': 16.35, 'Gemma 3-1B': 23.97, 'GPT-4o': 29.07, 'Gemini 2.5': 36.76},
            'CoT': {'Qwen 2.5-3B': 16.3, 'Gemma 3-1B': 23.8, 'GPT-4o': 51.74, 'Gemini 2.5': 33.37}
        }
    },
    'ROUGE-2': {
        'Sylheti': {
            'Zero Shot': {'Qwen 2.5-3B': 14.76, 'Gemma 3-1B': 12.41, 'GPT-4o': 22.98, 'Gemini 2.5': 22.56},
            'Few Shot': {'Qwen 2.5-3B': 13.61, 'Gemma 3-1B': 17.48, 'GPT-4o': 22.42, 'Gemini 2.5': 25.2},
            'CoT': {'Qwen 2.5-3B': 11.66, 'Gemma 3-1B': 24.88, 'GPT-4o': 21.6, 'Gemini 2.5': 22.86}
        },
        'Chittagonian': {
            'Zero Shot': {'Qwen 2.5-3B': 4.2, 'Gemma 3-1B': 6.99, 'GPT-4o': 8.76, 'Gemini 2.5': 15.61},
            'Few Shot': {'Qwen 2.5-3B': 4.06, 'Gemma 3-1B': 7.21, 'GPT-4o': 9.11, 'Gemini 2.5': 14.95},
            'CoT': {'Qwen 2.5-3B': 4.06, 'Gemma 3-1B': 6.28, 'GPT-4o': 26.84, 'Gemini 2.5': 12.1}
        }
    },
    'ROUGE-L': {
        'Sylheti': {
            'Zero Shot': {'Qwen 2.5-3B': 31.18, 'Gemma 3-1B': 32.13, 'GPT-4o': 45.61, 'Gemini 2.5': 45.4},
            'Few Shot': {'Qwen 2.5-3B': 29.28, 'Gemma 3-1B': 38.78, 'GPT-4o': 45.38, 'Gemini 2.5': 49.97},
            'CoT': {'Qwen 2.5-3B': 26.99, 'Gemma 3-1B': 46.53, 'GPT-4o': 44.72, 'Gemini 2.5': 46.02}
        },
        'Chittagonian': {
            'Zero Shot': {'Qwen 2.5-3B': 15.71, 'Gemma 3-1B': 23.3, 'GPT-4o': 27.46, 'Gemini 2.5': 37.06},
            'Few Shot': {'Qwen 2.5-3B': 15.21, 'Gemma 3-1B': 23.51, 'GPT-4o': 28.67, 'Gemini 2.5': 38.85},
            'CoT': {'Qwen 2.5-3B': 15.21, 'Gemma 3-1B': 22.74, 'GPT-4o': 50.57, 'Gemini 2.5': 32.3}
        }
    },
    'METEOR': {
        'Sylheti': {
            'Zero Shot': {'Qwen 2.5-3B': 22.47, 'Gemma 3-1B': 25.84, 'GPT-4o': 41.55, 'Gemini 2.5': 41.09},
            'Few Shot': {'Qwen 2.5-3B': 21.11, 'Gemma 3-1B': 33.44, 'GPT-4o': 40.78, 'Gemini 2.5': 43.82},
            'CoT': {'Qwen 2.5-3B': 18.93, 'Gemma 3-1B': 43.6, 'GPT-4o': 39.71, 'Gemini 2.5': 41.54}
        },
        'Chittagonian': {
            'Zero Shot': {'Qwen 2.5-3B': 10.15, 'Gemma 3-1B': 18.44, 'GPT-4o': 22.86, 'Gemini 2.5': 32.86},
            'Few Shot': {'Qwen 2.5-3B': 9.91, 'Gemma 3-1B': 19.1, 'GPT-4o': 23.44, 'Gemini 2.5': 34.96},
            'CoT': {'Qwen 2.5-3B': 9.91, 'Gemma 3-1B': 18.9, 'GPT-4o': 47.3, 'Gemini 2.5': 28.07}
        }
    }
}

# Colors for different prompting strategies
colors = {
    'Zero Shot': '#5A8FB4',
    'Few Shot': '#7CB89D',
    'CoT': '#E8A87C'
}

# Create a large figure with all metrics (5 rows x 2 columns)
fig, axes = plt.subplots(5, 2, figsize=(12, 18))
fig.suptitle('Performance Comparison Across All Metrics and Prompting Strategies', 
             fontsize=14, fontweight='bold', y=0.995)

metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR']
dialects = ['Sylheti', 'Chittagonian']

# Generate plots for each metric
for row, metric in enumerate(metrics):
    for col, dialect in enumerate(dialects):
        ax = axes[row, col]
        
        models = list(data[metric][dialect]['Zero Shot'].keys())
        x = np.arange(len(models))
        width = 0.25
        
        # Plot bars for each prompting strategy
        for i, (strategy, color) in enumerate(colors.items()):
            values = [data[metric][dialect][strategy][model] for model in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=strategy, 
                         color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)
        
        # Customize subplot
        ax.set_ylabel(f'{metric} Score', fontsize=10, fontweight='bold')
        
        # Only add x-label to bottom row
        if row == 4:
            ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        
        # Title with subplot letter
        subplot_letter = chr(97 + row * 2 + col)  # a, b, c, d, ...
        ax.set_title(f'({subplot_letter}) {dialect}', fontsize=10, fontweight='bold', loc='left')
        
        ax.set_xticks(x)
        if row == 4:  # Only show labels on bottom row
            ax.set_xticklabels(models, rotation=15, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
        
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add vertical line separating open/closed source
        ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text annotations for model types (only on first row)
        if row == 0:
            ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Open Source', 
                   ha='center', fontsize=8, style='italic', color='gray')
            ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Closed Source', 
                   ha='center', fontsize=8, style='italic', color='gray')
        
        # Only show legend on the top-right subplot
        if row == 0 and col == 1:
            ax.legend(title='Prompting Strategy', loc='upper left', 
                     frameon=True, fontsize=8, title_fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_all_metrics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("Combined graph with all metrics saved as 'model_comparison_all_metrics.pdf'")
print("\nNow use this in LaTeX:")
print("\\includegraphics[width=0.95\\textwidth]{diagrams/model_comparison_all_metrics.pdf}")