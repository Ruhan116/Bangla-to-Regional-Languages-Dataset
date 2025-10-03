import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set style for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.8

# Data from the table - ROUGE-L scores
data = {
    'Sylheti': {
        'Zero Shot': {
            'Qwen 2.5-3B': 31.18,
            'Gemma 3-1B': 32.13,
            'GPT-4o': 45.61,
            'Gemini 2.5': 45.4
        },
        'Few Shot': {
            'Qwen 2.5-3B': 29.28,
            'Gemma 3-1B': 38.78,
            'GPT-4o': 45.38,
            'Gemini 2.5': 49.97
        },
        'CoT': {
            'Qwen 2.5-3B': 26.99,
            'Gemma 3-1B': 46.53,
            'GPT-4o': 44.72,
            'Gemini 2.5': 46.02
        }
    },
    'Chittagonian': {
        'Zero Shot': {
            'Qwen 2.5-3B': 15.71,
            'Gemma 3-1B': 23.3,
            'GPT-4o': 27.46,
            'Gemini 2.5': 37.06
        },
        'Few Shot': {
            'Qwen 2.5-3B': 15.21,
            'Gemma 3-1B': 23.51,
            'GPT-4o': 28.67,
            'Gemini 2.5': 38.85
        },
        'CoT': {
            'Qwen 2.5-3B': 15.21,
            'Gemma 3-1B': 22.74,
            'GPT-4o': 50.57,
            'Gemini 2.5': 32.3
        }
    }
}

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('ROUGE-L Score Comparison Across Prompting Strategies and Dialects', 
             fontsize=14, fontweight='bold', y=1.02)

# Colors for different prompting strategies
colors = {
    'Zero Shot': '#5A8FB4',  # Blue
    'Few Shot': '#7CB89D',    # Teal
    'CoT': '#E8A87C'          # Orange
}

# Process each dialect
for idx, (dialect, ax) in enumerate(zip(['Sylheti', 'Chittagonian'], axes)):
    models = list(data[dialect]['Zero Shot'].keys())
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars for each prompting strategy
    for i, (strategy, color) in enumerate(colors.items()):
        values = [data[dialect][strategy][model] for model in models]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=strategy, 
                     color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Customize subplot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('ROUGE-L Score', fontsize=14, fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) {dialect}', fontsize=14, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add horizontal line separating open/closed source
    ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add text annotations for model types
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Open Source', 
           ha='center', fontsize=10, style='italic', color='gray')
    ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Closed Source', 
           ha='center', fontsize=10, style='italic', color='gray')
    
    if idx == 1:  # Only show legend on the right subplot
        ax.legend(title='Prompting Strategy', loc='upper left', 
                 frameon=True, fontsize=9, title_fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_rouge-L.pdf', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph saved as 'model_comparison.pdf' and 'model_comparison.png'")