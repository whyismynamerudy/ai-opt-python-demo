import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def format_time(seconds):
    """Format time in seconds to a more readable format"""
    return f"{seconds:.2f}s"

def safe_get(data, keys, default=None):
    """Safely get nested values from dictionary"""
    if not isinstance(keys, list):
        keys = [keys]
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def plot_time_comparison(comparison_data, optimized_data, regular_data, output_dir, prefix):
    """Plot time comparison between optimized and regular versions"""
    plt.figure(figsize=(10, 6))
    
    labels = ['Optimized', 'Regular']
    
    # Try to get times from individual data files
    optimized_time = safe_get(optimized_data, 'total_duration', 0)
    regular_time = safe_get(regular_data, 'total_duration', 0)
    times = [optimized_time, regular_time]
    
    # Calculate percentage difference
    if regular_time > 0:
        time_diff = optimized_time - regular_time
        diff_pct = (time_diff / regular_time) * 100
    else:
        diff_pct = 0
    
    bars = plt.bar(labels, times, color=['#4CAF50', '#2196F3'])
    
    # Add time difference percentage
    diff_text = f"{abs(diff_pct):.2f}% faster" if diff_pct < 0 else f"{diff_pct:.2f}% slower"
    
    # Add time values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                format_time(height), ha='center', va='bottom')
    
    plt.ylabel('Time (seconds)')
    plt.title(f'Total Execution Time\n({diff_text})')
    
    # Add a horizontal line for average time
    avg_time = np.mean(times)
    plt.axhline(y=avg_time, color='gray', linestyle='--', alpha=0.7)
    plt.text(1.5, avg_time, f"Avg: {format_time(avg_time)}", va='bottom', ha='center')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_time_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_tokens_comparison(comparison_data, optimized_data, regular_data, output_dir, prefix):
    """Plot token usage comparison"""
    plt.figure(figsize=(10, 6))
    
    labels = ['Optimized', 'Regular']
    
    # Safely get token data with fallbacks
    opt_prompt = safe_get(optimized_data, ['total_tokens', 'prompt'], 0)
    reg_prompt = safe_get(regular_data, ['total_tokens', 'prompt'], 0)
    prompt_tokens = [opt_prompt, reg_prompt]
    
    opt_completion = safe_get(optimized_data, ['total_tokens', 'completion'], 0)
    reg_completion = safe_get(regular_data, ['total_tokens', 'completion'], 0)
    completion_tokens = [opt_completion, reg_completion]
    
    # Total tokens
    opt_total = safe_get(optimized_data, ['total_tokens', 'total'], opt_prompt + opt_completion)
    reg_total = safe_get(regular_data, ['total_tokens', 'total'], reg_prompt + reg_completion)
    total_tokens = [opt_total, reg_total]
    
    # Calculate difference
    diff = opt_total - reg_total
    diff_pct = (diff / reg_total) * 100 if reg_total > 0 else 0
    diff_text = f"{diff:,} tokens ({diff_pct:.1f}%)"
    
    # Create stacked bar
    plt.bar(labels, prompt_tokens, label='Prompt Tokens', color='#4CAF50', alpha=0.7)
    plt.bar(labels, completion_tokens, bottom=prompt_tokens, label='Completion Tokens', color='#2196F3', alpha=0.7)
    
    # Add total token count on top
    for i, total in enumerate(total_tokens):
        plt.text(i, total + max(1000, total * 0.05), f"{total:,}", ha='center')
    
    # Add annotation about difference
    plt.annotate(
        f"Difference: {diff_text}",
        xy=(0.5, 0.9),
        xycoords='axes fraction',
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.ylabel('Token Count')
    plt.title('Token Usage Comparison')
    plt.legend()
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_token_usage.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_api_calls_and_steps(comparison_data, optimized_data, regular_data, output_dir, prefix):
    """Plot API calls and steps comparison"""
    plt.figure(figsize=(10, 6))
    
    width = 0.35
    labels = ['API Calls', 'Steps Taken']
    
    # Use individual data files
    optimized_values = [
        safe_get(optimized_data, 'api_calls', 0), 
        safe_get(optimized_data, 'steps_taken', 0)
    ]
    
    regular_values = [
        safe_get(regular_data, 'api_calls', 0), 
        safe_get(regular_data, 'steps_taken', 0)
    ]
    
    x = np.arange(len(labels))
    
    plt.bar(x - width/2, optimized_values, width, label='Optimized', color='#4CAF50')
    plt.bar(x + width/2, regular_values, width, label='Regular', color='#2196F3')
    
    plt.xticks(x, labels)
    plt.ylabel('Count')
    plt.title('API Calls and Steps Comparison')
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(optimized_values):
        plt.text(i - width/2, v + 0.1, str(v), ha='center')
    
    for i, v in enumerate(regular_values):
        plt.text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_api_calls_steps.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_actions_timeline(optimized_data, regular_data, output_dir, prefix):
    """Plot timeline of actions performed by both versions"""
    optimized_actions = safe_get(optimized_data, 'actions_performed', [])
    regular_actions = safe_get(regular_data, 'actions_performed', [])
    
    # Determine figure height based on number of actions
    max_actions = max(len(optimized_actions), len(regular_actions))
    fig_height = max(6, max_actions * 0.5 + 2)  # Minimum height of 6 inches
    
    plt.figure(figsize=(10, fig_height))
    
    # Prepare data
    opt_labels = [f"{i+1}. {a.get('type', 'unknown')}" for i, a in enumerate(optimized_actions)]
    reg_labels = [f"{i+1}. {a.get('type', 'unknown')}" for i, a in enumerate(regular_actions)]
    
    # Create y-positions
    y_positions = np.arange(max(len(opt_labels), len(reg_labels)))
    
    # Plot actions as a scatter plot
    plt.scatter([0] * len(opt_labels), y_positions[:len(opt_labels)], marker='o', 
              s=100, label='Optimized', color='#4CAF50')
    plt.scatter([1] * len(reg_labels), y_positions[:len(reg_labels)], marker='o', 
              s=100, label='Regular', color='#2196F3')
    
    # Add action labels
    for i, label in enumerate(opt_labels):
        plt.text(0, y_positions[i], f" {label}", va='center', ha='left')
    
    for i, label in enumerate(reg_labels):
        plt.text(1, y_positions[i], f" {label}", va='center', ha='left')
    
    # Set axis properties
    plt.yticks([])
    plt.xticks([0, 1], ['Optimized', 'Regular'])
    plt.title('Actions Performed Sequence')
    plt.grid(False)
    
    # Add some buffer space
    plt.xlim(-0.5, 1.5)
    plt.ylim(-1, max(len(opt_labels), len(reg_labels)))
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_actions_timeline.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_detection_metrics(optimized_data, regular_data, output_dir, prefix):
    """Plot detection time and page load time"""
    plt.figure(figsize=(10, 6))
    
    labels = ['Page Load Time', 'Detection Time']
    
    optimized_values = [
        safe_get(optimized_data, 'page_load_time', 0), 
        safe_get(optimized_data, 'detection_time', 0)
    ]
    
    regular_values = [
        safe_get(regular_data, 'page_load_time', 0), 
        safe_get(regular_data, 'detection_time', 0)
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, optimized_values, width, label='Optimized', color='#4CAF50')
    plt.bar(x + width/2, regular_values, width, label='Regular', color='#2196F3')
    
    plt.ylabel('Time (seconds)')
    plt.title('Page Load and Detection Time')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(optimized_values):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}s", ha='center', va='bottom')
    
    for i, v in enumerate(regular_values):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}s", ha='center', va='bottom')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_detection_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_summary_table(comparison_data, optimized_data, regular_data, output_dir, prefix):
    """Create a summary table of key metrics with safe data access"""
    plt.figure(figsize=(10, 8))
    
    # Safely get values
    opt_time = safe_get(optimized_data, 'total_duration', 0)
    reg_time = safe_get(regular_data, 'total_duration', 0)
    time_diff = opt_time - reg_time
    time_diff_pct = (time_diff / reg_time * 100) if reg_time > 0 else 0
    
    # Safely get token values
    opt_tokens = safe_get(optimized_data, ['total_tokens', 'total'], 0)
    reg_tokens = safe_get(regular_data, ['total_tokens', 'total'], 0)
    tokens_diff = opt_tokens - reg_tokens
    
    # Safely get API calls and steps
    opt_api = safe_get(optimized_data, 'api_calls', 0)
    reg_api = safe_get(regular_data, 'api_calls', 0)
    opt_steps = safe_get(optimized_data, 'steps_taken', 0)
    reg_steps = safe_get(regular_data, 'steps_taken', 0)
    
    # Safely get page load and detection times
    opt_page_load = safe_get(optimized_data, 'page_load_time', 0)
    reg_page_load = safe_get(regular_data, 'page_load_time', 0)
    opt_detection = safe_get(optimized_data, 'detection_time', 0)
    reg_detection = safe_get(regular_data, 'detection_time', 0)
    
    # Safely get content lengths
    opt_content_len = safe_get(comparison_data, 'optimized_content_length', 0)
    reg_content_len = safe_get(comparison_data, 'regular_content_length', 0)
    content_diff = opt_content_len - reg_content_len
    
    metrics = [
        ["Metric", "Optimized", "Regular", "Difference"],
        ["Total Time", f"{opt_time:.2f}s", f"{reg_time:.2f}s", 
         f"{time_diff:.2f}s ({time_diff_pct:.2f}%)"],
        ["Total Tokens", f"{opt_tokens:,}", f"{reg_tokens:,}", 
         f"{tokens_diff:,}"],
        ["API Calls", str(opt_api), str(reg_api), 
         str(opt_api - reg_api)],
        ["Steps Taken", str(opt_steps), str(reg_steps), 
         str(opt_steps - reg_steps)],
        ["Page Load Time", f"{opt_page_load:.3f}s", f"{reg_page_load:.3f}s", 
         f"{opt_page_load - reg_page_load:.3f}s"],
        ["Detection Time", f"{opt_detection:.3f}s", f"{reg_detection:.3f}s", 
         f"{opt_detection - reg_detection:.3f}s"],
        ["Content Length", str(opt_content_len), str(reg_content_len), 
         str(content_diff)]
    ]
    
    # Hide axes
    plt.axis('tight')
    plt.axis('off')
    
    # Create table
    table = plt.table(
        cellText=metrics,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight header row
    for j in range(4):
        table[(0, j)].set_facecolor('#e6e6e6')
        table[(0, j)].set_text_props(weight='bold')
    
    # Title for the table
    plt.title('Summary of Key Metrics', pad=20)
    
    # Add source URLs as subtitle
    opt_url = safe_get(optimized_data, 'url', 'N/A')
    reg_url = safe_get(regular_data, 'url', 'N/A')
    url_text = f"Comparing: {opt_url} (optimized) vs {reg_url} (regular)"
    plt.figtext(0.5, 0.01, url_text, ha='center', fontsize=8)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{prefix}_summary_table.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_task_description_image(optimized_data, output_dir, prefix):
    """Create an image with the task description"""
    task_description = safe_get(optimized_data, 'task', 'Task description not available')
    
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    
    # Create text box with task description
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.5, 0.5, f"Task:\n\n{task_description}", 
             fontsize=10, ha='center', va='center', wrap=True,
             bbox=props, transform=plt.gca().transAxes)
    
    plt.title('Task Description', fontsize=14)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{prefix}_task_description.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Visualize AI Optimizer comparison data')
    parser.add_argument('comparison_path', help='Path to comparison_results_standardized.json file')
    parser.add_argument('optimized_path', help='Path to optimized_metrics_standardized.json file')
    parser.add_argument('regular_path', help='Path to regular_metrics_standardized.json file')
    parser.add_argument('--output-dir', '-o', default='ai_optimizer_visualizations',
                        help='Output directory for visualization images')
    parser.add_argument('--prefix', '-p', default='',
                        help='Prefix for output filenames')
    parser.add_argument('--debug', action='store_true',
                        help='Print loaded JSON data structure for debugging')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Add separator to prefix if provided
    prefix = args.prefix
    if prefix and not prefix.endswith('_'):
        prefix = f"{prefix}_"
    
    # Load data
    comparison_data = load_json_data(args.comparison_path)
    optimized_data = load_json_data(args.optimized_path)
    regular_data = load_json_data(args.regular_path)
    
    # Debug option to print data structure
    if args.debug:
        print("=== COMPARISON DATA KEYS ===")
        print(list(comparison_data.keys()))
        
        print("\n=== OPTIMIZED DATA KEYS ===")
        print(list(optimized_data.keys()))
        
        print("\n=== REGULAR DATA KEYS ===")
        print(list(regular_data.keys()))
    
    # Generate all visualizations as separate files
    generated_files = []
    
    print("Generating visualizations...")
    
    # Create task description image
    task_file = create_task_description_image(optimized_data, output_dir, prefix)
    generated_files.append(task_file)
    print(f"Created task description image: {task_file}")
    
    # Create time comparison visualization
    time_file = plot_time_comparison(comparison_data, optimized_data, regular_data, output_dir, prefix)
    generated_files.append(time_file)
    print(f"Created time comparison visualization: {time_file}")
    
    # Create token usage visualization
    token_file = plot_tokens_comparison(comparison_data, optimized_data, regular_data, output_dir, prefix)
    generated_files.append(token_file)
    print(f"Created token usage visualization: {token_file}")
    
    # Create API calls and steps visualization
    api_file = plot_api_calls_and_steps(comparison_data, optimized_data, regular_data, output_dir, prefix)
    generated_files.append(api_file)
    print(f"Created API calls and steps visualization: {api_file}")
    
    # Create detection metrics visualization
    detection_file = plot_detection_metrics(optimized_data, regular_data, output_dir, prefix)
    generated_files.append(detection_file)
    print(f"Created detection metrics visualization: {detection_file}")
    
    # Create actions timeline visualization
    actions_file = plot_actions_timeline(optimized_data, regular_data, output_dir, prefix)
    generated_files.append(actions_file)
    print(f"Created actions timeline visualization: {actions_file}")
    
    # Create summary table visualization
    summary_file = create_summary_table(comparison_data, optimized_data, regular_data, output_dir, prefix)
    generated_files.append(summary_file)
    print(f"Created summary table visualization: {summary_file}")
    
    print(f"\nAll visualizations saved to directory: {output_dir}")
    print(f"Total files generated: {len(generated_files)}")

if __name__ == "__main__":
    main()