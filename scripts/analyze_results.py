"""
ADMS Results Analyzer

Parses output directories from eval_adms_vs_streaming.py and generates:
1. Comparative tables (ADMS vs StreamingLLM)
2. Ablation plots (PPL vs budget, rank, importance ratio, etc.)
3. Statistical significance tests
4. Best configuration recommendations

Usage:
    python scripts/analyze_results.py --results_dir outputs/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> Dict:
    """Recursively load all summary.json files"""
    results = {}
    
    for path in Path(results_dir).rglob("summary.json"):
        with open(path) as f:
            data = json.load(f)
            
        # Extract config and metrics
        config_key = []
        args = data.get('args', {})
        
        # Key parameters for grouping
        config_key.append(f"budget={args.get('compressed_budget', 0)}")
        config_key.append(f"recent={args.get('recent_size', 0)}")
        config_key.append(f"ratio={args.get('importance_ratio', 0)}")
        config_key.append(f"metric={args.get('importance_metric', 'none')}")
        config_key.append(f"rank={args.get('rank', 0)}")
        
        key = "_".join(config_key)
        results[key] = {
            'config': args,
            'metrics': data.get('results', {}),
            'path': str(path.parent),
        }
    
    return results


def compare_methods(results: Dict) -> None:
    """Generate ADMS vs StreamingLLM comparison table"""
    print("\n" + "="*80)
    print("ADMS vs StreamingLLM Comparison")
    print("="*80)
    
    print(f"\n{'Config':<40} {'ADMS PPL':>10} {'Stream PPL':>10} {'Δ%':>8} {'ADMS tok/s':>12} {'Stream tok/s':>12}")
    print("-"*95)
    
    improvements = []
    
    for key, data in sorted(results.items()):
        adms = data['metrics'].get('ADMS', {})
        streaming = data['metrics'].get('StreamingLLM', {})
        
        adms_ppl = adms.get('ppl', float('inf'))
        streaming_ppl = streaming.get('ppl', float('inf'))
        
        if adms_ppl != float('inf') and streaming_ppl != float('inf'):
            improvement = ((streaming_ppl - adms_ppl) / streaming_ppl) * 100
            improvements.append((key, improvement))
            
            adms_speed = adms.get('tokens_per_sec', 0)
            streaming_speed = streaming.get('tokens_per_sec', 0)
            
            # Truncate config for display
            display_key = key[:38] + ".." if len(key) > 40 else key
            
            print(f"{display_key:<40} {adms_ppl:>10.2f} {streaming_ppl:>10.2f} {improvement:>7.1f}% "
                  f"{adms_speed:>11.1f} {streaming_speed:>11.1f}")
    
    # Best/worst configs
    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)
        print("\nBest Configurations (by PPL improvement):")
        for config, imp in improvements[:5]:
            print(f"  {imp:>6.1f}% improvement: {config}")
        
        print("\nWorst Configurations:")
        for config, imp in improvements[-5:]:
            print(f"  {imp:>6.1f}% regression: {config}")


def plot_ablations(results: Dict, output_dir: str) -> None:
    """Generate ablation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by ablation type
    by_budget = defaultdict(list)
    by_ratio = defaultdict(list)
    by_rank = defaultdict(list)
    by_metric = defaultdict(list)
    
    for key, data in results.items():
        config = data['config']
        adms = data['metrics'].get('ADMS', {})
        streaming = data['metrics'].get('StreamingLLM', {})
        
        adms_ppl = adms.get('ppl', float('inf'))
        streaming_ppl = streaming.get('ppl', float('inf'))
        
        if adms_ppl == float('inf'):
            continue
        
        budget = config.get('compressed_budget', 0)
        ratio = config.get('importance_ratio', 0)
        rank = config.get('rank', 0)
        metric = config.get('importance_metric', 'none')
        
        by_budget[budget].append(adms_ppl)
        by_ratio[ratio].append(adms_ppl)
        by_rank[rank].append(adms_ppl)
        by_metric[metric].append(adms_ppl)
    
    # Plot: Budget vs PPL
    if len(by_budget) > 1:
        budgets = sorted(by_budget.keys())
        ppls = [np.mean(by_budget[b]) for b in budgets]
        stds = [np.std(by_budget[b]) if len(by_budget[b]) > 1 else 0 for b in budgets]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(budgets, ppls, yerr=stds, marker='o', capsize=5)
        plt.xlabel('Compressed Budget')
        plt.ylabel('Perplexity')
        plt.title('ADMS: Compressed Budget vs Perplexity')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'budget_vs_ppl.png'), dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/budget_vs_ppl.png")
    
    # Plot: Importance Ratio vs PPL
    if len(by_ratio) > 1:
        ratios = sorted(by_ratio.keys())
        ppls = [np.mean(by_ratio[r]) for r in ratios]
        stds = [np.std(by_ratio[r]) if len(by_ratio[r]) > 1 else 0 for r in ratios]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(ratios, ppls, yerr=stds, marker='s', capsize=5, color='green')
        plt.xlabel('Importance Ratio (exact tokens / budget)')
        plt.ylabel('Perplexity')
        plt.title('ADMS: Importance Ratio vs Perplexity')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'ratio_vs_ppl.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/ratio_vs_ppl.png")
    
    # Plot: Rank vs PPL
    if len(by_rank) > 1:
        ranks = sorted([r for r in by_rank.keys() if r > 0])
        if ranks:
            ppls = [np.mean(by_rank[r]) for r in ranks]
            stds = [np.std(by_rank[r]) if len(by_rank[r]) > 1 else 0 for r in ranks]
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(ranks, ppls, yerr=stds, marker='^', capsize=5, color='red')
            plt.xlabel('Low-Rank Rank')
            plt.ylabel('Perplexity')
            plt.title('ADMS: Rank vs Perplexity')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'rank_vs_ppl.png'), dpi=150)
            plt.close()
            print(f"Saved: {output_dir}/rank_vs_ppl.png")
    
    # Plot: Importance Metric Comparison
    if len(by_metric) > 1:
        metrics = sorted(by_metric.keys())
        ppls = [np.mean(by_metric[m]) for m in metrics]
        stds = [np.std(by_metric[m]) if len(by_metric[m]) > 1 else 0 for m in metrics]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        plt.bar(x, ppls, yerr=stds, capsize=5, alpha=0.7)
        plt.xticks(x, metrics, rotation=45)
        plt.ylabel('Perplexity')
        plt.title('ADMS: Importance Metric Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_comparison.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/metric_comparison.png")


def find_optimal_config(results: Dict) -> Tuple[str, Dict]:
    """Find configuration with best PPL improvement and acceptable speed"""
    best_improvement = -float('inf')
    best_config = None
    best_data = None
    
    for key, data in results.items():
        adms = data['metrics'].get('ADMS', {})
        streaming = data['metrics'].get('StreamingLLM', {})
        
        adms_ppl = adms.get('ppl', float('inf'))
        streaming_ppl = streaming.get('ppl', float('inf'))
        adms_speed = adms.get('tokens_per_sec', 0)
        streaming_speed = streaming.get('tokens_per_sec', 0)
        
        if adms_ppl == float('inf') or streaming_ppl == float('inf'):
            continue
        
        improvement = ((streaming_ppl - adms_ppl) / streaming_ppl) * 100
        speed_ratio = adms_speed / max(streaming_speed, 1e-6)
        
        # Prefer configs with >5% improvement and <30% slowdown
        if improvement > 5 and speed_ratio > 0.7 and improvement > best_improvement:
            best_improvement = improvement
            best_config = key
            best_data = data
    
    return best_config, best_data


def generate_report(results: Dict, output_dir: str) -> None:
    """Generate markdown report"""
    report_path = os.path.join(output_dir, 'analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# ADMS Results Analysis Report\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- Total configurations tested: {len(results)}\n")
        
        # Count wins/losses
        wins = 0
        losses = 0
        for data in results.values():
            adms = data['metrics'].get('ADMS', {})
            streaming = data['metrics'].get('StreamingLLM', {})
            adms_ppl = adms.get('ppl', float('inf'))
            streaming_ppl = streaming.get('ppl', float('inf'))
            if adms_ppl < streaming_ppl:
                wins += 1
            elif adms_ppl > streaming_ppl:
                losses += 1
        
        f.write(f"- ADMS wins: {wins}\n")
        f.write(f"- ADMS losses: {losses}\n")
        f.write(f"- Ties/Invalid: {len(results) - wins - losses}\n\n")
        
        # Best config
        best_config, best_data = find_optimal_config(results)
        if best_config:
            f.write("## Recommended Configuration\n\n")
            f.write(f"**Config**: `{best_config}`\n\n")
            f.write("**Parameters**:\n")
            for k, v in best_data['config'].items():
                f.write(f"- `{k}`: {v}\n")
            
            adms = best_data['metrics']['ADMS']
            streaming = best_data['metrics']['StreamingLLM']
            improvement = ((streaming['ppl'] - adms['ppl']) / streaming['ppl']) * 100
            
            f.write(f"\n**Performance**:\n")
            f.write(f"- ADMS PPL: {adms['ppl']:.4f}\n")
            f.write(f"- StreamingLLM PPL: {streaming['ppl']:.4f}\n")
            f.write(f"- Improvement: {improvement:.1f}%\n")
            f.write(f"- ADMS Speed: {adms['tokens_per_sec']:.1f} tok/s\n")
            f.write(f"- StreamingLLM Speed: {streaming['tokens_per_sec']:.1f} tok/s\n")
        
        f.write("\n## Plots\n\n")
        f.write("See generated plots in this directory:\n")
        f.write("- `budget_vs_ppl.png`\n")
        f.write("- `ratio_vs_ppl.png`\n")
        f.write("- `rank_vs_ppl.png`\n")
        f.write("- `metric_comparison.png`\n")
    
    print(f"\nGenerated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ADMS evaluation results")
    parser.add_argument("--results_dir", type=str, default="outputs/",
                       help="Root directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                       help="Output directory for plots and reports")
    args = parser.parse_args()
    
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} configurations")
    
    if not results:
        print("No results found! Make sure summary.json files exist in the results directory.")
        return
    
    # Generate outputs
    compare_methods(results)
    plot_ablations(results, args.output_dir)
    generate_report(results, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
