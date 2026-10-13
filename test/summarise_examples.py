"""Collate the three separate demonstrations into concise proposal result tables.

Run from the repository root: python test/summarise_examples.py
Missing runs are marked pending; rerun once they finish. No dialogue is exported.
"""
import json
from pathlib import Path
import pandas as pd

RUNS = {"HFC": ("output/hfc_scored", "output/hfc_mpnet"),
        "Rental": ("output/rental_full", "output/rental_full"),
        "CaSiNo": ("output/casino_full", "output/casino_full")}


def summarise_examples():
    """Write coverage, primary importance statistics, and model/baseline metrics."""
    output = Path('output/proposal_results')
    output.mkdir(parents=True, exist_ok=True)
    coverage, importance, performance = [], [], []
    for dataset, (scoring, training) in RUNS.items():
        scoring, training = Path(scoring), Path(training)
        if not (scoring/'scoring_summary.json').exists():
            coverage.append({'dataset': dataset, 'status': 'pending'})
            continue
        counts = json.loads((scoring/'scoring_summary.json').read_text())
        row = {'dataset': dataset, 'status': 'complete', **{k: v for k, v in counts.items() if isinstance(v, (int, float))}}
        row['candidate_percent_of_eligible_turns'] = 100 * counts['candidate_unique_turns']/counts['unique_eligible_turns']
        config = json.loads((training/'configuration.json').read_text())
        row.update({f'{key}_conversations': value for key, value in config['split_counts'].items()})
        coverage.append(row)
        table = pd.read_csv(scoring/'importance_summary.csv')
        table = table[table.target_relation.ne('own')].copy()
        table.insert(0, 'dataset', dataset);importance.append(table)
        metrics = pd.read_csv(training/'metrics.csv');metrics.insert(0, 'dataset', dataset);performance.append(metrics)
    pd.DataFrame(coverage).to_csv(output/'dataset_summary.csv',index=False)
    if importance:pd.concat(importance,ignore_index=True).to_csv(output/'importance_summary.csv',index=False)
    if performance:pd.concat(performance,ignore_index=True).to_csv(output/'prediction_metrics.csv',index=False)
    (output/'README.txt').write_text(
        'Preliminary descriptive demonstration; datasets trained separately.\n'
        'Importance = absolute consecutive-prefix prediction update, in outcome units.\n'
        'Primary summaries use shared and counterpart outcomes; own outcomes stay in each run.\n'
        'Candidate rule: positive importance >= empirical outcome/relation/class 95th percentile; ties retained.\n'
        'This rule selects an upper tail by design, not statistically significant or independently validated turning points.\n'
        'Unique candidate turns are deduplicated across outcomes; flags count turn-outcome records.\n'
        'Continuous importance uses training-set standard deviation units; classification uses probability changes.\n'
        'First turns have undefined updates and are excluded from importance distributions.\n'
        'Prediction metrics include model and training-only baselines; inspect these before claiming meaningful discovery.\n'
        'CaSiNo uses ordered 1-5 ratings and joint points = sum of supplied individual points.\n'
        'Dataset reuse is not evidence that an HFC-trained model transfers to human conversations.\n')
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)

if __name__=='__main__':
    summarise_examples()
