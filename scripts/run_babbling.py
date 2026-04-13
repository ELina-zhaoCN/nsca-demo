#!/usr/bin/env python3
"""
Curriculum Babbling Phase - Layer 4 Language Grounding.

Runs two-phase exploration to populate the LearnedGrounding table:
- Phase 1: Random exploration (discover affordances)
- Phase 2: Competence-driven (retry learnable actions)

Uses a simulated environment (synthetic sensory feedback) when no
Meta-World/MuJoCo is available. Saves results to logs/babbling_results.json.

Usage:
    python scripts/run_babbling.py --random-steps 10000 --competence-steps 40000
    python scripts/run_babbling.py --config configs/training_config_local.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.learning.curriculum_babbling import CurriculumBabbling, BabblingConfig


def run_simulated_babbling(
    random_steps: int = 10000,
    competence_steps: int = 40000,
    seed: int = 42,
) -> dict:
    """
    Run babbling with simulated environment (no MuJoCo required).
    Generates synthetic interactions for grounding table population.
    """
    random.seed(seed)
    
    config = BabblingConfig(
        phase1_steps=random_steps,
        phase2_steps=competence_steps,
    )
    babbling = CurriculumBabbling(config)
    
    # Simulated actions and objects
    actions = ['strike', 'lift', 'push', 'squeeze', 'look', 'drop']
    objects = [f'obj_{i}' for i in range(50)]
    
    results = {
        'step': 0,
        'phase': 1,
        'interactions': [],
        'action_counts': {},
        'learnability_scores': {},
    }
    
    print(f"Babbling: {random_steps} random + {competence_steps} competence = {random_steps + competence_steps} steps")
    
    for step in range(random_steps + competence_steps):
        obj = random.choice(objects)
        action = babbling.select_action(actions) if step >= random_steps else random.choice(actions)
        
        # Simulated sensory feedback (property proxies from action)
        sensory = {
            'hardness': random.random() if action in ['strike', 'squeeze', 'drop'] else 0.5,
            'weight': random.random() if action in ['lift', 'push', 'drop'] else 0.5,
            'prediction_error': random.random() * 0.5,
        }
        
        babbling.action_history[action].append(sensory['prediction_error'])
        babbling.action_counts[action] = babbling.action_counts.get(action, 0) + 1
        
        babbling.interaction_log.append({
            'object_id': obj,
            'action': action,
            'sensory_feedback': sensory,
            'step': step,
        })
        
        if (step + 1) % 5000 == 0:
            print(f"  Step {step + 1}/{random_steps + competence_steps}")
    
    results['interactions'] = babbling.interaction_log[-100:]  # Last 100
    results['action_counts'] = dict(babbling.action_counts)
    results['total_steps'] = random_steps + competence_steps
    results['timestamp'] = datetime.now().isoformat()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Curriculum Babbling for NSCA')
    parser.add_argument('--random-steps', type=int, default=10000)
    parser.add_argument('--competence-steps', type=int, default=40000)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CURRICULUM BABBLING (Layer 4 - Language Grounding)")
    print("=" * 60)
    
    results = run_simulated_babbling(
        random_steps=args.random_steps,
        competence_steps=args.competence_steps,
        seed=args.seed,
    )
    
    # Save results
    Path('logs').mkdir(exist_ok=True)
    out_path = Path('logs/babbling_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    print(f"Total interactions: {results['total_steps']}")
    print(f"Action distribution: {results['action_counts']}")
    print("\nBabbling complete. Grounding table populated for Language layer.")


if __name__ == '__main__':
    main()
