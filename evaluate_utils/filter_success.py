#!/usr/bin/env python3
"""
Cyclic Peptide Success Filtering Script (Parallel Version)
根据条件2（head-to-tail环化）从生成结果中筛选出满足几何约束的环肽。

条件: N端C原子与C端C原子距离在 3~8 Å 之间

用法:
    python filter_success.py --input <results.jsonl> --output <good_results.jsonl>
"""

import argparse
import json
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter cyclic peptide generation results by head-to-tail distance (3~8 A)."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to results.jsonl file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output good_results.jsonl file"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-entry details"
    )
    return parser.parse_args()


def process_pdb(pdb_abs_path, lig_chain):
    """Worker function: parse a single PDB file and return distance."""
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('peptide', pdb_abs_path)
        if len(structure) == 0:
            return None
        chain = structure[0][lig_chain]
        residues = list(chain.get_residues())
        first_atom = residues[0]['C']
        last_atom = residues[-1]['C']
        distance = np.linalg.norm(first_atom.coord - last_atom.coord)
        return distance
    except Exception as e:
        return None


def process_batch(batch_data):
    """Process a batch of lines and return results."""
    results = []
    for item in batch_data:
        pdb_abs_path = item['pdb_abs_path']
        lig_chain = item['lig_chain']
        if os.path.exists(pdb_abs_path):
            distance = process_pdb(pdb_abs_path, lig_chain)
            if distance is not None:
                results.append({
                    'index': item['index'],
                    'peptide_id': item['peptide_id'],
                    'pdb_rel_path': item['pdb_rel_path'],
                    'python_object': item['python_object'],
                    'distance': distance,
                    'is_valid': 3 <= distance <= 8
                })
    return results


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_workers = args.workers or mp.cpu_count()

    print("=" * 50)
    print("  Cyclic Peptide Filtering (Condition 2)")
    print("=" * 50)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Workers: {n_workers}")
    print(f"  Rule:   N-term C to C-term C distance 3~8 A")
    print("-" * 50)

    # Read and preprocess all lines
    input_dir = os.path.dirname(args.input)
    all_peptide_list = []
    batch_data = []

    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            python_object = json.loads(line.strip())

            peptide_id = python_object['id']
            pdb_path = python_object['gen_pdb']
            pdb_rel_path = pdb_path.lstrip('./')
            if os.path.isabs(pdb_path):
                pdb_abs_path = pdb_path
            else:
                project_root = input_dir
                while project_root and not os.path.exists(os.path.join(project_root, 'generate.py')):
                    project_root = os.path.dirname(project_root)
                if not project_root:
                    project_root = os.getcwd()
                pdb_abs_path = os.path.normpath(os.path.join(project_root, pdb_rel_path))
            lig_chain = python_object.get('lig_chain', 'L')

            if peptide_id not in all_peptide_list:
                all_peptide_list.append(peptide_id)

            batch_data.append({
                'index': idx,
                'peptide_id': peptide_id,
                'pdb_rel_path': pdb_rel_path,
                'python_object': python_object,
                'pdb_abs_path': pdb_abs_path,
                'lig_chain': lig_chain
            })

    total_counter = len(batch_data)
    print(f"  Total samples to process: {total_counter}")
    print("-" * 50)

    # Split into chunks for parallel processing
    chunk_size = max(1, total_counter // (n_workers * 4))
    batches = [batch_data[i:i + chunk_size] for i in range(0, len(batch_data), chunk_size)]

    print(f"  Processing in {len(batches)} batches with {n_workers} workers...")

    # Process batches in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"  Batch {futures[future]} failed: {e}", file=sys.stderr)

    # Sort results by original index to maintain order
    all_results.sort(key=lambda x: x['index'])

    # Aggregate results
    meet_peptide_dic = {}
    meet_peptide_dic_path = {}
    counter_meet = 0
    distance_list = [r['distance'] for r in all_results]

    with open(args.output, 'w', encoding='utf-8') as good_file:
        for result in all_results:
            counter_meet += 1 if result['is_valid'] else 0
            peptide_id = result['peptide_id']
            pdb_rel_path = result['pdb_rel_path']

            if result['is_valid']:
                if peptide_id not in meet_peptide_dic:
                    meet_peptide_dic[peptide_id] = 1
                    meet_peptide_dic_path[peptide_id] = [pdb_rel_path]
                    good_file.write(json.dumps(result['python_object']) + '\n')
                else:
                    meet_peptide_dic[peptide_id] += 1
                    if pdb_rel_path not in meet_peptide_dic_path[peptide_id]:
                        meet_peptide_dic_path[peptide_id].append(pdb_rel_path)
                        good_file.write(json.dumps(result['python_object']) + '\n')

    print("-" * 50)
    print(f"Total samples:             {total_counter}")
    print(f"Unique targets:           {len(all_peptide_list)}")
    print(f"Meet condition (3~8 A):    {counter_meet}")
    if distance_list:
        print(f"Mean N-C distance:         {np.mean(distance_list):.4f} A")
    print(f"Output saved to:           {args.output}")


if __name__ == '__main__':
    main()
