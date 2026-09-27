"""Resume crossing-witness equalization without revisiting completed attempts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from test_coxeter import S5_GAUSS, S5_SEEDS
from equistick.coxeter import verify_map

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'scripts/12_equalize_crossings.py'
SCRIPT = ROOT / 'scripts/11_crossing_changes.py'


class CrossingEqualizationTests(unittest.TestCase):
    def test_unreached_target_distinguishes_timed_out_sources(self):
        spec = importlib.util.spec_from_file_location('equalize_stage', STAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = {'sources_reached': [], 'attempts': []}
        self.assertEqual(module._target_state(entry, True, True),
                         'not_reached_in_completed_sources')
        self.assertEqual(module._target_state(entry, True, False), 'not_reached')

    def test_archived_selected_witness_can_be_loaded_without_scratch(self):
        spec = importlib.util.spec_from_file_location('equalize_stage', STAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        coords = np.arange(30, dtype=np.float64).reshape(10, 3)
        source_id, name = 'crss:Example', 'K13n501'
        source_record = {'source_sha256': 'source-digest',
                         'coordinates_sha256': 'input-coordinates-digest'}
        witness = {'move_index': 7, 'move': {'vertex': 2},
                   'coordinates': coords.tolist(), 'crossing': {'edge': 5}}
        archive = {'code_sha256': 'scan-digest', 'witnesses': {name: {
            'source_id': source_id, 'source_sha256': 'source-digest',
            'source_coordinates_sha256': 'input-coordinates-digest',
            'candidate_coordinates_sha256': hashlib.sha256(coords.tobytes()).hexdigest(),
            **witness}}}
        self.assertEqual(module._archived_witness(archive, 'scan-digest', source_id,
                         source_record, name), witness)

    def test_qualified_map_accepts_the_diagrams_actual_strand_count(self):
        spec = importlib.util.spec_from_file_location('equalize_stage', STAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        proof = {'passed': True, 'generation_passed': True, 'relations_checked': 13,
                 'strand_count': 13, 'group': 'S5', 'group_order': 120}
        self.assertTrue(module._qualified({'exact_rank_four_passed': True, 'exact_map': proof}))

    def test_existing_certification_has_a_finished_status_not_pending(self):
        spec = importlib.util.spec_from_file_location('equalize_stage', STAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = {'existing_certified': 'known-hash', 'sources_reached': ['eddy:Source'],
                 'exact_rank_four_passed': True, 'attempts': []}
        self.assertEqual(module._target_state(entry, True), 'certified_existing')

    def test_existing_root_certified_polygon_is_not_queued_for_equalization(self):
        spec = importlib.util.spec_from_file_location('equalize_stage', STAGE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        known = module.existing_certified('K13n288', ROOT / 'results/ten_new_stick_knots')
        self.assertEqual(known.name, 'K13n288_equilateral_10sticks.txt')
        self.assertIsNone(module.existing_certified('K15n52944', ROOT / 'results/ten_new_stick_knots'))

    def test_invalid_witness_is_logged_and_not_retried_or_published(self):
        known = ROOT / 'results/ten_new_stick_knots/K15n59007_equilateral_10sticks.txt'
        source_id = 'ours:K15n59007'
        source = {'id': source_id, 'label': 'K15n59007', 'kind': 'ours',
                  'path': str(known), 'sha256': hashlib.sha256(known.read_bytes()).hexdigest()}
        code = [SCRIPT, ROOT / 'equistick/crossing_change.py',
                ROOT / 'equistick/coxeter.py', ROOT / 'equistick/crss.py']
        code_hash = hashlib.sha256(b''.join(path.read_bytes() for path in code)).hexdigest()
        coords = np.loadtxt(known)
        coord_hash = hashlib.sha256(coords.tobytes()).hexdigest()
        proof = verify_map(S5_GAUSS, S5_SEEDS, 'S5')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / 'scratch'
            scan_dir = scratch / code_hash[:16]
            scan_dir.mkdir(parents=True)
            result_path = scan_dir / (hashlib.sha256(f'source:{source_id}'.encode()).hexdigest()[:20]
                                      + '.output.json')
            witness = {'move_index': 1, 'move': {'vertex': 0,
                       'destination': coords[0].tolist()},
                       'coordinates': coords.tolist(), 'crossing': {'edge': 3}}
            result_path.write_text(json.dumps({'status': 'completed', 'source': source,
                'source_sha256': source['sha256'], 'coordinates_sha256': coord_hash,
                'scan': {'candidates': {'K15n41189': [witness]}}}))
            manifest = {'schema': 1, 'code_sha256': code_hash, 'inventory_count': 1,
                'completed_count': 1, 'candidate_budget_seconds': 12,
                'sources': {source_id: {'status': 'completed', 'source': source,
                    'source_sha256': source['sha256'], 'coordinates_sha256': coord_hash,
                    'four_bridge_reached': {'K15n41189': {'exact_rank_four_passed': True}}}},
                'reached_four_bridge': {'K15n41189': {source_id: {
                    'exact_rank_four_passed': True}}},
                'map_classifications': {'K15n41189': {'exact_rank_four_passed': True,
                    'exact_map': proof}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'pool.json').write_text(json.dumps({'results': [
                {'name': 'K15n41189', 'status': 'timeout'}]}))
            args = [sys.executable, str(STAGE), '--manifest', str(root / 'manifest.json'),
                    '--scratch', str(scratch), '--output', str(root / 'out.json'),
                    '--results', str(root / 'results'), '--pool-results', str(root / 'pool.json'),
                    '--candidate-budget', '12', '--max-attempts', '1']
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=25)
            first = json.loads((root / 'out.json').read_text())
            self.assertEqual(first['targets']['K15n41189']['status'],
                             'reached_not_certified_within_budget')
            self.assertEqual(first['targets']['K15n41189']['attempts'][0]['status'],
                             'invalid_crossing')
            self.assertEqual(list((root / 'results').glob('*.txt')), [])
            subprocess.run(args, check=True, capture_output=True, text=True, timeout=25)
            second = json.loads((root / 'out.json').read_text())
            self.assertEqual(len(second['targets']['K15n41189']['attempts']), 1)


if __name__ == '__main__':
    unittest.main()
