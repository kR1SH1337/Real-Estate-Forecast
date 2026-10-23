"""Reference scenario, separation from data paths, and scenario boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launches_model', ROOT / 'model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class LaunchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/'examples/synthetic_input.json').read_text(encoding='utf-8'))

    def test_synthetic_scenario(self):
        result = model.run(self.data)
        self.assertEqual([r['year'] for r in result['annual']],list(range(2026,2031)))
        self.assertEqual(result['historical_validation']['n'],9)
        self.assertTrue(all(r['forward_scenario_launches']>0 for r in result['annual']))
        self.assertIn('SYNTHETIC',result['source_notes'][0])

    def test_no_anticipation_and_no_adjustment(self):
        result = model.run(self.data,anticipation=0)
        self.assertTrue(all(abs(r['anticipation_effect'])<1e-8 for r in result['annual']))
        result = model.run(self.data,adjustment=0)
        first = result['annual'][0]['forward_scenario_launches']
        self.assertTrue(all(r['forward_scenario_launches']==first for r in result['annual']))

    def test_invalid_weights(self):
        for kwargs in [{'adjustment':1.1},{'anticipation':-0.1},{'ratio':0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.run(self.data,**kwargs)

    def test_cli_from_other_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'nested'/'forecast.json'
            proc = subprocess.run([sys.executable,str(ROOT/'model.py'),
                                   '--input-json',str(ROOT/'examples/synthetic_input.json'),
                                   '--output-json',str(output)],cwd=folder,
                                  capture_output=True,text=True,check=True)
            result=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result,json.loads(proc.stdout))
            self.assertEqual(result['annual'][0]['year'],2026)


if __name__ == '__main__':
    unittest.main()
