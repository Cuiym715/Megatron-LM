"""CPU regression for native source reuse, measurement precedence and transfer.

Run: python3 tests/unit_tests/dspp/test_chunk_export.py
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    for name in ('megatron', 'megatron.core', 'megatron.core.datasets'):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / name.replace('.', '/'))]
        sys.modules[name] = module
spec = importlib.util.spec_from_file_location('chunk_export', ROOT/'tools/export_dspp_chunk_model.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
from megatron.core.datasets.dspp_chunk_cost import ChunkCostModel


class NativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT/'profiles/dspp-native-pp8'
        cls.payload = json.loads((cls.folder/'model.json').read_text())
        cls.model = ChunkCostModel(cls.payload)

    def test_rebuild_from_legacy_sources_is_reproducible(self):
        payload, diagnostics = exporter.export(self.folder/'sources.json')
        self.assertEqual(json.loads(json.dumps(payload)), self.payload)
        self.assertEqual(json.loads(json.dumps(diagnostics)),
                         json.loads((self.folder/'model.fit.json').read_text()))
        self.assertEqual(len(payload['groups']), 80)
        self.assertEqual(sum(len(g['exact']) for g in payload['groups']), 944)
        self.assertEqual(len(payload['p2p']['points']), 112)
        self.assertTrue(all(d['capacity_holdout_errors'] is None
                            for d in diagnostics if d['op'] in ('D','H')))

    def test_every_exported_measurement_bypasses_fitted_curve(self):
        # Deliberately corrupt the fitted coefficients; measured values must
        # still be returned unchanged, including W's capacity-only lookup.
        model = ChunkCostModel(self.payload)
        model.coefficients = {k: (999999.,)*7 for k in model.coefficients}
        for group in self.payload['groups']:
            for capacity, shape, ms in group['exact']:
                self.assertEqual(model.predict(group['stage'], group['chunk'], group['op'],
                                               capacity, tuple(map(tuple, shape))), ms)

    def test_transferred_shapes_are_predictions_not_pp8_measurements(self):
        for group in self.payload['groups']:
            key = group['stage'], group['chunk'], group['op']
            self.assertGreater(group['transferred_points'], 0)
            self.assertNotIn(6144, group['measured_capacities'])
            shape = ((6144,6144),) if group['op'] != 'W' else ((6144,0),)
            self.assertNotIn(key+(6144,shape), self.model.exact)
            expected = max(1e-9, sum(a*b for a,b in zip(
                group['coefficients'], exporter.features(6144,shape))))
            self.assertEqual(self.model.predict(*key,6144,shape), expected)

    def test_export_keeps_more_than_32_exact_points(self):
        base = json.loads((self.folder/'sources.json').read_text())
        source = exporter.read_json(self.folder/base['compute'][0]['path'])
        group = source['groups'][0]
        group['model']['exact'] = [dict(signature=f'c2048|2048:{2048+h}', median_ms=1+h/1000)
                                   for h in range(40)]
        source['groups'] = [group]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'compact.json'; path.write_text(json.dumps(source))
            entry = dict(path=str(path),execution=exporter.NATIVE,operations=['B_fused'])
            # Keep the genuine memory/P2P; isolate the >32 regression input.
            for key in ('compute','local_compute','memory'):
                for e in base.get(key,[]):
                    e['path'] = str((self.folder/e['path']).resolve())
            base['p2p'] = str((self.folder/base['p2p']).resolve())
            base.update(compute=[entry],local_compute=[])
            manifest = Path(tmp)/'sources.json'; manifest.write_text(json.dumps(base))
            model,_ = exporter.export(manifest)
        self.assertEqual(len(model['groups'][0]['exact']), 40)

    def test_memory_excludes_resident_inherited_from_other_model(self):
        manifest = json.loads((self.folder/'sources.json').read_text())
        native = [exporter.read_json(self.folder/e['path']) for e in manifest['memory'][:2]]
        historical = exporter.read_json(self.folder/manifest['memory'][-1]['path'])
        current = self.payload['memory']['stages'][0]
        self.assertEqual(current['resident_bytes'], max(m['stages'][0]['resident_bytes'] for m in native))
        self.assertLess(current['resident_bytes'], historical['stages'][0]['resident_bytes'])
        self.assertEqual(current['pending_bytes_per_token'], historical['stages'][0]['pending_bytes_per_token'])

    def test_native_declaration_does_not_relabel_canonical_samples(self):
        source = dict(arithmetic_modes=dict(fused='fused-canonical-global-query-order-v1'), groups=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'compact.json'; path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'diagnostic profiles'):
                exporter.collect([dict(path=str(path),execution=exporter.NATIVE,operations=['F'])],
                                 Path(tmp), {}, [])

    def test_holdout_does_not_use_held_capacity_as_transfer_anchor(self):
        p2 = (2048,((2048,0),)); p4 = (4096,((4096,0),)); p6 = (6144,((6144,0),))
        target = {p2:2., p4:400.}
        local = {p2:1., p4:1., p6:3.}
        seen = []
        def fit(x,y):
            seen.append(list(y))
            return [1.,0.,0.,0.,0.,0.,0.]
        with patch.object(exporter,'fit',side_effect=fit):
            exporter.capacity_holdout(target,local)
        self.assertEqual(seen, [[1200.,400.],[6.,2.]])


if __name__ == '__main__':
    unittest.main()
