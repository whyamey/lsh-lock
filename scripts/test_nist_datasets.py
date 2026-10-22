"""Dataset driver checks using synthetic readings and no native NIST dependency."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nist_datasets as driver


class DatasetDriverTests(unittest.TestCase):
    def setup_data(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        for name in ['face_embeddings_noah_nir','face_dedup','iris_embeddings/test','finger_embeddings']:
            for label in ['a','b']:
                folder=root/name/label;folder.mkdir(parents=True);(folder/'r').write_text('0,1,0,1')
        return SimpleNamespace(data_root=root,face_templates=root/'face_dedup',output_root=root/'out',
            nist_bin='/unused',trials=1000,seed=67,jobs=2,timeout=180,no_controls=False,keep_inputs=False)

    def test_all_inputs_preflight_before_any_native_run(self):
        args=self.setup_data();(args.data_root/'finger_embeddings/b/r').write_text('bad')
        with patch.object(driver,'validate_options'),patch.object(driver,'run') as native:
            with self.assertRaisesRegex(ValueError,'binary'):
                driver.run_datasets(args)
            native.assert_not_called()
        self.assertFalse(args.output_root.exists())

    def test_three_datasets_and_explicit_face_export_forwarded(self):
        args=self.setup_data()
        with patch.object(driver,'validate_options'),patch.object(driver,'run') as native:
            driver.run_datasets(args)
        self.assertEqual(native.call_count,3)
        calls=[x.args[0] for x in native.call_args_list]
        self.assertEqual(calls[0].templates,args.face_templates.resolve())
        self.assertEqual(calls[1].templates,(args.data_root/'iris_embeddings/test').resolve())
        self.assertEqual([x.output.name for x in calls],['face','iris','finger'])
        self.assertTrue(all(x.seed==67 and x.trials==1000 and x.jobs==2 for x in calls))
        self.assertTrue(json.loads((args.output_root/'complete.json').read_text())['controls'])

    def test_failure_leaves_no_completion_marker(self):
        args=self.setup_data()
        with patch.object(driver,'validate_options'),patch.object(driver,'run',side_effect=RuntimeError('native')):
            with self.assertRaisesRegex(RuntimeError,'native'):
                driver.run_datasets(args)
        self.assertFalse((args.output_root/'complete.json').exists())


if __name__=='__main__':
    unittest.main()
