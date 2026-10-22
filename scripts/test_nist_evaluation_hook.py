"""Companion NIST hook behavior with a synthetic driver; no lab/native jobs.

The fixture reproduces the parent's foreground cache/exit-status contract.
Applying the complete patch to the reviewed parent snapshot is a separate check.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def patched_hook():
    lines = []
    for line in (ROOT / 'docs/sample-ind-correct-nist.patch').read_text().splitlines():
        if line.startswith('+++'):
            continue
        if line.startswith(('+', ' ')):
            lines.append(line[1:])
    match = re.search(r'^g_nist\(\) \{\n.*?^\}', '\n'.join(lines), re.M | re.S)
    if not match:
        raise ValueError('No complete NIST function in companion patch')
    return match.group(0)


class NistEvaluationHookTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='nist hook ')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkout = self.root / 'checkout with spaces'
        scripts = self.checkout / 'scripts'
        scripts.mkdir(parents=True)
        self.events = self.root / 'calls.jsonl'
        (scripts / 'nist_datasets.py').write_text('''import json, os, pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = pathlib.Path(args['--output-root'])
out.mkdir(parents=True, exist_ok=False)
with pathlib.Path(os.environ['TEST_EVENTS']).open('a') as f:
    f.write(json.dumps(args)+'\\n')
if os.environ.get('TEST_FAIL') == '1':
    (out/'partial.txt').write_text('preserve failed evidence')
    sys.exit(7)
(out/'complete.json').write_text('{}')
print('completed synthetic NIST driver')
''')
        self.out = self.root / 'results'
        self.out.mkdir()
        self.harness = self.root / 'hook.sh'
        self.harness.write_text('''set -u
NRUN=0; NSKIP=0; NFAIL=0
run() {
  local name=$1; shift
  if [ -s "$OUT/$name.txt" ]; then NSKIP=$((NSKIP+1)); return; fi
  if [ "$DRY" = 1 ]; then echo "would run $name"; NRUN=$((NRUN+1)); return; fi
  if "$@" > "$OUT/$name.part" 2> "$OUT/$name.err"; then
    mv "$OUT/$name.part" "$OUT/$name.txt"; NRUN=$((NRUN+1))
  else NFAIL=$((NFAIL+1)); fi
}
''' + patched_hook() + '''
g_nist
echo "$NRUN run, $NSKIP skipped, $NFAIL failed"
[ "$NFAIL" -eq 0 ] || exit 1
''')
        self.env = {**os.environ, 'TEST_EVENTS': str(self.events),
                    'NIST_LSH_ROOT': str(self.checkout), 'NIST_PYTHON': sys.executable,
                    'D': str(self.root / 'biometric inputs'), 'FACED': '/fixture/original face',
                    'LSH': str(self.checkout / 'target/release/lsh-lock'),
                    'OUT': str(self.out), 'DRY': '0'}
        for name in ['NIST_TAG', 'NIST_FACE', 'NIST_TRIALS', 'NIST_SEED', 'NIST_JOBS',
                     'NIST_TIMEOUT', 'NIST_EA_NON_IID', 'TEST_FAIL']:
            self.env.pop(name, None)

    def invoke(self, **changes):
        return subprocess.run(['bash', str(self.harness)], cwd=self.root,
                              env={**self.env, **changes}, text=True,
                              capture_output=True, timeout=15)

    def calls(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_legacy_log_and_identical_rerun_never_reuse_results(self):
        legacy = self.out / 'nist_full.txt'
        legacy.write_text('old result for a different export')
        first, second = self.invoke(), self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn('1 run, 0 skipped, 0 failed', first.stdout)
        self.assertIn('1 run, 0 skipped, 0 failed', second.stdout)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]['--output-root'], calls[1]['--output-root'])
        self.assertEqual(legacy.read_text(), 'old result for a different export')
        for call in calls:
            self.assertTrue((Path(call['--output-root']) / 'complete.json').is_file())

    def test_changed_export_and_all_execution_options_are_forwarded_fresh(self):
        self.assertEqual(self.invoke().returncode, 0)
        changed = self.invoke(NIST_FACE='/fixture/new 1024 export', NIST_TRIALS='17',
                              NIST_SEED='999', NIST_JOBS='2', NIST_TIMEOUT='12.5',
                              NIST_EA_NON_IID='/fixture/new native binary', NIST_TAG='new-export')
        self.assertEqual(changed.returncode, 0, changed.stderr)
        first, second = self.calls()
        for key, value in {'--face-templates': '/fixture/new 1024 export', '--trials': '17',
                           '--seed': '999', '--jobs': '2', '--timeout': '12.5',
                           '--nist-bin': '/fixture/new native binary'}.items():
            self.assertEqual(second[key], value)
        self.assertNotEqual(first['--output-root'], second['--output-root'])
        self.assertTrue(Path(second['--output-root']).parent.name.startswith('nist_new-export_'))

    def test_failed_run_is_preserved_and_retry_uses_a_new_directory(self):
        failed = self.invoke(TEST_FAIL='1')
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('1 failed', failed.stdout)
        failed_path = Path(self.calls()[0]['--output-root'])
        evidence = failed_path / 'partial.txt'
        self.assertEqual(evidence.read_text(), 'preserve failed evidence')
        retried = self.invoke()
        self.assertEqual(retried.returncode, 0, retried.stderr)
        retry_path = Path(self.calls()[1]['--output-root'])
        self.assertNotEqual(failed_path, retry_path)
        self.assertTrue(evidence.is_file())
        self.assertTrue((retry_path / 'complete.json').is_file())

    def test_dry_run_never_allocates_attempt_or_launches_driver(self):
        result = self.invoke(DRY='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('would run nist_full_NEW', result.stdout)
        self.assertFalse(self.events.exists())
        self.assertEqual(list(self.out.iterdir()), [])

    def test_invalid_tag_is_rejected_before_allocating_or_launching(self):
        for tag in ['../outside', 'two words', 'x;echo']:
            result = self.invoke(NIST_TAG=tag)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('NIST_TAG', result.stderr)
        self.assertFalse(self.events.exists())
        self.assertEqual(list(self.out.iterdir()), [])

    def test_source_root_is_derived_from_selected_binary(self):
        del self.env['NIST_LSH_ROOT']
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]['--face-templates'], self.env['FACED'])

    def test_missing_wrapper_is_rejected_before_creating_attempt(self):
        result = self.invoke(NIST_LSH_ROOT='/fixture/missing checkout')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('NIST wrapper missing', result.stderr)
        self.assertFalse(self.events.exists())
        self.assertEqual(list(self.out.iterdir()), [])

    def test_parallel_invocations_receive_distinct_attempt_directories(self):
        processes = [subprocess.Popen(['bash', str(self.harness)], cwd=self.root,
                                     env=self.env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE) for _ in range(2)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn('1 run, 0 skipped', stdout)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]['--output-root'], calls[1]['--output-root'])


if __name__ == '__main__':
    unittest.main()
