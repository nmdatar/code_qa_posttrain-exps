import hashlib
from pathlib import Path
import tempfile
import unittest
from dataset_builder.source_evidence import inspect_reference


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = {}
        self.put('src/a.py', b'one\ntwo\nthree\n')

    def put(self, name, data):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.inventory[name] = hashlib.sha256(data).hexdigest()

    def inspect(self, text):
        return inspect_reference(text, self.root, self.inventory)

    def test_ranges_and_unique_basename(self):
        for text in ('`src/a.py` (lines 1-3)', 'src/a.py:1-3', 'lines 1-3 in `a.py`',
                     'https://github.com/org/repo/blob/abcdef/src/a.py#L1-L3'):
            with self.subTest(text=text):
                r = self.inspect(text)
                self.assertEqual(r['status'], 'source_spans_verified')
                self.assertEqual(r['evidence'][0]['path'], 'src/a.py')
                self.assertEqual(r['evidence'][0]['end_line'], 3)
                self.assertEqual(r['semantic_validation'], 'not_performed')

    def test_real_citation_variants(self):
        for text in ('(src/a.py: line 1-3)', '`src/a.py`, lines 1 to 3',
                     '`./src/a.py` at lines 1—3', '`src/a.py` on line 2',
                     '```1:3:src/a.py\nsource contents\n```'):
            with self.subTest(text=text):
                r = self.inspect(text)
                self.assertEqual(r['status'], 'source_spans_verified')
                self.assertEqual(r['evidence'][0]['path'], 'src/a.py')

    def test_sentence_final_filename(self):
        r = self.inspect('The implementation is in src/a.py.')
        self.assertEqual(r['file_references'][0]['path'], 'src/a.py')
        self.assertEqual(self.inspect('src/a.py.method()')['file_references'], [])

    def test_file_only_does_not_invent_span(self):
        r = self.inspect('See `a.py`, and then lines 3-7 discuss an unrelated function.')
        self.assertEqual(r['evidence'], [])
        self.assertEqual(r['file_references'][0]['kind'], 'file_reference')
        self.assertNotIn('start_line', r['file_references'][0])
        self.assertEqual(r['status'], 'no_resolvable_citations')

    def test_ambiguous_missing_and_partial(self):
        self.put('other/a.py', b'data\n')
        r = self.inspect('a.py:1 and missing.py:1 plus src/a.py:2')
        self.assertEqual(r['status'], 'partial')
        self.assertEqual([x['reason'] for x in r['unresolved']], ['ambiguous_basename', 'not_tracked'])
        self.assertEqual(len(r['evidence']), 1)

    def test_bad_ranges(self):
        for citation in ('src/a.py:0-1', 'src/a.py:3-2', 'src/a.py:1-4'):
            self.assertEqual(self.inspect(citation)['unresolved'][0]['reason'], 'line_range_out_of_bounds')

    def test_binary_and_encoding(self):
        self.put('binary.py', b'abc\x00def')
        self.put('bad.py', b'\xff')
        r = self.inspect('binary.py:1 bad.py:1')
        self.assertEqual([x['reason'] for x in r['unresolved']], ['binary_file', 'non_utf8_file'])

    def test_hash_mismatch(self):
        (self.root / 'src/a.py').write_text('changed\n')
        self.assertEqual(self.inspect('src/a.py:1')['unresolved'][0]['reason'], 'file_hash_mismatch')

    def test_symlink_component(self):
        (self.root / 'alias').symlink_to(self.root / 'src', target_is_directory=True)
        self.inventory['alias/a.py'] = self.inventory['src/a.py']
        self.assertEqual(self.inspect('alias/a.py:1')['unresolved'][0]['reason'], 'symlink_not_allowed')

    def test_confinement(self):
        r = self.inspect('../a.py:1 /src/a.py:1 C:\\a.py:1')
        self.assertEqual(len(r['unresolved']), 3)
        self.assertTrue(all(x['reason'] == 'unsafe_path' for x in r['unresolved']))

    def test_missing_tracked_file_and_dedup(self):
        self.inventory['gone.py'] = '0' * 64
        self.assertEqual(self.inspect('gone.py:1')['unresolved'][0]['reason'], 'missing_file')
        r = self.inspect('src/a.py:1-2 then `src/a.py` (lines 1-2)')
        self.assertEqual(len(r['evidence']), 1)

    def test_no_context_inference(self):
        r = self.inspect('See src/a.py for details. Another method is implemented at lines 20-25.')
        self.assertEqual(r['evidence'], [])
        self.assertEqual(len(r['file_references']), 1)


if __name__ == '__main__':
    unittest.main()
