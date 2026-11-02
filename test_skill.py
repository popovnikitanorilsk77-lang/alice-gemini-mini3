import importlib.util
import json
import pathlib
import unittest
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'api' / 'index.py'
spec = importlib.util.spec_from_file_location('alice', SCRIPT)
alice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alice)


def sample(utterance, new=False, state=None):
    return {
        'session': {'skill_id': 'skill-123', 'new': new},
        'request': {'original_utterance': utterance},
        'state': {'session': state or {}}
    }


class TestAliceGemini(unittest.TestCase):
    def test_greeting(self):
        status, response = alice.process_alice(sample('', True), 'abc', 'skill-123')
        self.assertEqual(status, 200)
        self.assertIn('Привет', response['response']['text'])

    def test_invalid_skill(self):
        status, response = alice.process_alice(sample('привет'), 'abc', 'wrong-skill')
        self.assertEqual(status, 403)

    def test_exit(self):
        status, response = alice.process_alice(sample('выход'), 'abc', 'skill-123')
        self.assertTrue(response['response']['end_session'])

    def test_reset(self):
        status, response = alice.process_alice(sample('сброс', state={'h': [['hi', 'yes']]}), 'abc', 'skill-123')
        self.assertEqual(response['session_state']['h'], [])

    def test_generate_payload_and_history(self):
        with patch.object(alice, '_ask_gemini', return_value='Привет!') as ask:
            status, response = alice.process_alice(sample('Кто ты?'), 'abc', 'skill-123')
            self.assertEqual(status, 200)
            self.assertEqual(ask.call_args.args[0], 'Кто ты?')
            self.assertEqual(response['session_state']['h'], [['Кто ты?', 'Привет!']])
        self.assertLess(len(json.dumps(response['session_state'], ensure_ascii=False).encode('utf-8')), 1024)

    def test_large_history_fits(self):
        pairs = [['а' * 160, 'б' * 220]] * 10
        compact = alice._history(pairs)
        self.assertLessEqual(len(json.dumps({'h': compact}, ensure_ascii=False).encode('utf-8')), 850)

    def test_timeout(self):
        with patch.object(alice, '_ask_gemini', side_effect=TimeoutError):
            _, response = alice.process_alice(sample('Объясни сложное'), 'abc', 'skill-123')
            self.assertIn('долго', response['response']['text'])


if __name__ == '__main__':
    unittest.main()
