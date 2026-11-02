import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.core.question_bank import extract_question_texts, parse_question_file
from src.core.rag_engine import _paragraphs_with_line_ranges


class TestDocQAExtract(unittest.TestCase):
    def test_extract_questions_numbered_blocks(self):
        text = """一、单选题
1. 关于A的描述，哪项正确？
- A. 选项A
- B. 选项B
2. 关于B的描述，哪项正确？
- A. 选项A
- B. 选项B
"""
        qs = extract_question_texts(text)
        self.assertEqual(len(qs), 2)
        self.assertIn("1.", qs[0])
        self.assertIn("选项A", qs[0])
        self.assertIn("2.", qs[1])

    def test_paragraph_split_on_question_starts(self):
        lines = [
            "一、单选题\n",
            "1. 第一题？\n",
            "- A. a\n",
            "- B. b\n",
            "2. 第二题？\n",
            "- A. a\n",
        ]
        paras = _paragraphs_with_line_ranges(lines)
        self.assertGreaterEqual(len(paras), 3)
        self.assertTrue(any(p[2].startswith("1.") for p in paras))
        self.assertTrue(any(p[2].startswith("2.") for p in paras))

    def test_parse_sample_question_bank(self):
        path = os.path.join(BASE_DIR, "data", "样题.txt")
        questions = parse_question_file(path)
        self.assertEqual(len(questions), 72)
        self.assertEqual(questions[0].qid, 1)
        self.assertEqual(len(questions[0].options), 4)
        self.assertEqual(len(questions[50].options), 6)
        self.assertEqual(len(questions[60].options), 2)
        self.assertEqual(len(questions[70].options), 10)


if __name__ == "__main__":
    unittest.main()

