"""对真实 PDF、向量索引及中文查询进行端到端校验。"""

import re
import unittest

from rag_kb import DEFAULT_PDF, KnowledgeBase, extract_records


class KnowledgeBaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase()

    def test_pdf_records_match_reviewed_plain_text(self):
        records, pages = extract_records(DEFAULT_PDF)
        plain = DEFAULT_PDF.with_suffix('.txt').read_text(encoding='utf-8')
        starts = list(re.finditer(r'(?m)^(\d{2})\. ', plain))
        self.assertEqual(pages, 5)
        self.assertEqual(len(records), 30)
        self.assertEqual(len(starts), 30)
        for i, record in enumerate(records):
            end = starts[i+1].start() if i+1 < len(starts) else len(plain)
            original = plain[starts[i].start():end]
            # PDF 自动换行和字间空格不改变材料、数量、版本、URL 内容。
            self.assertEqual(re.sub(r'\s+', '', record['text']), re.sub(r'\s+', '', original))

    def test_pdf_pages_and_critical_recipe_notes(self):
        by_id = {r['recipe_number']: r for r in self.kb.records}
        self.assertEqual(by_id[7]['crafting_station'], '熔炉。')
        self.assertIn('1.4.5.7', by_id[28]['notes'])
        self.assertIn('魔力再生手环', by_id[28]['notes'])
        self.assertIn('反光墨镜', by_id[14]['notes'])
        for record in self.kb.records:
            self.assertEqual(record['page_start'], record['page_end'])
            self.assertTrue(1 <= record['page_start'] <= 5)

    def test_semantic_retrieval_top_three(self):
        cases = [
            ('泰拉闪耀靴怎么合成？', '泰拉闪耀靴'),
            ('十字章护身符需要什么材料？', '十字章护身符'),
            ('魔法手铐怎么做，旧版本和现在有什么区别？', '魔法手铐'),
            ('哪个配方需要20个黑曜石？', '黑曜石骷髅头'),
            ('强力手套和血肉指虎能合成什么？', '狂战士手套'),
            ('天界壳怎么制作？', '天界壳'),
            ('大自然的恩赐可以合成哪个饰品？', '魔力花'),
        ]
        for question, target in cases:
            with self.subTest(question=question):
                results = self.kb.search(question, top_k=3)
                self.assertIn(target, [r['title'] for r in results])
                self.assertTrue(all(-1.001 <= r['score'] <= 1.001 for r in results))

    def test_reject_empty_query_and_invalid_k(self):
        with self.assertRaises(ValueError):
            self.kb.search('   ')
        with self.assertRaises(ValueError):
            self.kb.search('泰拉闪耀靴', top_k=0)


if __name__ == '__main__':
    unittest.main()
