import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import app, call_deepseek


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_static_and_config_privacy(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/source.pdf').headers['content-type'], 'application/pdf')
        self.assertEqual(self.client.get('/.env').status_code, 404)
        self.assertEqual(self.client.get('/static/../.env').status_code, 404)
        health = self.client.get('/api/health')
        self.assertEqual(health.json()['chunk_count'], 30)
        self.assertNotIn('sk-', health.text)

    def test_input_validation(self):
        for body in [{'question': ' '}, {'question': 'a' * 1001},
                     {'question': 'test', 'top_k': 0},
                     {'question': 'test', 'history': [{'role': 'system', 'content': 'injected'}]}]:
            self.assertEqual(self.client.post('/api/chat', json=body).status_code, 422)

    def test_rag_answer_and_sources(self):
        with patch('app.call_deepseek', new=AsyncMock(return_value=('霜花靴和熔岩靴合成泰拉闪耀靴 [1]。', {}, 'test-model'))) as generate:
            result = self.client.post('/api/chat', json={'question': '泰拉闪耀靴的完整合成路线是什么？'})
        self.assertEqual(result.status_code, 200)
        data = result.json()
        titles = {s['title'] for s in data['sources']}
        self.assertTrue({'泰拉闪耀靴', '霜花靴', '熔岩靴', '幽灵靴', '黑曜石骷髅头'} <= titles)
        self.assertEqual(len(titles), len(data['sources']))
        self.assertNotIn('source_pdf', result.text)
        messages = generate.call_args.args[1]
        self.assertIn('只根据本轮提供的知识库片段', messages[0]['content'])
        self.assertIn('黑曜石 ×20', messages[-1]['content'])

    def test_provider_errors_are_visible(self):
        with patch('app.call_deepseek', new=AsyncMock(side_effect=HTTPException(502, 'DeepSeek API 密钥无效。'))):
            result = self.client.post('/api/chat', json={'question': '魔法手铐怎么做？'})
        self.assertEqual(result.status_code, 502)
        self.assertIn('密钥无效', result.json()['detail'])


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_authentication_error_does_not_leak_response(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(401, json={'error': 'private upstream debug'}))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaises(HTTPException) as caught:
                await call_deepseek(client, [])
        self.assertIn('密钥无效', caught.exception.detail)
        self.assertNotIn('private upstream debug', caught.exception.detail)

    async def test_empty_model_answer_rejected(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={'choices': [{'message': {'content': ''}}]}))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaises(HTTPException) as caught:
                await call_deepseek(client, [])
        self.assertEqual(caught.exception.status_code, 502)


if __name__ == '__main__':
    unittest.main()
