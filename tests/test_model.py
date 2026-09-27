import asyncio
import json
import unittest

import httpx

from asyncroll.model import GenerationError, VLLMBackend, initial_messages
from asyncroll.workload import Problem


class ModelTest(unittest.TestCase):
    def test_persistent_async_client_schema_and_overlap(self):
        async def exercise():
            requests = []
            active, peak = 0, 0

            async def respond(request):
                nonlocal active, peak
                requests.append(json.loads(request.content))
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.02)
                active -= 1
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": '{"type":"final","answer":"5"}'}}],
                    "usage": {"completion_tokens": 8}})

            backend = VLLMBackend("http://test", "model", transport=httpx.MockTransport(respond))
            client = backend.client
            problem = Problem("p", "2+3", [{"name": "add"}])
            try:
                results = await asyncio.gather(*(backend.generate(problem, initial_messages(problem), 0)
                                                 for _ in range(3)))
                self.assertEqual(peak, 3)
                self.assertIs(backend.client, client)
                self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
                self.assertEqual(results[0].usage["completion_tokens"], 8)
                self.assertTrue(all(r.request_seconds >= 0.02 for r in results))
            finally:
                await backend.aclose()
            self.assertTrue(client.is_closed)
        asyncio.run(exercise())

    def test_invalid_response_and_total_timeout_are_distinct(self):
        async def exercise():
            problem = Problem("p", "1")
            for slow in (False, True):
                async def respond(request):
                    if slow:
                        await asyncio.sleep(1)
                    return httpx.Response(200, json={"choices": [{"message": {"content": "bad json"}}]})
                backend = VLLMBackend("http://test", "model", timeout=.02,
                                      transport=httpx.MockTransport(respond))
                try:
                    with self.assertRaises(GenerationError) as caught:
                        await backend.generate(problem, initial_messages(problem), 0)
                    self.assertEqual(caught.exception.timings["response_received"], not slow)
                finally:
                    await backend.aclose()
        asyncio.run(exercise())

    def test_http_error_counts_as_received_response(self):
        async def exercise():
            backend = VLLMBackend("http://test", "model", transport=httpx.MockTransport(
                lambda request: httpx.Response(503)))
            try:
                with self.assertRaises(GenerationError) as caught:
                    await backend.generate(Problem("p", "1"), [], 0)
                self.assertTrue(caught.exception.timings["response_received"])
                self.assertEqual(caught.exception.timings["http_status"], 503)
            finally:
                await backend.aclose()
        asyncio.run(exercise())
