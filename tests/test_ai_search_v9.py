import importlib
import unittest

from fastapi.testclient import TestClient


class AiSearchV9StartupTests(unittest.TestCase):
    def test_health_endpoint(self):
        module = importlib.import_module("ai_search_v9")
        client = TestClient(module.app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
