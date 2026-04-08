import time
import httpx


class PharmaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=10.0)

    def _post_with_retry(self, endpoint, json=None, retries=3):
        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries):
            try:
                response = self.client.post(url, json=json)
                response.raise_for_status()
                return response.json()

            except Exception:
                if attempt == retries - 1:
                    return None
                time.sleep(2)

    def reset(self):
        return self._post_with_retry("/reset")

    def step(self, action):
        return self._post_with_retry("/step", json=action)