"""Tests for demo data generation and download helpers."""

from types import SimpleNamespace

from smart_agri_gnn.acquire import create_demo_inputs, download_world_bank_climate
from smart_agri_gnn.config import PipelineConfig


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = b"country_code,yield\nUGA,2.5\n"

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=30):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


def test_create_demo_inputs(tmp_path):
    config = PipelineConfig(data_root=tmp_path, use_demo_data=True)
    config.ensure_directories()
    create_demo_inputs(config, n_regions=5)
    assert config.climate_path.is_file()
    assert config.yield_path.is_file()
    assert config.soil_path.is_file()


def test_download_climate_writes_csv(tmp_path):
    config = PipelineConfig(data_root=tmp_path)
    config.ensure_directories()
    client = FakeClient(
        {
            "properties": {
                "parameter": {
                    "T2M": {"2022": 21.5, "2023": 22.1},
                }
            }
        }
    )
    download_world_bank_climate(config, client=client)
    text = config.climate_path.read_text(encoding="utf-8")
    assert "2022" in text and "21.5" in text
    assert client.calls[0]["params"]["latitude"] == config.climate_latitude
