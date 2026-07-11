from __future__ import annotations

import pytest
from adapter_contract import AdapterContract

from recallops.adapters.local import LocalIndexAdapter


class TestLocalAdapterContract(AdapterContract):
    @pytest.fixture()
    def adapter(self, tmp_path):
        return LocalIndexAdapter(tmp_path / "index")
