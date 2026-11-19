"""Additional streamed K-tail and eligibility coverage."""
import pytest
from test_w4a8_group_scales import test_group128_streamed_random_reference as check_reference


@pytest.mark.parametrize("k", [1536, 1664, 1792, 1920])
def test_scale_ring_edges(k):
    check_reference(k)


@pytest.mark.parametrize("k", [128, 256, 384, 512, 640, 768, 896])
def test_streamed_rejects_unsupported_short_k(k):
    # Existing streamed eligibility requires K >= 1024; do not silently
    # exercise a different kernel when checking the scale-ring boundaries.
    with pytest.raises(RuntimeError, match="streamed tile is unavailable"):
        check_reference(k)
