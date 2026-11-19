"""NPU helpers shared by the detect / realtime / benchmark scripts.

The only thing here is the VX-delegate graph cache. The i.MX 8M Plus VX delegate
recompiles the network into an NPU binary on the first inference of each process
(~30-60 s). Pointing it at a persistent cache directory makes every launch after
the first one start almost instantly. These environment variables must be set
*before* the delegate is loaded, so call enable_npu_cache() first.
"""
import os


def enable_npu_cache(cache_dir):
    """Persist the compiled NPU graph under cache_dir. Returns the resolved path.
    Pass a falsy cache_dir to leave caching disabled (env untouched)."""
    if not cache_dir:
        return None
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # VeriSilicon / i.MX VX delegate graph-binary cache (NXP i.MX ML User's Guide).
    os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
    os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = cache_dir
    return cache_dir
