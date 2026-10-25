"""vLLM integration: a '--representation statetree' flag that stores every BF16 linear layer losslessly as a state tree.

statetree is a weight *representation*, not a quantization: the weights decode to their exact BF16 bits. It therefore
has its own flag and composes with --quantization (layers a quantization method claims stay quantized; the remaining
BF16 linear layers become state trees).

Installed as a vLLM general plugin (entry point 'vllm.general_plugins' -> statetree.vllm_plugin:register), which vLLM
loads before it builds its argument parser:

    vllm serve unsloth/Llama-3.2-1B --dtype bfloat16 --representation statetree

    import statetree.vllm_plugin                         # Python API: registers the 'representation' argument
    LLM(model="unsloth/Llama-3.2-1B", dtype="bfloat16", representation="statetree")

Equivalent without the flag: --additional-config '{"representation": "statetree"}' (or additional_config=... in
Python), or the environment variable VLLM_REPRESENTATION=statetree.

Works with vLLM's defaults and parallelism: chunked prefill (large token batches decode the tree into a scratch matrix
for cuBLAS; vocab-sized tables run in 512-row slices instead, e.g. prompt logprobs), prefix caching and any KV-cache
dtype (the representation only touches weights), tensor parallelism (each rank encodes its own shard: column- and
row-parallel layers, vocab-parallel embedding / lm_head; a shard whose shape does not fit the 16 x 64 blocks stays
BF16) and pipeline parallelism (each stage encodes the layers it owns).

The BF16 checkpoint is loaded as usual (tensor-parallel sharding and vLLM's fused q/k/v and gate/up layers included);
each linear layer is then encoded on the GPU, checked bit-exact, and its BF16 copy freed before vLLM sizes the KV cache.
The input embedding and lm_head are stored the same way (a tied pair shares one tree).
Kernel plans and workspaces for every CUDA-graph batch size are prepared right after loading (autotuned the first time
a shape is seen, cached in ~/.cache/statetree/tune.json).

Environment: STATETREE_VLLM_VERIFY=0 skips the load-time bit-exactness check; STATETREE_VLLM_EMBEDDINGS=0 keeps the
embedding and lm_head in BF16; STATETREE_AUTOTUNE=0 uses heuristics.
"""
REPRESENTATIONS = ("bf16", "statetree", "dense16")    # dense16: ablation control


def register():
    """vLLM general-plugin entry point (idempotent): adds '--representation' and the state-tree linear layers."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        return
    from . import _vllm
    _vllm.install()


register()
