"""statetree: lossless state-tree BF16 weights with fused CUDA kernels.

    import statetree
    model = statetree.convert(model)                    # BF16 linears (q/k/v and gate/up fused), MoE experts, lm_head,
                                                        # embeddings
    statetree.prepare(model, batch_sizes=[1, 8, 512])   # optional: pick / autotune kernels ahead of time
    statetree.save_pretrained(model, "out_dir", tokenizer)
    model = statetree.load_pretrained("out_dir")
"""
from .format import decode_reference, encode
from .experts import TreeExperts
from .linear import TreeEmbedding, TreeLinear, TreeSlice
from .model import DEFAULT_FUSE, convert, load_pretrained, prepare, save_pretrained, size_report, tree_modules

__version__ = "0.5.2"
__all__ = ["TreeLinear", "TreeSlice", "TreeEmbedding", "TreeExperts", "DEFAULT_FUSE", "convert", "prepare", "save_pretrained",
           "load_pretrained", "size_report", "tree_modules", "encode", "decode_reference"]
from .dense16 import Dense16Linear  # noqa: E402,F401  (ablation control)
