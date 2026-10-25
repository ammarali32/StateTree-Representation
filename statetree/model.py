"""Model-level API: convert a model's linear layers (and embeddings) to state trees, prepare/tune, save and load."""
import contextlib
import fnmatch
import json
import os
import time

import torch
import torch.nn as nn

from .linear import TreeEmbedding, TreeLinear, TreeSlice

FORMAT_NAME = "statetree"
FORMAT_VERSION = 3                                  # 3: leaf format v2 (format.LEAF_FORMAT); 2 is migrated on load
WEIGHTS_NAME = "model.safetensors"
INFO_NAME = "statetree.json"
# sibling Linears that read the same input: attention and MLP of Llama / Mistral / Qwen / Gemma blocks, and the input
# projections of Gated DeltaNet (Qwen3.5, Qwen3-Next) linear-attention blocks
DEFAULT_FUSE = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj"),
                "in_proj_all": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
                "in_proj_qkvzba": ("in_proj_qkvz", "in_proj_ba")}


def _match(name, patterns):
    return any(fnmatch.fnmatch(name, p) or name.split(".")[-1] == p or name == p for p in patterns)


def _parent(model, name):
    *path, attr = name.split(".")
    mod = model
    for p in path:
        mod = getattr(mod, p)
    return mod, attr


def _eligible(mod):
    return (isinstance(mod, nn.Linear) and mod.weight.dtype == torch.bfloat16 and mod.out_features % 16 == 0
            and mod.in_features % 64 == 0 and mod.weight.device.type != "meta")


def tree_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, TreeLinear)]


@torch.no_grad()
def convert(model, include=None, exclude=(), fuse=DEFAULT_FUSE, embeddings=True, verify=True, verbose=True):
    """Store the model's BF16 linear layers (out % 16 == 0, in % 64 == 0) as state trees, in place.

    fuse: {fused name: part names}. Sibling Linears that read the same input (q/k/v, gate/up) become one TreeLinear
          '<parent>.<fused name>' and the parts become TreeSlice views of its output: one kernel launch instead of
          three (two). None keeps every layer separate.
    embeddings: also store nn.Embedding tables as trees (exact rows decoded on lookup). A table tied to the output
          layer reuses the output layer's tree, so a tied model keeps one compressed copy.
    include / exclude: module-name patterns (fnmatch, or a bare attribute name such as "lm_head").
    verify: decode every converted matrix and check it is bit-identical to the original weights.
    A model loaded with an accelerate device map (device_map="auto" across GPUs) is converted in place, each layer on
    its own GPU; layers offloaded to CPU / disk (weights on the meta device) are left in BF16 and reported."""
    t0 = time.perf_counter()
    cfg = getattr(model, "config", None)
    out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    in_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tied = (isinstance(out_emb, nn.Linear) and isinstance(in_emb, nn.Embedding) and out_emb.weight is in_emb.weight)

    def wanted(name):
        return not ((exclude and _match(name, exclude)) or (include and not _match(name, include)))

    todo, skipped = set(), []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and wanted(name):
            (todo.add(name) if _eligible(mod) else skipped.append(name))
    st = dict(before=0, after=0, layers=0, fused=0, emb=0)
    new_out = None

    # 1) fused groups of siblings
    for pname, parent in list(model.named_modules()):
        for fname, parts in (fuse or {}).items():
            full = [f"{pname}.{p}" if pname else p for p in parts]
            if hasattr(parent, fname) or not all(f in todo for f in full):
                continue
            lins = [getattr(parent, p) for p in parts]
            hb = [lin.bias is not None for lin in lins]
            if len({lin.in_features for lin in lins}) != 1 or (any(hb) and not all(hb)):
                continue
            W = torch.cat([lin.weight for lin in lins], 0)
            b = torch.cat([lin.bias for lin in lins], 0) if all(hb) else None
            tl = TreeLinear.from_weight(W, b, verify=verify)
            del W
            setattr(parent, fname, tl)
            start = 0
            for p, lin in zip(parts, lins):
                setattr(parent, p, TreeSlice(tl, start, lin.out_features, len(parts)))
                start += lin.out_features
                st["before"] += 2 * lin.weight.numel()
            st["after"] += tl.storage_bytes()
            st["layers"] += len(parts)
            st["fused"] += 1
            todo -= set(full)

    # 2) the remaining linear layers (including BF16 layers fused earlier by statetree.bench.fuse_bf16, whose
    #    TreeSlice views are re-pointed to the new TreeLinear)
    slices = {}
    for _, m in model.named_modules():
        if isinstance(m, TreeSlice) and not isinstance(m.fused, TreeLinear):
            slices.setdefault(id(m.fused), (m.fused, []))[1].append(m)
    for name in sorted(todo):
        parent, attr = _parent(model, name)
        lin = getattr(parent, attr)
        tl = TreeLinear.from_linear(lin, verify=verify)
        setattr(parent, attr, tl)
        rec = slices.pop(id(lin), None)
        if rec is not None and rec[0] is lin:
            for m in rec[1]:
                object.__setattr__(m, "fused", tl)
        if lin is out_emb:
            new_out = tl
        st["before"] += 2 * lin.weight.numel()
        st["after"] += tl.storage_bytes()
        st["layers"] += 1
        del lin
    del slices
    if tied and new_out is not None and cfg is not None:
        cfg.tie_word_embeddings = False          # tying is now handled by TreeEmbedding(source=lm_head tree)

    # 3) embeddings
    if embeddings:
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Embedding) or not wanted(name) or mod.max_norm is not None:
                continue
            w = mod.weight
            if w.dtype != torch.bfloat16 or w.device.type == "meta" or mod.embedding_dim % 64 or mod.num_embeddings % 16:
                continue
            if mod is in_emb and tied:
                if new_out is None:
                    continue                     # the output layer stayed BF16: keep sharing its table
                te = TreeEmbedding(mod.num_embeddings, mod.embedding_dim, source=new_out)
            else:
                te = TreeEmbedding.from_embedding(mod, verify=verify)
                st["after"] += te.table.storage_bytes()
            parent, attr = _parent(model, name)
            setattr(parent, attr, te)
            st["before"] += 2 * w.numel()
            st["emb"] += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(f"statetree: {st['layers']} linear layers ({st['fused']} fused groups) and {st['emb']} embeddings in "
              f"{time.perf_counter() - t0:.1f}s: {st['before'] / 2**30:.3f} GiB -> {st['after'] / 2**30:.3f} GiB"
              + (f", skipped {len(skipped)} (not bf16, shape, or offloaded by a device map)" if skipped else "")
              + (", all bit-exact" if verify else ""))
    return model


def prepare(model, batch_sizes=(1,), verbose=False):
    """Build side tables and pick (autotune, cached in ~/.cache/statetree) the kernel for each batch size, for every
    TreeLinear. Call it before capturing CUDA graphs."""
    from .linear import BIG, _autotune_on, _state, _tuned_get, bucket_of
    groups = {}
    for _, m in tree_modules(model):
        if m.tree.device.type != "cuda":           # offloaded by a device map: tuned when it first runs on the GPU
            continue
        m._esc_lists()
        groups.setdefault((m.tree.device, m.out_features, m.in_features), []).append(m)
    for (dev, M, N), mods in groups.items():
        for B in batch_sizes:
            bk = bucket_of(B)
            t0 = time.perf_counter()
            if bk != BIG and _autotune_on() and _tuned_get(_state(dev), M, N, bk) is None:
                mods[0].tune(bk, group=mods)                  # time the candidates across all layers of this shape
            for m in mods:
                m._plan(B)
            if verbose:
                p = mods[0]._plan(B)
                print(f"statetree: {M}x{N} batch {B}: {p[0]} {p[1]} ({time.perf_counter() - t0:.2f}s)")
    return model


def size_report(model):
    tl = tree_modules(model)
    tree_bytes = sum(m.storage_bytes() for _, m in tl)
    n_w = sum(m.in_features * m.out_features for _, m in tl)
    n_esc = sum(m.esc_rows.numel() for _, m in tl)
    other = 0
    seen = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        other += t.numel() * t.element_size()
    other -= tree_bytes
    return dict(tree_layers=len(tl), tree_weights=n_w, tree_bytes=tree_bytes, bf16_bytes=2 * n_w,
                bits_per_weight=8 * tree_bytes / max(n_w, 1), escape_fraction=n_esc / max(n_w, 1),
                other_bytes=other, total_bytes=tree_bytes + other)


# ----------------------------------------------------------------------------------------- save / load
def _structure(model):
    names = {id(m): n for n, m in tree_modules(model)}
    modules = {n: dict(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
               for n, m in tree_modules(model)}
    fused, embs = {}, {}
    for n, m in model.named_modules():
        if isinstance(m, TreeSlice):
            fused.setdefault(names[id(m.fused)], []).append([n, m.start, m.out_features])
        elif isinstance(m, TreeEmbedding):
            embs[n] = dict(num_embeddings=m.num_embeddings, embedding_dim=m.embedding_dim,
                           source=None if m.table is not None else names[id(m.source)])
    return modules, fused, embs


def save_pretrained(model, path, tokenizer=None):
    """Write config.json, statetree.json (module structure) and model.safetensors (tied tensors stored once)."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    sd = model.state_dict()
    first, aliases, tensors = {}, {}, {}
    for k, v in sd.items():
        key = (v.device, v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()),
               v.dtype) if v.numel() else None
        if key is not None and key in first:
            aliases[k] = first[key]
            continue
        if key is not None:
            first[key] = k
        tensors[k] = v.detach().contiguous()
    modules, fused, embs = _structure(model)
    info = dict(format=FORMAT_NAME, version=FORMAT_VERSION, modules=modules, fused=fused, embeddings=embs,
                aliases=aliases, dtype="bfloat16", architecture=type(model).__name__)
    save_file(tensors, os.path.join(path, WEIGHTS_NAME), metadata={"format": "pt", FORMAT_NAME: str(FORMAT_VERSION)})
    with open(os.path.join(path, INFO_NAME), "w") as fh:
        json.dump(info, fh, indent=1)
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "save_pretrained"):
        cfg.save_pretrained(path)
    gen = getattr(model, "generation_config", None)
    if gen is not None and hasattr(gen, "save_pretrained"):
        try:
            gen.save_pretrained(path)
        except Exception:                                                     # noqa: BLE001
            pass
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    return path


@contextlib.contextmanager
def _empty_weights():
    """Parameters are created on the meta device (no allocation, no init); buffers stay real (e.g. RoPE tables)."""
    old = nn.Module.register_parameter

    def register(self, name, param):
        old(self, name, param)
        if param is not None:
            p = self._parameters[name]
            self._parameters[name] = type(p)(p.to("meta"), requires_grad=p.requires_grad)
    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = old


def _device_of(key, dmap):
    """Device of a parameter / buffer name under an accelerate device map (longest matching module prefix)."""
    best, dev = -1, None
    for name, d in dmap.items():
        if (name == "" or key == name or key.startswith(name + ".")) and len(name) > best:
            best, dev = len(name), d
    return dev


def _torch_device(d):
    return torch.device("cuda", d) if isinstance(d, int) else torch.device(d)


def _device_map(model, device_map, max_memory):
    from accelerate import infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    if isinstance(device_map, dict):
        dmap = dict(device_map)
    else:
        no_split = list(getattr(model, "_no_split_modules", None) or [])
        if device_map in ("auto", "balanced", "balanced_low_0"):
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split,
                                             low_zero=device_map == "balanced_low_0")
        elif device_map != "sequential":
            raise ValueError(f"device_map={device_map!r}: expected auto, balanced, balanced_low_0, sequential or a dict")
        dmap = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split,
                                     offload_buffers=True)                # trees are buffers: they offload too
    if any(v == "disk" for v in dmap.values()):
        raise ValueError("statetree: disk offload is not supported; give 'cpu' more max_memory instead")
    # a tied input embedding reads the lm_head's tree directly (not through accelerate): keep that tree on a GPU
    names = {id(m): n for n, m in model.named_modules()}
    gpu = next((v for v in dmap.values() if not isinstance(v, str) or v.startswith("cuda")), 0)
    for _, m in model.named_modules():
        if isinstance(m, TreeEmbedding) and m.table is None and m.source is not None:
            src = names[id(m.source)]
            if _device_of(src, dmap) in ("cpu", "disk"):
                if src not in dmap:
                    raise ValueError(f"statetree: the tied table {src} would be offloaded; give the GPUs more memory")
                dmap[src] = gpu
    return dmap


def load_pretrained(path, device=None, device_map="auto", max_memory=None, model_cls=None, **config_kwargs):
    """Rebuild a converted Hugging Face model from save_pretrained output, without materialising BF16 weights.

    device: put the whole model on this one device (e.g. "cuda:1"), as in statetree <= 0.2.
    device_map (used when device is None; needs accelerate, else everything goes to cuda:0): as in transformers'
        from_pretrained, "auto" / "balanced" / "balanced_low_0" / "sequential" or an explicit {module name: device}.
        Layers are spread over the GPUs and accelerate's hooks move activations between them; modules mapped to
        "cpu" keep their trees in CPU memory and are streamed to the GPU for each forward (12 bits/weight instead of
        16 cross PCIe). max_memory caps each device, e.g. {0: "20GiB", 1: "20GiB", "cpu": "64GiB"}."""
    from safetensors.torch import load_file
    with open(os.path.join(path, INFO_NAME)) as fh:
        info = json.load(fh)
    if info.get("format") != FORMAT_NAME:
        raise ValueError(f"{path}: not a statetree checkpoint")
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(path, **config_kwargs)
    cls = model_cls or AutoModelForCausalLM
    with _empty_weights():
        try:
            model = cls.from_config(config, torch_dtype=torch.bfloat16)
        except TypeError:
            model = cls.from_config(config, dtype=torch.bfloat16)
    embs = info.get("embeddings", {})
    for name, e in embs.items():                     # embeddings first: an owned table is a TreeLinear inside them
        parent, attr = _parent(model, name)
        table = None if e["source"] else TreeLinear(e["embedding_dim"], e["num_embeddings"], device="meta")
        setattr(parent, attr, TreeEmbedding(e["num_embeddings"], e["embedding_dim"], table=table))
    for name, m in info["modules"].items():
        parent, attr = _parent(model, name)
        if not isinstance(getattr(parent, attr, None), TreeLinear):
            setattr(parent, attr, TreeLinear(m["in_features"], m["out_features"], bias=m["bias"], device="meta"))
    for fname, parts in info.get("fused", {}).items():
        tl = model.get_submodule(fname)
        for pname, start, size in parts:
            parent, attr = _parent(model, pname)
            setattr(parent, attr, TreeSlice(tl, start, size, len(parts)))
    for name, e in embs.items():
        if e["source"]:
            object.__setattr__(model.get_submodule(name), "source", model.get_submodule(e["source"]))
    dmap = None
    if device is None:
        try:
            import accelerate  # noqa: F401
        except ImportError:
            device_map = None
        if device_map is None:
            device = "cuda"
        else:
            dmap = _device_map(model, device_map, max_memory)
            gpus = {v for v in dmap.values() if v != "cpu"}
            if len(gpus) <= 1 and "cpu" not in dmap.values():          # one GPU, nothing offloaded: no hooks needed
                device, dmap = _torch_device(next(iter(gpus), 0)), None
    sd = load_file(os.path.join(path, WEIGHTS_NAME), device=str(device) if dmap is None else "cpu")
    if info.get("version", 2) < 3:                  # statetree <= 0.3 checkpoint: leaf format 1 -> 2
        from .format import RECORD_BYTES, leaves_v1_to_v2
        for k in list(sd):
            v = sd[k]
            if k.endswith("tree") and v.dtype == torch.uint8 and v.dim() == 3 and v.shape[-1] == RECORD_BYTES:
                sd[k] = leaves_v1_to_v2(v)
    for alias, src in info.get("aliases", {}).items():
        sd[alias] = sd[src]
    if dmap is not None:
        sd = {k: v.to(_torch_device(_device_of(k, dmap) or "cpu")) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    del sd
    if getattr(config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()
    if dmap is None:
        model.to(device)
    else:
        for name, t in list(model.named_buffers()):                   # buffers not in the file (e.g. RoPE tables)
            d = _torch_device(_device_of(name, dmap) or "cpu")
            if t.device != d and t.device.type != "meta":
                parent, attr = _parent(model, name)
                parent._buffers[attr] = t.to(d)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left or unexpected:
        raise RuntimeError(f"statetree load: uninitialised {left[:5]}, unexpected {unexpected[:5]}")
    if dmap is not None:
        from accelerate import dispatch_model
        # trees are buffers: offload_buffers keeps CPU-mapped trees on the CPU (streamed in per forward)
        model = dispatch_model(model, device_map=dmap, offload_buffers="cpu" in dmap.values(),
                               skip_keys=getattr(model, "_skip_keys_device_placement", None))
        model.hf_device_map = dmap                                  # as from_pretrained sets it (generate, caches)
    model.eval()
    return model
