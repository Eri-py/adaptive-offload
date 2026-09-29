# Learnings — flower-model-finetuning

## Task 2 — mypy on `model.classifier[i]` assignment

Indexing/assigning into a torchvision model's `.classifier` (a
`nn.Sequential`) fails strict mypy even though `torchvision.*` has
`ignore_missing_imports = true`: the attribute type resolves through
`nn.Module.__getattr__` (`Tensor | Module`, not `Any`), so `classifier[i]`
and reassigning it hit `union-attr`/`operator`/`assignment` errors, not just
a simple `no-any-return`. Fix: `cast(nn.Sequential, model.classifier)` once,
then index/assign normally (`nn.Sequential` has proper `__getitem__`/
`__setitem__` stubs in torch, so no further casts are needed) — cleaner than
scattering `# type: ignore` comments per line, and it stays correct if the
classifier's exact submodule types change later.
