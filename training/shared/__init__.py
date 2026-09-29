"""Code shared across training attempts (`birds/`, `coco/`, and whatever comes next).

Named `shared`, not `common`: `database/` already owns the top-level `common`
package, and a second one on `sys.path` would silently shadow the DB layer.
"""
