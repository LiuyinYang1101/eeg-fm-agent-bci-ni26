"""Codabench worker-image import guard (lead review b388c0c item 3): refuse every import the worker image cannot satisfy.

Our dev envs (local neural_interface_challenge, sofia ni26b) carry packages the worker image (`tommoral/neural-compet:v2`
= python:3.12-slim + torch/torchvision/torchaudio + `pip install -r requirements.txt` of the kit) does not, e.g. timm.
A submission importing one passes every in-env test and fails on Codabench. This file is put first on PYTHONPATH so
Python imports it at start-up in EVERY process (benchopt subprocesses included); it is inert unless NI26_WORKER_GUARD
names a distribution allowlist (one normalised name per line, `dists_<kit sha>.txt`, written by `make_dists.py`).

Two rules, by WHO imports (the first non-importlib frame):
  * code under NI26_WORKER_GUARD_ROOTS (':'-separated: the submission dir, the kit) -> STRICT: allowed iff stdlib /
    built-in, or under a root, or provided by a distribution on the image allowlist;
  * library code (site-packages) -> allowed iff stdlib, or provided by a distribution in the ENV's dependency closure of
    the allowlist (installed metadata, extras ignored). The dev env's versions differ from the image's (sofia's pandas
    2.x needs pytz, the image's pandas 3.x does not), so a library's own hard dependencies must load; its OPTIONAL imports
    outside that closure (`try: import timm`, torch's `dill`) are refused and degrade as they do on the worker.
Availability probes (`importlib.util.find_spec`, e.g. transformers' optional-package checks) get None for a refused
package, as on the worker; real imports of it raise ModuleNotFoundError (what the worker raises) and are logged to stderr ('!! worker-guard') and, if set,
appended to NI26_WORKER_GUARD_LOG. When an env has no metadata mapping for an import (sofia's EasyBuild SciPy-bundle
ships scipy without one), the import name itself or a known alias stands in for the distribution name.

Known gap: a module a library already loaded is in sys.modules, so a later submission `import` of it never reaches the
guard (harmless for timm-like packages, which no allowed library can load).

The environment's own sitecustomize (EasyBuild's Python on sofia uses one to put the module torch on sys.path) is
found and executed after the guard is installed, so the guard never changes what the env can import.
"""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_env_sitecustomize():
    path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


# import name -> distribution name, used only when an env has no metadata mapping for the import (see find_spec)
_ALIASES = {"yaml": "PyYAML", "sklearn": "scikit-learn", "PIL": "pillow", "dateutil": "python-dateutil",
            "google": "protobuf", "attr": "attrs", "jwt": "PyJWT", "nacl": "PyNaCl", "mpl_toolkits": "matplotlib",
            "pkg_resources": "setuptools", "_distutils_hack": "setuptools", "github": "PyGithub", "mako": "Mako",
            "jinja2": "Jinja2", "markupsafe": "MarkupSafe", "pygments": "Pygments", "lazy_loader": "lazy-loader"}


class _WorkerGuard:
    def __init__(self, allow_file, roots):
        with open(allow_file) as f:
            self.allow = {_norm(ln.split("#")[0].strip()) for ln in f if ln.split("#")[0].strip()}
        self.roots = [os.path.abspath(r) + os.sep for r in roots if r]
        self.std = set(sys.builtin_module_names) | set(getattr(sys, "stdlib_module_names", ()))
        import sysconfig   # the interpreter's own lib dir (e.g. _sysconfigdata_*), never its site-packages
        self.stdlib = sorted({os.path.abspath(sysconfig.get_paths()[k]) + os.sep for k in ("stdlib", "platstdlib")})
        self.ok = {}                       # (name, strict) -> True
        self._p2d = None
        self._env = None

    def _dists(self, top):
        if self._p2d is None:
            import importlib.metadata as md
            self._p2d = md.packages_distributions()
        return self._p2d.get(top, [])

    def _env_closure(self):
        """the allowlist's dependency closure under THIS env's installed metadata (requirements behind an extra skipped)."""
        if self._env is None:
            import importlib.metadata as md
            import re
            req = {}
            for d in md.distributions():
                n = _norm(d.metadata.get("Name") or "")
                if not n:
                    continue
                for r in d.requires or []:
                    if ";" in r and "extra" in r.split(";", 1)[1]:
                        continue
                    m = re.match(r"\s*([A-Za-z0-9_.\-]+)", r)
                    if m:
                        req.setdefault(n, set()).add(_norm(m.group(1)))
            seen, stack = set(), list(self.allow)
            while stack:
                n = stack.pop()
                if n not in seen:
                    seen.add(n)
                    stack.extend(req.get(n, ()))
            self._env = seen
        return self._env

    def _in_root(self, path):
        return any(path.startswith(r) for r in self.roots)

    def _strict_caller(self):
        f = sys._getframe(2)
        while f is not None and f.f_code.co_filename.startswith("<"):
            f = f.f_back
        return f is not None and self._in_root(os.path.abspath(f.f_code.co_filename))

    def refusal(self, fullname, strict, path=None):
        """None if the worker image could import top-level module `fullname` from this caller, else the refusal message."""
        if (fullname, strict) in self.ok:
            return None
        if fullname in self.std or fullname.startswith("_frozen"):
            self.ok[(fullname, strict)] = True
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:   # not importable here either: let the normal machinery raise
            return None
        origin = spec.origin or (list(spec.submodule_search_locations or [""])[0])
        o = os.path.abspath(origin) if origin not in ("built-in", "frozen") else origin
        in_std = any(o.startswith(r) for r in self.stdlib) and not any(s in o for s in ("site-packages", "dist-packages"))
        if origin in ("built-in", "frozen") or in_std or self._in_root(o):
            self.ok[(fullname, strict)] = True
            return None
        dists = self._dists(fullname)
        names = [_norm(d) for d in dists] or [_norm(fullname), _norm(_ALIASES.get(fullname, fullname))]
        pool = self.allow if strict else self._env_closure()
        if any(n in pool for n in names):
            self.ok[(fullname, strict)] = True
            return None
        return (f"!! worker-guard: refused 'import {fullname}' ({'submission/kit' if strict else 'library'} import; {origin}; "
                f"distributions {dists or 'none'}) — not in the Codabench worker image "
                f"({os.path.basename(os.environ['NI26_WORKER_GUARD'])})")

    def _log(self, fullname, msg, kind):
        print(msg if kind != "probe" else msg.replace("refused 'import", "probe: absent '"), file=sys.stderr, flush=True)
        with open(os.environ.get("NI26_WORKER_GUARD_LOG", os.devnull), "a") as f:
            f.write(f"{fullname}\t{kind}\n")

    def find_spec(self, fullname, path=None, target=None):
        """a real import (meta-path finder): refused -> ModuleNotFoundError, what the worker raises."""
        if "." in fullname:
            return None
        strict = self._strict_caller()
        msg = self.refusal(fullname, strict, path)
        if msg is None:
            return None
        self._log(fullname, msg, "strict" if strict else "library")
        raise ModuleNotFoundError(msg, name=fullname)


def _install(guard):
    """meta-path finder for imports + a wrapped importlib.util.find_spec for availability PROBES (transformers'
    _is_package_available, ...): on the worker a probe of an absent package returns None instead of raising."""
    sys.meta_path.insert(0, guard)
    orig = importlib.util.find_spec

    def find_spec(name, package=None):
        full = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        top = full.split(".")[0]
        if top not in sys.modules:
            f = sys._getframe(1)
            strict = guard._in_root(os.path.abspath(f.f_code.co_filename))
            msg = guard.refusal(top, strict)
            if msg is not None:
                guard._log(top, msg, "probe")
                return None
        return orig(name, package)

    importlib.util.find_spec = find_spec


if os.environ.get("NI26_WORKER_GUARD"):
    _install(_WorkerGuard(os.environ["NI26_WORKER_GUARD"], os.environ.get("NI26_WORKER_GUARD_ROOTS", "").split(":")))
_chain_env_sitecustomize()
