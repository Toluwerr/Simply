"""Simply's weight home: GitHub Releases as its own model hub.

Generation 4's brain is ~23 MB per promoted checkpoint. Committing
every promotion into git history would bloat the repository by
hundreds of megabytes a month, so the champion weights now live on a
rolling GitHub Release instead: one release, tagged `champion`, whose
`model.pt` asset is always the current best brain.

How the loop uses it:
  - run start:   download_champion() pulls the latest promoted weights
                 (so a fresh checkout starts warm, no weights in git)
  - promotion:   upload_champion() replaces the asset with the new best

Both calls are best-effort: without a token (local runs, sandbox) they
do nothing, and any API failure is logged, never raised. The loop
keeps working either way - the promote-or-hold gate protects a fresh
or stale brain from ever overwriting the recorded best on metrics.
"""
import json
import os
import urllib.error
import urllib.request

TAG = "champion"
ASSET = "model.pt"
_UA = ("Simply-self-improving-model/4.0 "
       "(educational research; contact: noreply@users.noreply.github.com)")


def _repo():
    return os.environ.get("GITHUB_REPOSITORY", "")


def _token():
    return (os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN") or "")


def available():
    """True when running where GitHub API access makes sense (Actions)."""
    return bool(_repo() and _token())


def _headers(tok, extra=None):
    h = {"User-Agent": _UA, "Accept": "application/vnd.github+json"}
    h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def _call(url, headers, data=None, method="GET", timeout=120):
    req = urllib.request.Request(url, headers=headers, method=method)
    if data is not None:
        req.data = data
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.status


def _find_asset(rel_json, name=ASSET):
    for a in rel_json.get("assets", []):
        if a.get("name") == name:
            return a
    return None


def _get_release(tok):
    repo = _repo()
    url = f"https://api.github.com/repos/{repo}/releases/tags/{TAG}"
    try:
        body, _ = _call(url, _headers(tok))
        return json.loads(body)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _create_release(tok):
    repo = _repo()
    url = f"https://api.github.com/repos/{repo}/releases"
    body = json.dumps({
        "tag_name": TAG,
        "name": "Simply's champion weights",
        "body": ("The current best brain of the self-improvement loop, "
                 "written here automatically on every promotion. This "
                 "file is what each run loads before it starts learning."),
    }).encode()
    body, _ = _call(url, _headers(tok, {"Content-Type": "application/json"}),
                    data=body, method="POST")
    return json.loads(body)


def download_champion(dest, log=print):
    """Fetch the current champion weights from Releases into dest.

    Returns True on success, False when there is nothing to download
    (no release/asset, no token) or the transfer failed."""
    tok = _token()
    if not tok or not _repo():
        return False
    try:
        rel = _get_release(tok)
        if not rel:
            return False
        asset = _find_asset(rel)
        if not asset:
            return False
        url = f"https://api.github.com/repos/{_repo()}/releases/assets/{asset['id']}"
        body, _ = _call(url, _headers(tok, {"Accept": "application/octet-stream"}),
                        timeout=300)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, dest)
        mb = len(body) / 1e6
        log(f"champion weights pulled from Releases: {mb:.1f} MB "
            f"(v{rel.get('name') or TAG})")
        return True
    except Exception as e:
        log(f"champion download failed: {e}")
        return False


def upload_champion(path, version=None, log=print):
    """Replace the champion asset with the file at path (the new best).

    Gets-or-creates the `champion` release, deletes the old asset,
    uploads the new one. Returns True on success."""
    tok = _token()
    if not tok or not _repo():
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
        rel = _get_release(tok)
        if rel is None:
            rel = _create_release(tok)
        old = _find_asset(rel)
        if old:
            url = (f"https://api.github.com/repos/{_repo()}"
                   f"/releases/assets/{old['id']}")
            _call(url, _headers(tok), method="DELETE")
        upload_url = rel["upload_url"].split("{")[0]
        _call(f"{upload_url}?name={ASSET}",
              _headers(tok, {"Content-Type": "application/octet-stream"}),
              data=data, method="POST", timeout=300)
        log(f"champion weights uploaded to Releases: "
            f"{len(data) / 1e6:.1f} MB (v{version or '?'}) - the next "
            "run will load these")
        return True
    except Exception as e:
        log(f"champion upload failed: {e}")
        return False
