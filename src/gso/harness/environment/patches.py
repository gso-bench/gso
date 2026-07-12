def apply_patch_requests(test: str) -> str:
    patch_code = """
import requests
import hashlib
import os
import re
import io
import json
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

CACHE_DIR = './.url_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

original_get = requests.get

def url_to_filename(url):
    hash_digest = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, hash_digest)

# Wikimedia thumbnail-size compatibility shim.
# Wikimedia has tightened accepted thumbnail sizes, so URLs like
#   /thumb/{path}/{file}/{N}px-{file}
# now return 400 for arbitrary N. We rewrite to the original (full-size)
# image and resize with PIL to the requested {N}px width. See gso-bench/gso#31, #32.
_WIKI_THUMB_RE = re.compile(
    r"^(https://upload\\.wikimedia\\.org/wikipedia/commons/thumb/([^/]+/[^/]+)/([^/]+))/([0-9]+)px-([^/]+)$"
)

def _wikimedia_original_url(match):
    # https://upload.wikimedia.org/wikipedia/commons/<hash_dirs>/<file>
    _thumb_full, hash_dirs, file_name, _px, _repeated = match.groups()
    return "https://upload.wikimedia.org/wikipedia/commons/" + hash_dirs + "/" + file_name

def _resize_to_thumbnail_bytes(image_bytes, target_px, source_url):
    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes))
    # Wikimedia thumbnail semantics: N px is the WIDTH of the thumbnail,
    # with height scaled to preserve aspect ratio.
    w, h = im.size
    if w == 0:
        return image_bytes
    new_w = int(target_px)
    new_h = max(1, round(h * (new_w / w)))
    im = im.resize((new_w, new_h))
    buf = io.BytesIO()
    fmt = 'PNG' if source_url.lower().endswith('.png') else 'JPEG'
    save_kwargs = {'quality': 90} if fmt == 'JPEG' else {}
    im.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()

def _fetch_wikimedia_fallback(url, *args, **kwargs):
    match = _WIKI_THUMB_RE.match(url)
    if not match:
        return None
    orig_url = _wikimedia_original_url(match)
    target_px = int(match.group(4))
    if os.getenv("DEBUG_GSO") == "true":
        print(f"Wikimedia thumb 400; rewriting to original + resize({target_px}px): {url}")
    orig_resp = original_get(orig_url, *args, **kwargs)
    if orig_resp.status_code != 200:
        return None
    try:
        content = _resize_to_thumbnail_bytes(orig_resp.content, target_px, url)
    except Exception:
        return None
    return content

def patched_get(url, *args, **kwargs):
    cache_path = url_to_filename(url)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached_content = f.read()
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = cached_content
        response.headers['X-From-Cache'] = 'true'
        return response

    if os.getenv("DEBUG_GSO") == "true":
        print(f"WARN: cache miss for url: {url} in file: {__file__}")

    if 'verify' not in kwargs:
        kwargs["verify"] = False

    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = "CoolBot/1.0 (https://example.org/coolbot/; coolbot@example.org)"

    response = original_get(url, *args, **kwargs)

    # Wikimedia thumbnail 400 fallback: fetch original + resize to requested px
    if response.status_code == 400 and 'upload.wikimedia.org/wikipedia/commons/thumb/' in url:
        fallback_bytes = _fetch_wikimedia_fallback(url, *args, **kwargs)
        if fallback_bytes is not None:
            with open(cache_path, 'wb') as f:
                f.write(fallback_bytes)
            fake = requests.Response()
            fake.status_code = 200
            fake.url = url
            fake._content = fallback_bytes
            fake.headers['X-Wiki-Thumb-Rewrite'] = 'true'
            return fake

    if response.status_code == 200:
        with open(cache_path, 'wb') as f:
            f.write(response.content)
    return response

# replace w/ patched version
requests.get = patched_get

# Patch requests.Session to cache HuggingFace Hub API calls
# This intercepts all session requests including those from huggingface_hub
_original_session_request = requests.Session.request

def _cached_session_request(self, method, url, **kwargs):
    # Only cache GET requests to HuggingFace API
    if method.upper() == 'GET' and 'huggingface.co' in str(url):
        cache_path = url_to_filename(str(url))
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_content = f.read()
            response = requests.Response()
            response.status_code = 200
            response.url = str(url)
            response._content = cached_content
            response.headers['X-From-Cache'] = 'true'
            response.headers['Content-Type'] = 'application/json'
            if os.getenv("DEBUG_GSO") == "true":
                print(f"HF cache HIT: {url}")
            return response
        
        if os.getenv("DEBUG_GSO") == "true":
            print(f"HF cache MISS: {url}")
        
        response = _original_session_request(self, method, url, **kwargs)
        if response.status_code == 200:
            with open(cache_path, 'wb') as f:
                f.write(response.content)
        return response
    
    return _original_session_request(self, method, url, **kwargs)

requests.Session.request = _cached_session_request

# Patch httpx for HuggingFace Hub (which uses httpx, not requests)
try:
    import httpx
    
    _original_httpx_request = httpx.Client.request
    
    def _cached_httpx_request(self, method, url, **kwargs):
        url_str = str(url)
        if method.upper() == 'GET' and 'huggingface.co' in url_str:
            cache_path = url_to_filename(url_str)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_content = f.read()
                if os.getenv("DEBUG_GSO") == "true":
                    print(f"HTTPX cache HIT: {url_str[:80]}")
                # Build a proper request object for the response
                request = httpx.Request(method, url)
                response = httpx.Response(
                    status_code=200,
                    content=cached_content,
                    headers={'content-type': 'application/json', 'x-from-cache': 'true'},
                    request=request
                )
                return response
            
            if os.getenv("DEBUG_GSO") == "true":
                print(f"HTTPX cache MISS: {url_str[:80]}")
            
            response = _original_httpx_request(self, method, url, **kwargs)
            if response.status_code == 200:
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
            return response
        
        return _original_httpx_request(self, method, url, **kwargs)
    
    httpx.Client.request = _cached_httpx_request
    
except ImportError:
    pass  # httpx not installed
"""
    return patch_code + "\n\n" + test


PATCH_REGISTRY = {
    "requests": {
        "description": "Enable caching, Disable SSL verification, and Add User-agent in requests",
        "apply": apply_patch_requests,
        "instances": [],  # apply to all instances
    },
}


def apply_patches(instance_id: str, tests: list[str]) -> list[str]:
    patched_tests = tests.copy()
    for patch_name, patch_info in PATCH_REGISTRY.items():
        patch_instances = patch_info.get("instances", [])
        if patch_instances == [] or instance_id in patch_info.get("instances", []):
            patch_func = patch_info.get("apply")
            if patch_func:
                patched_tests = [patch_func(test) for test in patched_tests]

    return patched_tests


def apply_patches_to_tests(patch_name: str, tests: list[str]) -> list[str]:
    """Apply a patch to a given list of tests"""
    patch_fn = PATCH_REGISTRY.get(patch_name, {}).get("apply")

    if not patch_fn:
        raise ValueError(f"Patch '{patch_name}' not found in registry.")
    patched_tests = []
    for test in tests:
        patched_tests.append(patch_fn(test))
    return patched_tests
