def apply_patch_requests(test: str) -> str:
    patch_code = """
import requests
import hashlib
import os
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

CACHE_DIR = './.url_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

original_get = requests.get

def url_to_filename(url):
    hash_digest = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, hash_digest)

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
    if response.status_code == 200:
        with open(cache_path, 'wb') as f:
            f.write(response.content)
    return response

# replace w/ patched version
requests.get = patched_get
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
