def apply_ssl_patch(test: str) -> str:
    patch_code = """
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
original_get = requests.get
def patched_get(*args, **kwargs):
    if 'verify' not in kwargs:
        kwargs["verify"] = False
    return original_get(*args, **kwargs)

# patch to disable SSL verification
requests.get = patched_get
"""
    return patch_code + "\n\n" + test


PATCH_REGISTRY = {
    "ssl": {
        "description": "Disable SSL verification in requests",
        "apply": apply_ssl_patch,
        "instances": [
            "numpy__numpy-09db9c7",
            "numpy__numpy-e801e7a",
            "numpy__numpy-ee75c87",
            "numpy__numpy-ef5e545",
            "numpy__numpy-728fedc",
            "numpy__numpy-83c780d",
            "numpy__numpy-cb461ba",
            "numpy__numpy-567b57d",
            "numpy__numpy-19bfa3f",
            "numpy__numpy-b862e4f",
            "numpy__numpy-1b861a2",
            "numpy__numpy-68eead8",
            "pydantic__pydantic-addf1f9",
            "python-pillow__Pillow-4bc33d3",
            "tornadoweb__tornado-1b464c4",
            "tornadoweb__tornado-9a18f6c",
        ],
    },
}


def apply_patches(instance_id: str, tests: list[str]) -> list[str]:
    patched_tests = tests.copy()
    for patch_name, patch_info in PATCH_REGISTRY.items():
        if instance_id in patch_info.get("instances", []):
            patch_func = patch_info.get("apply")
            if patch_func:
                patched_tests = [patch_func(test) for test in patched_tests]

    return patched_tests
