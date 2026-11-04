"""Audit the explicit public source tree for broken links and private artifacts."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {'README.md','README.en.md','LICENSE','THIRD_PARTY_NOTICES.md',
             'pyproject.toml','MANIFEST.in','.gitignore','.gitattributes','CITATION.cff','CHANGELOG.md'}
TOP_DIRS = {'src','tests','tools','docs','configs','.github'}
IGNORED_PARTS = {'__pycache__','.pytest_cache','.git','build','dist'}
ALLOWED_SUFFIXES = {'.py','.md','.json','.html','.svg','.csv','.yml','.yaml','.toml','.cff','.in'}
PUBLIC_MEDIA = {
    'docs/assets/motion_deinterlacing.gif': b'GIF89a',
    'docs/assets/pipeline_comparison.png': b'\x89PNG\r\n\x1a\n',
}
SENSITIVE = {
    'private_key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'),
    'private_network_address': re.compile(r'\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b'),
    'personal_home_path': re.compile(r'(?:[A-Za-z]:[\\/]+Users[\\/]+[^\s/\\]+|/home/(?!USER(?:/|\b))[^\s/]+/)'),
}


def public_files(root=ROOT):
    files=[]
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel=p.relative_to(root)
        if any(x in IGNORED_PARTS or x.endswith('.egg-info') for x in rel.parts):
            continue
        if (len(rel.parts)==1 and rel.name in TOP_FILES) or rel.parts[0] in TOP_DIRS:
            files.append(p)
    return sorted(files)


def audit(root=ROOT):
    files=public_files(root)
    errors=[]
    relative={p.relative_to(root).as_posix() for p in files}
    for required in ['README.md','LICENSE','pyproject.toml','src/mlx90640_pipeline/cli.py']:
        if required not in relative:
            errors.append(f'Missing required file: {required}')
    for p in files:
        rel=p.relative_to(root).as_posix()
        if rel in PUBLIC_MEDIA:
            with p.open('rb') as stream:
                signature = stream.read(len(PUBLIC_MEDIA[rel]))
            if signature != PUBLIC_MEDIA[rel] or p.stat().st_size > 5_000_000:
                errors.append(f'Invalid or oversized showcase media: {rel}')
            continue
        if p.name not in TOP_FILES and p.suffix not in ALLOWED_SUFFIXES:
            errors.append(f'Unexpected public file type: {rel}')
            continue
        if p.stat().st_size > 2_000_000:
            errors.append(f'Unexpected large source artifact: {rel}')
        try:
            text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            errors.append(f'Non-text public file: {rel}')
            continue
        # The scanner's own literal detection patterns are expected.
        if rel != 'tools/audit_release.py':
            for label,pattern in SENSITIVE.items():
                if pattern.search(text):
                    errors.append(f'{label}: {rel}')
        if p.suffix == '.py':
            try: ast.parse(text,filename=rel)
            except SyntaxError as exc: errors.append(f'Python syntax: {rel}:{exc.lineno}')
        if p.suffix == '.json':
            try: json.loads(text)
            except json.JSONDecodeError: errors.append(f'Invalid JSON: {rel}')
        links=[]
        if p.suffix == '.md':
            links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        elif p.suffix == '.html':
            links=re.findall(r'(?:href|src)=["\']([^"\']+)["\']',text)
        for link in links:
            if link.startswith(('http:','https:','mailto:','data:','#')):
                continue
            target=(p.parent/link.split('#')[0]).resolve()
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f'Broken or nonportable link: {rel} -> {link}')
    return dict(public_files=len(files), source_bytes=sum(p.stat().st_size for p in files),
                errors=errors, passed=not errors)


if __name__ == '__main__':
    result=audit()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
