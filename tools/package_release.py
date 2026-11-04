"""Build a source-only ZIP from the audited allowlist, never from the whole workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from audit_release import ROOT, audit, public_files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit()
    if not result['passed']:
        raise ValueError(json.dumps(result))
    output=args.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError('Release output must be outside the repository')
    output.mkdir(parents=True,exist_ok=False)
    files=public_files()
    listing=[]
    path=output/'mlx90640-thermal-pipeline-v0.1.0-source.zip'
    prefix='mlx90640-thermal-pipeline/'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            data=file.read_bytes()
            relative=file.relative_to(ROOT).as_posix()
            listing.append(f'{hashlib.sha256(data).hexdigest()}  {relative}')
            z.writestr(prefix+relative,data)
        manifest='\n'.join(listing)+'\n'
        z.writestr(prefix+'SHA256SUMS.txt',manifest.encode())
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise AssertionError('ZIP CRC check failed')
        for item in listing:
            expected,relative=item.split('  ',1)
            if hashlib.sha256(z.read(prefix+relative)).hexdigest()!=expected:
                raise AssertionError(relative)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'SHA256SUMS.txt').write_text(f'{digest}  {path.name}\n',encoding='utf-8')
    report=dict(source_files=len(files),source_bytes=result['source_bytes'],archive_bytes=path.stat().st_size,
                archive_sha256=digest,zip_crc_verified=True,all_member_hashes_verified=True,
                includes_models=False,includes_raw_data=False,includes_device_calibration=False)
    (output/'release_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
