"""Download official ISIC 2017 assets and COCO-pretrained YOLO11n.

Uses only Python's standard library. Raw archives and weights are kept outside
the Git checkout. Downloads resume with HTTP Range; extraction checks ZIP CRCs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import zipfile

BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2017/'
FILES = [
    'ISIC-2017_Training_Data.zip',
    'ISIC-2017_Validation_Data.zip',
    'ISIC-2017_Test_v2_Data.zip',
    'ISIC-2017_Training_Part1_GroundTruth.zip',
    'ISIC-2017_Validation_Part1_GroundTruth.zip',
    'ISIC-2017_Test_v2_Part1_GroundTruth.zip',
    'ISIC-2017_Training_Part3_GroundTruth.csv',
    'ISIC-2017_Validation_Part3_GroundTruth.csv',
    'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
]
WEIGHTS_URL = 'https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt'
WEIGHTS_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
PRINT_LOCK = threading.Lock()

def log(message):
    with PRINT_LOCK:
        print(message, flush=True)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def segmented_download(url, destination, expected_size, connections=16):
    """Resume completed HTTP ranges in a preallocated partial file."""
    partial = destination.with_name(destination.name + '.part')
    journal = destination.with_name(destination.name + '.ranges.json')
    block_size = 16 * 1024 * 1024
    if journal.exists():
        state = json.loads(journal.read_text())
        if state['url'] != url or state['size'] != expected_size:
            raise ValueError(f'Changed source for existing partial file: {journal}')
        if not partial.exists():
            raise ValueError(f'Range journal exists without partial file: {partial}')
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > expected_size:
            raise ValueError(f'Partial file exceeds source size: {partial}')
        state = {'url':url,'size':expected_size,'prefix':prefix,'block_size':block_size,'done':[]}
        journal.write_text(json.dumps(state))
    block_size = state['block_size']
    ranges = [(start,min(start+block_size,expected_size)-1)
              for start in range(state['prefix'],expected_size,block_size)]
    done = set(state['done'])
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd,expected_size)
    def fetch(item):
        start,end=item
        for attempt in range(5):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{expected_size}':
                        raise RuntimeError('Server returned an unexpected HTTP range')
                    position=start
                    for block in iter(lambda:response.read(1024*1024),b''):
                        if position+len(block)>end+1: raise RuntimeError('Range response too large')
                        view=memoryview(block)
                        while view:
                            written=os.pwrite(fd,view,position)
                            if written<=0: raise OSError('Range write did not advance')
                            position+=written
                            view=view[written:]
                    if position!=end+1: raise RuntimeError('Range response truncated')
                return start
            except Exception:
                if attempt==4: raise
                time.sleep(1)
    log(f'Parallel download {destination.name}: {connections} connections; {len(done)}/{len(ranges)} ranges complete')
    last_log=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures=[pool.submit(fetch,item) for item in ranges if item[0] not in done]
            for future in as_completed(futures):
                done.add(future.result())
                state['done']=sorted(done)
                temp=journal.with_suffix('.json.tmp')
                temp.write_text(json.dumps(state))
                os.replace(temp,journal)
                if time.monotonic()-last_log>=20 or len(done)==len(ranges):
                    complete=state['prefix']+sum(end-start+1 for start,end in ranges if start in done)
                    log(f'{destination.name}: {complete/expected_size:.1%} ({complete/1e6:.0f} MB verified ranges)')
                    last_log=time.monotonic()
        os.fsync(fd)
    finally:
        os.close(fd)
    partial.rename(destination)

def download(url, destination, expected_sha=None, connections=16):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        expected_size = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    partial = destination.with_name(destination.name + '.part')
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise ValueError(f'Existing file has unexpected size; inspect before replacing: {destination}')
        log(f'Already downloaded: {destination.name}')
    elif expected_size > 100*1024*1024:
        segmented_download(url,destination,expected_size,connections=connections)
    else:
        for attempt in range(1, 6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == expected_size:
                break
            if offset > expected_size:
                raise ValueError(f'Partial file exceeds expected size: {partial}')
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and response.status != 206:
                        raise RuntimeError('Server did not support requested resume; partial file retained')
                    log(f'Downloading {destination.name}: {offset/1e6:.1f}/{expected_size/1e6:.1f} MB')
                    last_log = time.monotonic()
                    with partial.open('ab' if offset else 'wb') as output:
                        for block in iter(lambda: response.read(2 * 1024 * 1024), b''):
                            output.write(block)
                            offset += len(block)
                            if time.monotonic() - last_log >= 20:
                                log(f'{destination.name}: {offset/expected_size:.1%} ({offset/1e6:.0f} MB)')
                                last_log = time.monotonic()
                if partial.stat().st_size != expected_size:
                    raise RuntimeError('Incomplete response; will resume')
                break
            except Exception as error:
                log(f'Attempt {attempt} for {destination.name}: {error}')
                if attempt == 5:
                    raise
                time.sleep(2)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f'Incorrect downloaded size: {partial}')
        partial.rename(destination)
    digest = sha256(destination)
    if expected_sha and digest != expected_sha:
        raise ValueError(f'Official SHA256 mismatch: {destination}')
    log(f'Verified {destination.name}: {expected_size:,} bytes, SHA256 {digest}')
    return {'url': url, 'path': str(destination), 'bytes': expected_size,
            'etag': etag, 'sha256': digest,
            'publisher_sha256_verified': bool(expected_sha)}

def extract(record, root):
    archive = Path(record['path'])
    marker = root / ('.' + archive.stem + '.extracted.json')
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('archive_sha256') == record['sha256'] and all(
            (root/name).is_file() and (root/name).stat().st_size == size
            for name, size in saved.get('files', {}).items()
        ):
            log(f'Extraction already complete: {archive.name}')
            return
    log(f'Extracting and CRC-checking {archive.name}')
    extracted = {}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = root/member.filename
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink in archive: {member.filename}')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(target.name + '.extracting')
            with source.open(member) as incoming, staged.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=2*1024*1024)
            if staged.stat().st_size != member.file_size:
                raise ValueError(f'Incorrect extracted size: {target}')
            os.replace(staged, target)
            extracted[member.filename] = member.file_size
    marker.write_text(json.dumps({'archive_sha256': record['sha256'], 'files': extracted}, indent=2))
    log(f'Extracted {archive.name}: {len(extracted)} files')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--connections', type=int, default=16,
                        help='Parallel HTTP ranges per large archive (1-64)')
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.connections <= 64:
        parser.error('--workers must be positive and --connections must be 1-64')
    root = args.workspace_root.resolve()
    raw = root/'data/ISIC2017'
    archives = raw/'archives'
    metadata = raw/'metadata'
    raw.mkdir(parents=True, exist_ok=True)
    jobs = [(BASE+name, (archives if name.endswith('.zip') else metadata)/name, None) for name in FILES]
    jobs.insert(0, (WEIGHTS_URL, root/'weights/yolo11n.pt', WEIGHTS_SHA256))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download,*job,connections=args.connections) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'complete': False,
                        'source_page': 'https://challenge.isic-archive.com/data/#2017', 'files': records}
            (raw/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    for record in records:
        if record['path'].endswith('.zip'):
            extract(record,raw/'raw')
    manifest['complete'] = True
    (raw/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    log('All official assets downloaded and ZIP extraction verified. No training was run.')

if __name__=='__main__':
    main()
