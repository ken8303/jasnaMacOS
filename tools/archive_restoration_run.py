#!/usr/bin/env python3
"""Preserve a previous run before a fresh start, under the workflow's existing lock."""
import argparse
import json
import os
from pathlib import Path
import tempfile

MARKER = '.jasna-fresh-start.json'
LOCK = '.jasna-workflow-lock'


def archive_run(work: Path, output: Path, owner_pid: int) -> Path:
    if work.is_symlink() or output.is_dir():
        raise ValueError('refusing to archive a symlinked work directory or directory output')
    work = work.resolve(strict=True)
    output = output.parent.resolve(strict=True) / output.name
    if work.parent != output.parent or work == output:
        raise ValueError('work and output must be distinct siblings')
    if owner_pid <= 1 or (work / LOCK / 'pid').read_text().strip() != str(owner_pid):
        raise ValueError('fresh start requires ownership of the workflow lock')
    os.kill(owner_pid, 0)
    marker = work / MARKER
    if marker.exists():
        plan = json.loads(marker.read_text())
        backup = Path(plan['backup'])
        if backup.parent != work.parent or not backup.name.startswith(output.stem + '.jasna-previous-'):
            raise ValueError('invalid fresh-start backup location')
        if plan['output'] != str(output) or backup.is_symlink() or not backup.is_dir():
            raise ValueError('fresh-start backup does not match this output')
    else:
        backup = Path(tempfile.mkdtemp(prefix=output.stem + '.jasna-previous-', dir=work.parent))
        (backup / 'work').mkdir()
        plan = {
            'backup': str(backup), 'output': str(output),
            'entries': [p.name for p in work.iterdir() if p.name not in (LOCK, MARKER)],
            'has_output': os.path.lexists(output),
        }
        # Publish the complete plan before moving anything. If interrupted, a fresh
        # start resumes this plan; an ordinary resume must not mix old/new caches.
        temporary = backup / 'plan.json'
        temporary.write_text(json.dumps(plan))
        os.replace(temporary, marker)
    for name in plan['entries']:
        if not name or Path(name).name != name or name in ('.', '..', LOCK, MARKER):
            raise ValueError('invalid fresh-start work entry')
        move_once(work / name, backup / 'work' / name)
    if plan['has_output']:
        move_once(output, backup / output.name)
    marker.unlink()
    return backup


def move_once(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        if os.path.lexists(source):
            raise ValueError('both original and backup exist: ' + str(source))
        return
    # rename stays on the same volume, avoiding a second copy of large videos/caches.
    os.rename(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('owner_pid', type=int)
    args = parser.parse_args()
    try:
        backup = archive_run(args.work, args.output, args.owner_pid)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'error: a previous fresh start was interrupted or could not finish: {error}\nChoose Start Fresh again to finish preserving the previous run.\n')
    print(f'Previous output and saved work preserved at: {backup}', flush=True)


if __name__ == '__main__':
    main()
