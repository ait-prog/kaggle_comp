"""Track the two submitted Kaggle runs without creating new submissions."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {58281588: 'dev3', 58281594: 'dev4'}
api = KaggleApi()
api.authenticate()
deadline = time.monotonic() + 12 * 60 * 60

while time.monotonic() < deadline:
    try:
        terminal = set()
        for submission in api.competition_submissions('name-that-llm'):
            ref = int(submission.ref)
            if ref not in TARGETS:
                continue
            run = TARGETS[ref]
            path = ROOT / 'out' / run / 'submission_receipt.json'
            receipt = json.loads(path.read_text())
            receipt.update(
                status=str(submission.status).split('.')[-1],
                public_score=submission.public_score or None,
                error_description=submission.error_description or None,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(receipt, indent=2) + '\n')
            os.replace(temporary, path)
            print(json.dumps(receipt), flush=True)
            if receipt['status'] not in ('PENDING', 'RUNNING'):
                terminal.add(ref)
                readme = ROOT / 'README.md'
                lines = readme.read_text().splitlines()
                for index, line in enumerate(lines):
                    if line.startswith(f'| {run} v1 |'):
                        cells = line.split('|')
                        new_state = receipt['status']
                        if receipt['public_score']:
                            new_state += f"; public score {receipt['public_score']}"
                        change = 'Sinkhorn off' if run == 'dev3' else 'test-time grouping off'
                        cells[-2] = f' flashpm/ntl-{run}; submission {ref} {new_state}; {change} '
                        lines[index] = '|'.join(cells)
                readme.write_text('\n'.join(lines) + '\n')
        if terminal == set(TARGETS):
            print('Both submissions reached a terminal status.', flush=True)
            break
    except Exception as error:
        print(f'Polling failed: {type(error).__name__}: {error}', flush=True)
    time.sleep(60)
else:
    print('Monitoring window elapsed; consult receipt timestamps before using results.', flush=True)
