"""Validate a submission locally, then submit it to Kaggle and wait for the score.

    python submit.py subs/v1.csv "message"

The Kaggle token is injected by the session's network proxy; the placeholder
below is never a real credential.
"""
import os
import sys
import time

import requests
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiListSubmissionsRequest,
    ApiStartSubmissionUploadRequest,
)

from verify import score_file

COMP = "santa-2026"
path, message = sys.argv[1], sys.argv[2]

local = score_file(path, "data/puzzles.csv")  # raises SystemExit if invalid
print(f"local score: {local}")

with KaggleClient(env=KaggleEnv.PROD, api_token="proxy-injected") as k:
    api = k.competitions.competition_api_client
    req = ApiStartSubmissionUploadRequest()
    req.competition_name = COMP
    req.file_name = os.path.basename(path)
    req.content_length = os.path.getsize(path)
    req.last_modified_epoch_seconds = int(os.path.getmtime(path))
    up = api.start_submission_upload(req)
    with open(path, "rb") as f:
        r = requests.put(up.create_url, data=f)
    if r.status_code not in (200, 201):
        raise SystemExit(f"upload failed: {r.status_code} {r.text[:300]}")
    sub = ApiCreateSubmissionRequest()
    sub.competition_name = COMP
    sub.blob_file_tokens = up.token
    sub.submission_description = message
    resp = api.create_submission(sub)
    print("submitted:", resp.message)

    for _ in range(60):
        time.sleep(10)
        lr = ApiListSubmissionsRequest()
        lr.competition_name = COMP
        latest = api.list_submissions(lr).submissions[0]
        status = str(latest.status)
        print("status:", status, "public score:", latest.public_score)
        if "PENDING" not in status.upper():
            break
