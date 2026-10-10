# Downloads every scored submission file of the team (needs ~/.kaggle/access_token).
# mysubs.csv: `kaggle competitions submissions 3rd-wear-dataset-challenge-hasca-2026 --page-size 200 -v > mysubs.csv`
import os, csv, subprocess
from kaggle.api import kaggle_api_extended as K
from kagglesdk.competitions.types.competition_api_service import ApiDownloadSubmissionRequest
api=K.KaggleApi(); api.authenticate()
rows=list(csv.DictReader(l for l in open('mysubs.csv') if not l.startswith('Next')))
ok=fail=0
for r in rows:
    if not r['publicScore']: continue
    dest=f"own_subs/sub_{r['ref']}.csv"
    if os.path.exists(dest) and os.path.getsize(dest)>5000: ok+=1; continue
    try:
        q=ApiDownloadSubmissionRequest(); q.submission_id=int(r['ref'])
        with api.build_kaggle_client() as c: resp=c.competitions.competition_api_client.download_submission(q)
        url=getattr(resp,'url',None) or str(resp)
        subprocess.run(['curl','-s','-L','-m','120','-o',dest,url])
        h=open(dest).readline()
        if 'id' in h: ok+=1
        else: fail+=1; print('BAD',r['ref'],h[:60])
    except Exception as e: fail+=1; print('FAIL',r['ref'],str(e)[:150])
print('ok',ok,'fail',fail)
