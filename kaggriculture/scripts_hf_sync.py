"""Mirror this folder to the private HF repo (token from $HF_TOKEN; never stored in files)."""
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=os.path.dirname(os.path.abspath(__file__)), repo_id="Nabidnur/kaggriculture-agent",
                  repo_type="model", commit_message=os.environ.get("MSG", "sync"),
                  ignore_patterns=["replays/**", "tapes/**", "runs/**", "research/forum_raw/**", "**/__pycache__/**", "*.pyc"])
print("synced")
