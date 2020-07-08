import git
from LambdaZero.datasets.brutal_dock import ROOT_DIR


def create_logging_tags(execution_file_name: str):
    repo = git.Repo(ROOT_DIR)
    git_hash = repo.head.object.hexsha

    tags = {"mlflow.source.git.commit": git_hash,
            "mlflow.source.name": execution_file_name}

    return tags
