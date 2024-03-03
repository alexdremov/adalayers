import wandb

wandb.login()

dry_run = False
api: wandb.Api = wandb.Api()
project: wandb.apis.public.projects.Project = api.project('adalayers')

for artifact_type in project.artifacts_types():
    if artifact_type.name != "model":
        continue
    for artifact_collection in artifact_type.collections():
        if len(artifact_collection.artifacts()) == 0 and not dry_run:
            artifact_collection.delete()
            continue
        for version in artifact_collection.artifacts():
            version: wandb.Artifact
            if len(version.aliases) == 0 and not dry_run:
                version.delete()