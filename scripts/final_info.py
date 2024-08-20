import os

import wandb
import wandb.apis
import wandb.sdk

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["xtick.major.pad"] = "18"
plt.rcParams["ytick.major.pad"] = "12"
plt.rcParams["axes.labelpad"] = "40"

api = wandb.Api(
    overrides=dict(
        base_url="https://api.wandb.ai",
        entity="alexdremov",
        project="adalayers",
    )
)

LANG=''
STRINGS = dict(
    base_solutions="базовые решения",
    proposed_solution="предложенное решение",
    layer_number="Номер слоя",
    step="Шаг",
)

if os.environ.get("LANG") == "en":
    LANG = 'en'
    STRINGS = dict(
        base_solutions="base solutions",
        proposed_solution="proposed solution",
        layer_number="layer number",
        step="step",
    )


def get_artifacts(run):
    artifacts = iter(run.logged_artifacts())
    res = []
    while True:
        try:
            artifact = next(artifacts)
            res.append(artifact)
        except StopIteration:
            break
        except Exception as e:
            print(e)
            break
    return res


def make_info_from_run(name, run_name, model_name, metric_name):
    run: wandb.sdk.wandb_run.Run = api.run(path=run_name)
    model = api.artifact(model_name, type="model")
    metric = model.metadata[metric_name]
    metric = metric if metric > 1 else metric * 100
    return {
        "name": name,
        "run": run,
        "run_name": run_name,
        "metric": metric,
        "metric_name": metric_name,
        "model": model,
        "model_name": model.name,
        "model_artifact": model.qualified_name,
    }


imdb = make_info_from_run(
    name="imdb",
    run_name="aric0wtm",
    model_name="diploma_imdb_adalayers_model_best:v0",
    metric_name="acc",
)
imdb_bart = make_info_from_run(
    name="imdb_bart",
    run_name="kfoh5n75",
    model_name="diploma_bart_imdb_adalayers_model_best:v0",
    metric_name="acc",
)

cola = make_info_from_run(
    name="cola",
    run_name="hknj8ol8",
    model_name="diploma_glue_cola_adalayers_model_best:v3",
    metric_name="acc",
)
cola_bart = make_info_from_run(
    name="cola_bart",
    run_name="xitu3gsf",
    model_name="diploma_bart_glue_cola_adalayers_model_best:v1",
    metric_name="acc",
)

conll = make_info_from_run(
    name="conll",
    run_name="tjluj296",
    model_name="diploma_conll_adalayers_token_model_best:v7",
    metric_name="f1",
)
conll_bart = make_info_from_run(
    name="conll_bart",
    run_name="riprvtow",
    model_name="diploma_bart_conll_adalayers_token_model_best:v2",
    metric_name="f1",
)

all_entities = [
    imdb,
    imdb_bart,
    conll,
    conll_bart,
    cola,
    cola_bart,
]

all_entities_mapping = {i["name"]: i for i in all_entities}

for entity in all_entities:
    print(
        entity["name"],
        f'https://wandb.ai/alexdremov/adalayers/runs/{entity["run_name"]}/overview',
    )
    print(entity)
