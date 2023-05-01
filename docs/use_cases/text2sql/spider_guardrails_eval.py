import argparse
import os
from pathlib import Path
import json
from guardrails.embedding import ManifestEmbedding
from guardrails.applications.text2sql import Text2Sql
from manifest import Manifest
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, List, Dict
import sys
from rich.console import Console

console = Console(soft_wrap=True)

sys.path.append("/home/lorr1/projects/code/mlcore/notebooks")
from spider_metrics.spider import evaluation as spider_evaluation  # type: ignore
from spider_metrics.test_suite_sql_eval import (  # type: ignore
    evaluation as test_suite_evaluation,
)

os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/lorr1/.cache/tiktoken"


def compute_exact_match_metric(
    predictions: List, references: List, gold_dbs: List, kmaps: Dict, db_dir: str
) -> tuple[Any, List[int | None]]:
    """Compute exact match metric."""
    evaluator = spider_evaluation.Evaluator(db_dir, kmaps, "match")
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs), total=len(predictions)
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            em_metrics = evaluator.evaluate_one(gold_db, reference, prediction)
            by_row_metrics.append(int(em_metrics["exact"]))
        except Exception as e:
            print(e)
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores["all"]["exact"], by_row_metrics


def compute_test_suite_metric(
    predictions: List, references: List, gold_dbs: List, kmaps: Dict, db_dir: str
) -> tuple[Any, List[int | None]]:
    """Compute test suite execution metric."""
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=kmaps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores: dict[str, list] = {"exec": [], "exact": []}
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db in tqdm(
        zip(predictions, references, gold_dbs), total=len(predictions)
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            ex_metrics = evaluator.evaluate_one(
                gold_db,
                reference,
                prediction,
                turn_scores,
                idx=turn_idx,
            )
            by_row_metrics.append(int(ex_metrics["exec"]))
        except Exception as e:
            print(e)
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores["all"]["exec"], by_row_metrics


def compute_metrics(
    pred_sqls: list[str],
    max_run: int = -1,
    spider_dir: str = "spider_data",
) -> dict[str, str]:
    """Compute all metrics for data slice."""
    gold_path = Path(spider_dir) / "spider" / "dev_gold.sql"
    gold_input_path = Path(spider_dir) / "spider" / "dev.json"
    tables_path = Path(spider_dir) / "spider" / "tables.json"
    database_dir = str(Path(spider_dir) / "spider" / "database")

    kmaps = test_suite_evaluation.build_foreign_key_map_from_json(str(tables_path))
    gold_sqls, gold_dbs = zip(
        *[l.strip().split("\t") for l in gold_path.open("r").readlines()]
    )
    gold_sql_dict = json.load(gold_input_path.open("r"))

    # Subselect
    if max_run > 0:
        print(f"Subselecting {max_run} examples")
        gold_sqls = gold_sqls[:max_run]
        gold_dbs = gold_dbs[:max_run]
        gold_sql_dict = gold_sql_dict[:max_run]
        pred_sqls = pred_sqls[:max_run]
    # Data validation
    assert len(gold_sqls) == len(
        pred_sqls
    ), "Sample size doesn't match between pred and gold file"
    assert len(gold_sqls) == len(
        gold_sql_dict
    ), "Sample size doesn't match between gold file and gold dict"
    all_metrics: dict[str, Any] = {}

    # Execution Accuracy
    metrics, by_row_metrics_exec = compute_test_suite_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir
    )
    all_metrics["exec"] = metrics

    # Exact Match Accuracy
    metrics, by_row_metrics_exact = compute_exact_match_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir
    )
    all_metrics["exact"] = metrics

    # Merge all results into a single dict
    for i in range(len(gold_sql_dict)):
        gold_sql_dict[i]["pred"] = pred_sqls[i]
        gold_sql_dict[i]["exec"] = by_row_metrics_exec[i]
        gold_sql_dict[i]["exact"] = by_row_metrics_exact[i]
        gold_sql_dict[i]["gold"] = gold_sqls[i]
    return all_metrics, gold_sql_dict


def run_eval(pred_sqls, output_dir, max_run, spider_dir):
    """Run evaluation."""
    all_metrics, gold_sql_dict = compute_metrics(pred_sqls, max_run, spider_dir)
    # Write results to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_dir}")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f)
    with open(output_dir / "dump.json", "w") as f:
        json.dump(gold_sql_dict, f)
    return all_metrics


def main():
    args = parse_args()

    print(json.dumps(vars(args), indent=2))

    root = Path(args.spider_dir)
    train_data = json.load(open(root / "spider/train_spider.json"))
    dev_data = json.load(open(root / "spider/dev.json"))
    tables = json.load(open(root / "spider/tables.json"))
    num_run = min(args.num_run, len(dev_data)) if args.num_run > 0 else len(dev_data)
    num_demonstrations = args.num_demonstrations
    # print("**Example**")
    # print(json.dumps(dev_data[0], indent=2))
    print(f"{len(dev_data)} dev examples")

    embedding_manifest = ManifestEmbedding(
        client_name="huggingfaceembedding",
        client_connection="http://127.0.0.1:5000",
        cache_name="sqlite",
        cache_connection="spider_guard_cache_emb.db",
    )

    manifest = Manifest(
        client_name="openai",
        engine="text-davinci-003",
        cache_name="sqlite",
        cache_connection="spider_guard_cache.db",
    )

    if not args.indb:
        examples = [
            {"question": ex["question"], "query": ex["query"]} for ex in train_data
        ]
    else:
        raise NotImplementedError("TODO: Implement in-db training")

    # Iterate over all unique database ids
    database_ids = set([ex["db_id"] for ex in dev_data])
    all_apps = {}
    # Use the same store for all databases
    store = None
    for i, db_id in enumerate(database_ids):
        conn_str = f"sqlite:////home/lorr1/projects/code/mlcore/notebooks/spider_data/spider/database/{db_id}/{db_id}.sqlite"
        if i == 0:
            app = Text2Sql(
                conn_str=conn_str,
                examples=examples,
                embedding=embedding_manifest,
                llm_api=manifest,
                num_relevant_examples=num_demonstrations,
            )
            store = app.store
        else:
            app = Text2Sql(
                conn_str=conn_str,
                examples=None,
                embedding=embedding_manifest,
                llm_api=manifest,
                num_relevant_examples=num_demonstrations,
            )
            assert store is not None
            app.store = store
        all_apps[db_id] = app

    pred_sqls = []
    gold_sqls = []
    for dev_ex in tqdm(dev_data[:num_run], desc="Predicting"):
        database_id = dev_ex["db_id"]
        app = all_apps[database_id]
        question = dev_ex["question"]
        gold_query = dev_ex["query"]
        pred_query = app(question)
        if not pred_query:
            print("BAD QUESTION", question)
            print(console.print(app.guard.state.most_recent_call.tree))
        pred_sqls.append(pred_query or "")
        gold_sqls.append(gold_query)

    print("Running eval")
    output_dir = (
        Path(args.output_dir) / f"{num_run}n_{num_demonstrations}d_{int(args.indb)}indb"
    )
    metrics = run_eval(
        pred_sqls=pred_sqls,
        output_dir=output_dir,
        max_run=num_run,
        spider_dir=str(root),
    )
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, default=200)
    parser.add_argument("--num_demonstrations", type=int, default=0)
    parser.add_argument("--indb", action="store_true")
    parser.add_argument(
        "--spider_dir",
        type=str,
        default="/home/lorr1/projects/code/mlcore/notebooks/spider_data",
    )
    parser.add_argument("--output_dir", type=str, default="spider_eval_output")
    return parser.parse_args()


if __name__ == "__main__":
    main()
