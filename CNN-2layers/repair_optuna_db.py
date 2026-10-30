import os
import yaml
import pandas as pd
import optuna
import sqlite3
import shutil
from datetime import datetime

EXPERIMENT_DIR = "scheduled_experiments"
DB_NAME = 'optuna_results.db'
DB_PATH = os.path.join(EXPERIMENT_DIR, DB_NAME)
DB_URI = f"sqlite:///{DB_PATH}"
YAML_FILE = "optunatest.yaml"
CSV_FILE = os.path.join(EXPERIMENT_DIR, 'results.csv')

optuna.logging.set_verbosity(optuna.logging.WARNING)


with open(YAML_FILE) as f:
    config = yaml.safe_load(f)

search_space  = config["search_space"]
param_names   = list(search_space.keys())
main_metric   = config["metrics"]["main"]
other_metrics = config["metrics"].get("others", [])
study_name    = os.path.splitext(os.path.basename(config["script"]))[0]

print(f"Study name : {study_name}")
print(f"Params     : {param_names}")
print(f"Main metric: {main_metric}")


df = pd.read_csv(CSV_FILE)
print(f"\nCSV rows   : {len(df)}")


def canonical_params(row):
    p = {}
    for pname in param_names:
        val = float(row[pname])
        for candidate in search_space[pname]:
            if abs(val - float(candidate)) < max(1e-9, abs(float(candidate)) * 1e-6):
                p[pname] = candidate
                break
        else:
            p[pname] = val
    return p

param_to_row = {}
for _, row in df.iterrows():
    p = canonical_params(row)
    key = tuple(sorted((k, round(float(v), 9)) for k, v in p.items()))
    param_to_row[key] = (p, row)

print(f"Unique combinations in CSV: {len(param_to_row)}")


if os.path.exists(DB_PATH):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{DB_PATH}.backup_{ts}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nDB backup  : {backup}")


study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    storage=DB_URI,
    load_if_exists=True,
)
study.set_user_attr("main_metric", main_metric)

existing_keys = set()
for trial in study.trials:
    key = tuple(sorted((k, round(float(v), 9)) for k, v in trial.params.items()))
    existing_keys.add(key)

print(f"Trials already in Optuna: {len(existing_keys)}")


def has_distribution_conflict():
    if not study.trials:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT trial_id FROM trials WHERE study_id=(SELECT study_id FROM studies WHERE study_name=?) LIMIT 1",
              (study_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    trial_id = row[0]
    c.execute("SELECT param_name, distribution_json FROM trial_params WHERE trial_id=?", (trial_id,))
    stored = {r[0]: r[1] for r in c.fetchall()}
    conn.close()
    import json
    for pname, choices in search_space.items():
        if pname not in stored:
            continue
        try:
            dist = json.loads(stored[pname])
            stored_choices = dist.get("choices", [])
            if set(float(x) for x in stored_choices) != set(float(x) for x in choices):
                return True
        except Exception:
            pass
    return False

def rebuild_study_with_new_distributions(old_study):
    pass
    old_trials = old_study.trials
    print(f"\n⚠️  Rebuilding study to update distributions (migrating {len(old_trials)} existing trials)...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
    result = cursor.fetchone()
    if result:
        sid = result[0]
        cursor.execute("SELECT trial_id FROM trials WHERE study_id=?", (sid,))
        tids = [r[0] for r in cursor.fetchall()]
        for tid in tids:
            for tbl in ("trial_params", "trial_user_attributes", "trial_system_attributes", "trial_values"):
                cursor.execute(f"DELETE FROM {tbl} WHERE trial_id=?", (tid,))
        cursor.execute("DELETE FROM trials WHERE study_id=?", (sid,))
        cursor.execute("DELETE FROM study_user_attributes WHERE study_id=?", (sid,))
        cursor.execute("DELETE FROM study_directions WHERE study_id=?", (sid,))
        cursor.execute("DELETE FROM studies WHERE study_id=?", (sid,))
        conn.commit()
    conn.close()

    import time; time.sleep(0.3)

    new_study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=DB_URI,
        load_if_exists=False,
    )
    new_study.set_user_attr("main_metric", main_metric)

    new_dists = {k: optuna.distributions.CategoricalDistribution(choices=search_space[k]) for k in param_names}
    migrated = 0
    seen_keys = set()
    for t in old_trials:
        key = tuple(sorted((k, round(float(v), 9)) for k, v in t.params.items()))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            new_study.add_trial(optuna.trial.create_trial(
                params=t.params,
                distributions=new_dists,
                value=t.value if t.value is not None else 0.0,
                user_attrs=t.user_attrs or {},
                state=t.state,
            ))
            migrated += 1
        except Exception as e:
            print(f"   ⚠️  Could not migrate trial {t.number}: {e}")

    print(f"   ✅ {migrated}/{len(old_trials)} old trials migrated")
    return new_study


if has_distribution_conflict():
    study = rebuild_study_with_new_distributions(study)

    existing_keys = set()
    for trial in study.trials:
        key = tuple(sorted((k, round(float(v), 9)) for k, v in trial.params.items()))
        existing_keys.add(key)


new_dists = {k: optuna.distributions.CategoricalDistribution(choices=search_space[k]) for k in param_names}
added = 0
skipped = 0
failed = 0

all_metric_cols = [c for c in df.columns if c not in ["experiment"] + param_names + ["execution time"]]

for key, (params_used, row) in sorted(param_to_row.items()):
    if key in existing_keys:
        skipped += 1
        continue


    metrics = {}
    for col in all_metric_cols:
        try:
            metrics[col] = float(row[col])
        except Exception:
            pass

    exec_time = float(row["execution time"]) if "execution time" in df.columns else 0.0

    if main_metric not in metrics:
        print(f"  ⚠️  No '{main_metric}' for {row.get('experiment','?')}, skipping")
        failed += 1
        continue

    try:
        trial = optuna.trial.create_trial(
            params=params_used,
            distributions=new_dists,
            value=metrics[main_metric],
            user_attrs={
                **{k: v for k, v in metrics.items() if k != main_metric},
                "execution_time": exec_time,
            },
        )
        study.add_trial(trial)
        added += 1
        print(f"  ✅  Added  {row.get('experiment','?'):8s}  {params_used}  →  {main_metric}={metrics[main_metric]:.4f}")
    except Exception as e:
        print(f"  ❌  Failed {row.get('experiment','?')}: {e}")
        failed += 1


print("\n" + "="*60)
print("✅  REPAIR COMPLETE")
print(f"   Added   : {added} new trials")
print(f"   Skipped : {skipped} (already in Optuna)")
print(f"   Failed  : {failed}")
print(f"   Total in Optuna now: {len(study.trials)}")
print("="*60)
