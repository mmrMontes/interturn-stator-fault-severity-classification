import sys
from pathlib import Path


STORAGE    = 'sqlite:///C:/Users/montes/OneDrive/DPI/2026-1/paper Leonel/Random_Forest_Info_Gain/Random_Forest_Info_Gain/github/random forest/scheduled_experiments/optuna_results.db'
OUTPUT_DIR = "."
FORMAT    = "html"
SVG_WIDTH  = 1400
SVG_HEIGHT = 700


def sanitize(name):
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def apply_layout(fig, title):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=13, color="#222"),
        title=dict(text=title, font=dict(size=15)),
    )
    return fig


def save_fig(fig, folder, name, title):
    fig = apply_layout(fig, title)

    if FORMAT == "svg":
        out = folder / (name + ".svg")
        try:
            fig.write_image(str(out), format="svg", width=SVG_WIDTH, height=SVG_HEIGHT, engine="kaleido")
        except Exception as e:
            raise RuntimeError(f"SVG error: {e} — install kaleido: pip install kaleido")
    else:
        out = folder / (name + ".html")
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True,
                       config={"displayModeBar": True, "displaylogo": False})

    print(f"      OK  ->  {out.name}")


def export_study(study, output_root):
    import optuna
    import optuna.visualization as ov

    study_name = study.study_name
    study_dir  = output_root / (sanitize(study_name) + "_plots")

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    params   = list(complete[0].params.keys()) if complete else []
    n        = len(complete)

    print(f"  Completed trials : {n}")
    print(f"  Parameters       : {params}")
    print(f"  Output format    : {FORMAT.upper()}")
    print()


    plots = [
        ("history",        "optimization_history", True,           lambda s: ov.plot_optimization_history(s)),
        ("history",        "timeline",             True,           lambda s: ov.plot_timeline(s)),
        ("history",        "intermediate_values",  True,           lambda s: ov.plot_intermediate_values(s)),
        ("analytics",      "parallel_coordinate",  True,           lambda s: ov.plot_parallel_coordinate(s)),
        ("analytics",      "slice",                True,           lambda s: ov.plot_slice(s)),
        ("analytics",      "contour",              len(params)>=2, lambda s: ov.plot_contour(s)),
        ("analytics",      "param_importances",    n >= 4,         lambda s: ov.plot_param_importances(s)),
        ("smart_selection", "edf",                 True,           lambda s: ov.plot_edf(s)),
        ("smart_selection", "rank",                True,           lambda s: ov.plot_rank(s)),
    ]

    for subfolder, name, condition, fn in plots:
        tag = f"[{subfolder}] {name}"

        if not condition:
            print(f"   {tag:<55} SKIPPED (condition not met)")
            continue

        try:
            fig    = fn(study)
            folder = study_dir / subfolder
            folder.mkdir(parents=True, exist_ok=True)
            title  = f"{study_name} - {name.replace('_', ' ').title()}"
            save_fig(fig, folder, name, title)
        except Exception as e:
            print(f"   {tag:<55} SKIPPED ({e})")


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if FORMAT not in ("html", "svg"):
        print(f'ERROR: FORMAT must be "html" or "svg", got "{FORMAT}"')
        sys.exit(1)

    output_root = Path(OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  Optuna Plot Exporter")
    print(f"  Storage : {STORAGE}")
    print(f"  Format  : {FORMAT.upper()}")
    print(f"  Output  : {output_root.resolve()}")
    print(f"{'='*60}\n")

    try:
        summaries = optuna.get_all_study_summaries(storage=STORAGE)
    except Exception as exc:
        print(f"ERROR opening storage: {exc}")
        sys.exit(1)

    if not summaries:
        print("No studies found in storage.")
        return

    print(f"Studies found: {len(summaries)}")
    for s in summaries:
        print(f"   {s.study_name}")
    print()

    for summary in summaries:
        print(f"{'─'*60}")
        print(f"Exporting: {summary.study_name}")
        try:
            study = optuna.load_study(study_name=summary.study_name, storage=STORAGE)
            export_study(study, output_root)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n{'='*60}")
    print(f"  Export complete")
    print(f"  Location : {output_root.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
