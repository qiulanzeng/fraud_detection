from sagemaker.experiments.run import Run
import matplotlib.pyplot as plt
import os

# Flag to control whether to use SageMaker Experiments
USE_SAGEMAKER_EXPERIMENTS = os.environ.get("USE_SAGEMAKER_EXPERIMENTS", "0") == "1"

def log_metrics(run_name, metrics_dict, output_dir="sagemaker-experiments"):
    """
    Log metrics to SageMaker Experiments or print locally.
    """
    os.makedirs(output_dir, exist_ok=True)

    if USE_SAGEMAKER_EXPERIMENTS:
        with Run(experiment_name="fraud-xgboost-experiment", run_name=run_name) as run:
            for metric_name, metric_value in metrics_dict.items():
                run.log_metric(metric_name, value=metric_value)
    else:
        print(f"[LOCAL] Metrics for run '{run_name}': {metrics_dict}")

def log_plot(run_name, fig, plot_name, output_dir="sagemaker-experiments"):
    """
    Save matplotlib figure and log artifact to SageMaker or local folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_name)
    fig.savefig(plot_path)

    if USE_SAGEMAKER_EXPERIMENTS:
        with Run(experiment_name="fraud-xgboost-experiment", run_name=run_name) as run:
            # Correct log_artifact usage
            run.log_artifact(name=os.path.basename(plot_path), value=plot_path)
    else:
        print(f"[LOCAL] Plot saved to {plot_path}")
