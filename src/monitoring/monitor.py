"""
Model monitoring using Evidently for data drift and performance monitoring.
"""
import logging
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    ClassificationPreset
)
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix
)
from prefect import flow, task

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@task(name="load_reference_data")
def load_reference_data(data_path: Path) -> pd.DataFrame:
    """Load reference data (training data)."""
    logger.info("Loading reference data...")
    reference_df = pd.read_csv(data_path / "train.csv")
    return reference_df


@task(name="load_current_data")
def load_current_data(data_path: Path) -> pd.DataFrame:
    """Load current data (production data or test data)."""
    logger.info("Loading current data...")
    current_df = pd.read_csv(data_path / "test.csv")
    return current_df


@task(name="generate_drift_report")
def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path
) -> dict:
    """Generate data drift report."""
    logger.info("Generating data drift report...")
    
    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'Class'
    column_mapping.prediction = None
    
    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset()
    ])
    
    # Run report
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Save report
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(report_file))
    logger.info(f"Drift report saved to {report_file}")
    
    # Extract metrics
    report_dict = report.as_dict()
    
    return report_dict


@task(name="generate_performance_report")
def generate_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path
) -> dict:
    """Generate model performance report."""
    logger.info("Generating performance report...")
    
    # Add predictions if not present (for demo purposes)
    if 'prediction' not in current_data.columns:
        # Load model and generate predictions
        # This is a placeholder - in production, you'd load actual predictions
        current_data = current_data.copy()
        # Simulate predictions (replace with actual model predictions)
        current_data['prediction'] = current_data['Class']
    
    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'Class'
    column_mapping.prediction = 'prediction'
    
    # Create report
    report = Report(metrics=[
        ClassificationPreset(),
        ClassificationQualityMetric(),
        ClassificationClassBalance(),
        ClassificationConfusionMatrix()
    ])
    
    # Run report
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Save report
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(report_file))
    logger.info(f"Performance report saved to {report_file}")
    
    # Extract metrics
    report_dict = report.as_dict()
    
    return report_dict


@task(name="check_alerts")
def check_alerts(drift_metrics: dict, performance_metrics: dict) -> list:
    """Check if any alerts should be triggered."""
    alerts = []
    
    # Check data drift
    try:
        drift_share = drift_metrics.get('metrics', [{}])[0].get('result', {}).get('share_of_drifted_columns', 0)
        if drift_share > 0.3:  # More than 30% of features drifted
            alerts.append({
                'type': 'data_drift',
                'severity': 'high',
                'message': f'High data drift detected: {drift_share*100:.1f}% of features drifted',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.warning(f"Could not extract drift metrics: {e}")
    
    # Check model performance
    try:
        # Extract performance metrics (structure depends on Evidently version)
        # This is a placeholder - adjust based on actual metric structure
        metrics_list = performance_metrics.get('metrics', [])
        for metric in metrics_list:
            if metric.get('metric') == 'ClassificationQualityMetric':
                result = metric.get('result', {})
                accuracy = result.get('current', {}).get('accuracy', 1.0)
                
                if accuracy < 0.90:  # Accuracy below 90%
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'high',
                        'message': f'Model accuracy dropped to {accuracy*100:.1f}%',
                        'timestamp': datetime.now().isoformat()
                    })
    except Exception as e:
        logger.warning(f"Could not extract performance metrics: {e}")
    
    return alerts


@task(name="save_alerts")
def save_alerts(alerts: list, output_path: Path) -> None:
    """Save alerts to file."""
    if alerts:
        output_path.mkdir(parents=True, exist_ok=True)
        alerts_file = output_path / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.warning(f"ALERTS TRIGGERED: {len(alerts)} alerts")
        for alert in alerts:
            logger.warning(f"  - {alert['type']}: {alert['message']}")
        
        logger.info(f"Alerts saved to {alerts_file}")
    else:
        logger.info("No alerts triggered")


@flow(name="monitoring_pipeline")
def monitoring_pipeline(
    data_path: str = "data/processed",
    output_path: str = "reports/monitoring"
):
    """Main monitoring pipeline."""
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Load data
    reference_data = load_reference_data(data_path)
    current_data = load_current_data(data_path)
    
    # Generate reports
    drift_metrics = generate_drift_report(reference_data, current_data, output_path)
    performance_metrics = generate_performance_report(reference_data, current_data, output_path)
    
    # Check for alerts
    alerts = check_alerts(drift_metrics, performance_metrics)
    
    # Save alerts
    save_alerts(alerts, output_path)
    
    logger.info("Monitoring pipeline complete!")
    
    return {
        'drift_metrics': drift_metrics,
        'performance_metrics': performance_metrics,
        'alerts': alerts
    }


def main():
    """Main function to run monitoring."""
    results = monitoring_pipeline()
    
    if results['alerts']:
        logger.warning("⚠️  MONITORING ALERTS DETECTED!")
    else:
        logger.info("✅ All monitoring checks passed")


if __name__ == "__main__":
    main()