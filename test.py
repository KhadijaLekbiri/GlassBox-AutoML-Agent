from agent.autofit import autofit
from agent.report  import report_to_explanation
report = autofit("titanic_dataset.csv", target_col="Fare")
print(report)
print(f"\n  Explanation:\n  {report_to_explanation(report)}")