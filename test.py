from agent.autofit import autofit
report = autofit("titanic_dataset.csv", target_col="Fare")
print(report)