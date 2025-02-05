SELECT po.ID as ID,po.Name as "Name", s.StressLevel as "Initial_Stress_Level", p.Accuracy as "XGBOOST_ACC",p.PredictedStressLevel as "XGBOOST_Prediction",
(s.StressLevel-p.PredictedStressLevel) as "XGB_Initial",r.PredictedStressLevel as "Random_Prediction",r.Accuracy as "Random_Accuracy",
(s.StressLevel-r.PredictedStressLevel) as "Random_Initial", l.PredictedStressLevel as "Linear_Prediction", l.Accuracy as "Linear_Accuracy",
(s.StressLevel-l.PredictedStressLevel) as "Linear_Initial"
FROM HealthData.dbo.StressMetrics s 
JOIN HealthData.dbo.StressXGBoostPredictions p
ON s.PersonID = p.PersonID
JOIN HealthData.dbo.StressRandomForestPredictions r
ON s.PersonID = r.PersonID
JOIN HealthData.dbo.StressRegressionPredictions l
ON s.PersonID = l.PersonID
JOIN HealthData.dbo.People po
ON s.PersonID = po.ID