SELECT * 
FROM HealthData.dbo.StressRegressionPredictions p
JOIN HealthData.dbo.StressMetrics s
ON p.PersonID = s.PersonID
WHERE p.PredictedStressLevel <> 1 AND p.PredictedStressLevel <> 0 AND p.PredictedStressLevel <> 2
