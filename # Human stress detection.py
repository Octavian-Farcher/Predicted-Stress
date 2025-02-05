# Human stress detection
#imports
import pandas as pd
import numpy as np
import zipfile as zipfile
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Extracting data in current working directory
with zipfile.ZipFile('C:\\Users\\FaOc463\\Desktop\\zip.zip', 'r') as zObject:
    zObject.extractall()
file = pd.read_csv("./Stress-Lysis.csv")    
names = pd.read_excel("./Names.xlsx")

# Transforming temperature field from fahrenheit into celsius

file["Temperature"] = ((file["Temperature"]-32)*5)/9

### Data checking

file.describe() 
#  No big deviations  or something that may be odd

na_counts = file.isna().apply(file.value_counts).T
# No NA values in our dataset

file["Temperature"].apply(type).value_counts()
file["Humidity"].apply(type).value_counts()
file["Step count"].apply(type).value_counts()
file["Stress Level"].apply(type).value_counts()
# The data types are ok no modifications to be done

# Creating plots
# plt.figure(figsize=(12, 8))
for i, column in enumerate(file.columns):
    plt.subplot(2, 2, i+1)
    sns.histplot(file[column], kde=True)
    plt.title(f'Distribuția {column}')
plt.tight_layout()
plt.show() 

# Pairplots

### Visualising outliers with graphs 
plot_df = file

plt.figure(figsize=(12,8))
sns.boxplot(plot_df)
plt.title("Box plot for humidity, temperature, step count, and stress level")
plt.xlabel("Indicators")
plt.ylabel("Values")
plt.show()


### We have outliers for seminficative outliers for the steps count field

### Calculating the correllation coeficient (Pearson)
def pearson_correlation(dataset,x,y):
    X = dataset[x]
    Y = dataset[y]
    correlation_matrix = np.corrcoef(X, Y) 
    correlation_coefficient = correlation_matrix[0, 1] #0.8326232433614356
    return correlation_coefficient

pearson_correlation(file,"Humidity","Stress Level") #0.9360362899114489
pearson_correlation(file,"Temperature","Stress Level") #0.9360362899114495
pearson_correlation(file,"Step count","Stress Level") #0.8326232433614356
pearson_correlation(file,"Step count","Humidity") #0.8704860484528991
pearson_correlation(file,"Temperature","Humidity") #1.0
pearson_correlation(file,"Step count","Temperature") #0.8704860484528993

# Also we can see the correlations between the indicators in the the below plot
correlation_matrix=file.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",vmin=-1,vmax=1)
plt.title("Corelation Matrix")
plt.show()

### Eliminating outliers for each indicator
# Calculating Q1 and Q3 to get the Inter quartile range outliers

def inner_quantile(file, variable):
    Q1 = file[variable].quantile(0.25)
    Q3 = file[variable].quantile(0.75)
    IQR = Q3-Q1
    outliers = file[(file[variable]<(Q1-1.5*IQR))| (file[variable]>(Q3+1.5*IQR))]
    print(f"IQR = {IQR}, Lower={Q1 -1.5 * IQR}, Higher={Q3 +1.5 * IQR}")
    return outliers
inner_quantile(file,"Temperature")
inner_quantile(file,"Step count")
inner_quantile(file,"Humidity")
inner_quantile(file,"Stress Level")

# No outliers found with this method

# Z score method 

def z_score_outliers(file, variable):
    mean = file[variable].mean()
    std = file[variable].std()
    file['z_score'] = (file[variable] - mean) / std
    outliers = file[(file['z_score'] > 3) | (file['z_score'] < -3)]
    print(f"Mean = {mean}, Std = {std}")
    return outliers

print(z_score_outliers(file, "Temperature"))
print(z_score_outliers(file, "Step count"))
print(z_score_outliers(file, "Humidity"))
print(z_score_outliers(file, "Stress Level"))

# Creating id for each row
file['id'] = range(1,len(file)+1)
file['id'].value_counts()
file[file['id'].duplicated() == True] 
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# Connectig to the local server database

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=(localdb)\\local;' 
    'DATABASE=StressData;'
    'Trusted_Connection=yes;'
)

stressMetricsTable_Create_Query = '''
CREATE TABLE StressMetrics (
ID int,
Humidity float,
Temperature float,
StepCount int,
StressLevel int
)'''
conn.execute(stressMetricsTable_Create_Query)
conn.commit()
query = "SELECT * FROM dbo.StressMetrics"
df = pd.read_sql(query, conn)
cursor = conn.cursor()
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics ON')
for index,row in file.iterrows():
    cursor.execute(''' 
    Insert INTO dbo.StressMetrics (ID,Humidity,Temperature,StepCount,StressLevel)
                VALUES(?, ?, ?, ?, ?) 
    ''',row['id'],row["Humidity"],row["Temperature"],row["Step count"],row["Stress Level"])
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics OFF')
conn.commit()
cursor.close()
conn.close()
# Creating second table to add in the database
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=(localdb)\\local;' 
    'DATABASE=StressData;'
    'Trusted_Connection=yes;'
)
query = '''Create Table People (
ID int Primary Key Identity(1,1)
,Name varchar(50),
Surname varchar(50))'''
cursor = conn.cursor()
cursor.execute(query)
cursor.execute('SET IDENTITY_INSERT dbo.People OFF')
conn.commit()
cursor.close()
# Inserting values inside our newly created table

cursor = conn.cursor()
cursor.execute('SET IDENTITY_INSERT dbo.People ON')
for index,row in names.iterrows():
    cursor.execute('''
INSERT INTO dbo.People(ID,Name,Surname)
                   VALUES(?,?,?)
''',
row["ID"],row["Name"],row["Surname"]
    )
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics OFF')
conn.commit()
cursor.close()
# creating relationship between the two created tables

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=(localdb)\\local;' 
    'DATABASE=StressData;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics ON')
cursor.execute(
    '''
ALTER TABLE dbo.StressMetrics
ADD PersonID int
'''
)
cursor.execute(
    '''
UPDATE dbo.StressMetrics
SET PersonID = ID
'''
)
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics OFF')
conn.commit()
cursor.close()


# Creating foreign key relationship between the tables

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=(localdb)\\local;' 
    'DATABASE=StressData;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics ON')
cursor.execute(
    '''
ALTER TABLE dbo.StressMetrics
ADD CONSTRAINT FK_StressMetrics_People
FOREIGN KEY (PersonID) REFERENCES People(ID);
'''
)
cursor.execute('SET IDENTITY_INSERT dbo.StressMetrics OFF')
conn.commit()
cursor.close()

# Getting back data from our database

join_Query = '''
SELECT * FROM dbo.People p
JOIN dbo.StressMetrics s
ON p.ID = s.PersonID ;
'''

joined_df = pd.read_sql(join_Query,conn)
conn.close()

joined_df.describe()
# Creating a prediction on the stress levels based on the independent variables
x= joined_df[["Humidity","Temperature","StepCount"]]
y= joined_df[["StressLevel"]]

# Getting the test and train sets ready
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Training the model for linear regression

model=LinearRegression()
model.fit(x_train,y_train)

score = model.score(x_test, y_test)
print(f'R^2 score: {score}')



# Calculating and observating the residuals
y_pred = model.predict(x_test)
residuals = y_test - y_pred

# Visualization of residuals
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Stress Level')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Stress Level')
plt.show()

# No concerns when coming to the residuals

# Now were getting the StressMetrics table from our database, drop the StressLevel field
# and use our model to predict on the initial data

stressMetrics_Query = '''
SELECT s.PersonID,s.Humidity,s.Temperature,s.StepCount
FROM dbo.StressMetrics s
'''
stressPredictions=pd.read_sql(stressMetrics_Query,conn)

linearPrediction = model.predict(stressPredictions[["Humidity","Temperature","StepCount"]])
print(linearPrediction)
linearPredictionDF = stressPredictions
linearPrediction[linearPrediction < 0.01] = 0 
linearPredictionDF ["StressLevel"] = linearPrediction

# Eliminating negative values for stressLevel field
# Adding the prediction table into the db
linearPredictionDF.describe()

createRegressionTable_Query = '''
CREATE TABLE StressRegressionPredictions (
ID int Primary Key Identity(1,1),
PersonID int,
Humidity float,
Temperature float,
StepCount int,
PredictedStressLevel float 
)
'''
createForeignKey_Query = '''
ALTER TABLE dbo.StressRegressionPredictions
ADD CONSTRAINT FK_StressPredictions_People
FOREIGN KEY (PersonID) REFERENCES People(ID)
'''
cursor = conn.cursor()
cursor.execute(createRegressionTable_Query)
cursor.execute(createForeignKey_Query)
cursor.commit()
cursor.execute('SET IDENTITY_INSERT dbo.StressRegressionPredictions ON')
for index,row in stressPredictions.iterrows():
    cursor.execute('''
    INSERT INTO dbo.StressRegressionPredictions(ID,PersonID,Humidity,Temperature,StepCount,PredictedStressLevel)
    VALUES (?,?,?,?,?,?)
    ''',row["PersonID"],row["PersonID"],row["Humidity"],row["Temperature"],row["StepCount"],row["StressLevel"])   
cursor = conn.cursor()

cursor.execute('SET IDENTITY_INSERT dbo.StressRegressionPredictions OFF')
conn.commit()
cursor.close()



# Creating accuracy column in predictions table SQL
addAccuracy_Query='''
ALTER TABLE dbo.StressRegressionPredictions
ADD Accuracy INT;
'''
updateAccuracyValue_Query = '''
DECLARE @epsilon FLOAT = 0.0001;

UPDATE p
SET p.Accuracy = CASE 
                            
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.05 THEN 99
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.1 THEN 90
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.2 THEN 80
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.3 THEN 70
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.4 THEN 60
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.5 THEN 50
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.6 THEN 40
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.7 THEN 30
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.8 THEN 20
                    WHEN s.StressLevel = 0 AND p.PredictedStressLevel <= 0.9 THEN 10
                    ELSE ABS(CAST((p.PredictedStressLevel * 100.0 / (s.StressLevel + @epsilon)) AS FLOAT))
                 END
FROM dbo.StressRegressionPredictions p
JOIN dbo.StressMetrics s
ON p.PersonID = s.PersonID;
'''

conn.execute(addAccuracy_Query)
conn.execute(updateAccuracyValue_Query)
conn.commit()




predictionTableFinal_Query ='''
SELECT * 
FROM StressRegressionPredictions
'''
stressRegressionPredictions = pd.read_sql(predictionTableFinal_Query,conn)

plt.figure(figsize=(10,8))
plt.hist(stressRegressionPredictions["Accuracy"])
plt.title("How much of the initial value is the prediction? ")
plt.xlabel("Accuracy %")
plt.ylabel("Total Values")

### Creating a new prediction with xgboost model


model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(x_train, y_train)

# prediction and evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Prediction
xgboostPrediction = model.predict(stressPredictions[["Humidity","Temperature","StepCount"]])
xgboostPredictionDF = stressPredictions
xgboostPrediction_2d_Array = xgboostPrediction.reshape(-1,1).astype(np.float64)
xgboostPrediction_2d_Array[xgboostPrediction_2d_Array < 0.01] = 0 # so that the values that are extremely low to be represented with 0 
#  and to not have any calculation problems
xgboostPredictionDF["StressLevelPredicted"] = xgboostPrediction_2d_Array
xgboostPredictionDF.describe()
### For xgboost there are no negative values for predicted stress levels, which is much more accurate than the linear regression 

# Inserting the xgboost table in the database
cursor = conn.cursor()
createXGboostTable_Query ='''
CREATE TABLE StressXGBoostPredictions (
ID int Primary Key Identity(1,1),
PersonID int,
Humidity float,
Temperature float,
StepCount int,
PredictedStressLevel float 
)
'''
conn.execute(createXGboostTable_Query)
conn.commit()

createForeignKey_Query = '''
ALTER TABLE dbo.StressXGBoostPredictions
ADD CONSTRAINT FK_StressXGBoostPredictions_People
FOREIGN KEY (PersonID) REFERENCES People(ID)
'''
conn.execute(createForeignKey_Query)
cursor = conn.cursor()
conn.commit()
cursor.execute('SET IDENTITY_INSERT dbo.StressXGBoostPredictions ON')
conn.execute(createForeignKey_Query)
for index,row in xgboostPredictionDF.iterrows():
    cursor.execute( '''
    INSERT INTO  StressXGBoostPredictions(ID,PersonID,Humidity,Temperature,StepCount,PredictedStressLevel)
        VALUES (?,?,?,?,?,?)
    ''', row["PersonID"],row["PersonID"],row["Humidity"],row["Temperature"],row["StepCount"],row["StressLevelPredicted"]
    )
conn.commit()
cursor.execute('SET IDENTITY_INSERT dbo.StressXGBoostPredictions OFF')
addAccuracy_Query='''
ALTER TABLE dbo.StressXGBoostPredictions
ADD Accuracy INT;
'''

updateAccuracyValue_Query = '''
DECLARE @epsilon FLOAT = 0.0001;

UPDATE p
SET p.Accuracy = CASE 
                    WHEN p.PredictedStressLevel = 0 AND s.StressLevel = 0 THEN 100
                    ELSE CAST((p.PredictedStressLevel * 100.0 / (s.StressLevel + @epsilon)) AS INT)
                 END
FROM dbo.StressXGBoostPredictions p
JOIN dbo.StressMetrics s
ON p.PersonID = s.PersonID;
'''
dropAccuracy_Query = '''
ALTER TABLE dbo.StressXGBoostPredictions
DROP COLUMN Accuracy,StressLevelPredicted
'''
# drop columns just to modify values that are <0,01 to not get blanks 
# on a accuracy column


conn.execute(addAccuracy_Query)
conn.execute(updateAccuracyValue_Query)
conn.commit()

# Creating a random forest regressor prediction model

model = RandomForestRegressor(n_estimators=100000, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
randomForestPrediction = model.predict(stressPredictions[["Humidity","Temperature","StepCount"]])
randomForestPrediction_2d_Array = randomForestPrediction.reshape(-1,1).astype(np.float64)
randomForestPrediction_2d_Array[randomForestPrediction < 0.01] = 0 
randomForestPredictionDF = stressPredictions
randomForestPredictionDF["PredictedStressLevel"] =  randomForestPrediction_2d_Array

# Creating table for random forest

cursor = conn.cursor()
createRandomForestTable_Query ='''
CREATE TABLE StressRandomForestPredictions (
ID int Primary Key Identity(1,1),
PersonID int,
Humidity float,
Temperature float,
StepCount int,
PredictedStressLevel float 
)
'''
createForeignKey_Query = '''
ALTER TABLE dbo.StressRandomForestPredictions
ADD CONSTRAINT FK_StressRandomForestPredictions_People
FOREIGN KEY (PersonID) REFERENCES People(ID)
'''
addAccuracy_Query='''
ALTER TABLE dbo.StressRandomForestPredictions
ADD Accuracy INT;
'''
updateAccuracyValue_Query = '''
DECLARE @epsilon FLOAT = 0.0001;

UPDATE p
SET p.Accuracy = CASE 
                    WHEN p.PredictedStressLevel = 0 AND s.StressLevel = 0 THEN 100
                    WHEN p.PredictedStressLevel = s.StressLevel THEN 100
                    WHEN p.PredictedStressLevel - s.StressLevel  BETWEEN 0.01 AND -0.01 THEN 90
                    WHEN p.PredictedStressLevel - s.StressLevel  BETWEEN 0.1 AND -0.1 THEN 80
                    ELSE CAST((p.PredictedStressLevel * 100.0 / (s.StressLevel + @epsilon)) AS FLOAT)
                 END
FROM dbo.StressRandomForestPredictions p
JOIN dbo.StressMetrics s
ON p.PersonID = s.PersonID;
'''
conn.execute("SET IDENTITY_INSERT dbo.StressRandomForestPredictions ON")
# renameTable_Query = """
# EXEC sp_rename 'RandomForestPredictions', 'StressRandomForestPredictions';
# """
for index,row in randomForestPredictionDF.iterrows():
    cursor.execute( '''
    INSERT INTO  StressRandomForestPredictions(ID,PersonID,Humidity,Temperature,StepCount,PredictedStressLevel)
        VALUES (?,?,?,?,?,?)
    ''', row["PersonID"],row["PersonID"],row["Humidity"],row["Temperature"],row["StepCount"],row["PredictedStressLevel"]
    )
conn.execute("SET IDENTITY_INSERT dbo.StressRandomForestPredictions OFF")
conn.execute(createRandomForestTable_Query)
conn.execute(createForeignKey_Query)
conn.execute(addAccuracy_Query)
conn.execute(updateAccuracyValue_Query)
conn.commit()





# Creating a Ridge regressor prediction model
ridge_reg = Ridge(alpha=1.0) 
ridge_reg.fit(x_train,y_train)
y_pred = ridge_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

ridgeReggPrediction = ridge_reg.predict(stressPredictions[["Humidity","Temperature","StepCount"]])
ridgeReggPrediction_2d = ridgeReggPrediction.reshape(-1,1).astype(np.float64)
ridgeReggPrediction_2d[ridgeReggPrediction_2d < 0] = 0 
ridgeReggPredictionDF = stressPredictions
ridgeReggPredictionDF["PredictedStressLevel"] = ridgeReggPrediction_2d

conn = pyodbc.connect(
'DRIVER={ODBC Driver 17 for SQL Server};'
'SERVER=(localdb)\\local;'
'DATABASE=StressData;'
'Trusted_Connection=yes;'
)
# Creating table
conn.execute('''
CREATE TABLE StressRidgePredictions (
ID int,
PersonID int,
Humidity float,
Temperature float,
StepCount int,
PredictedStressLevel float,
Accuracy int
)
''')
# Creating foreign key

conn.execute('''
ALTER TABLE dbo.StressRidgePredictions
ADD CONSTRAINT FK_StressRidgePredictions_People
FOREIGN KEY (PersonID) REFERENCES People(ID)
''')
conn.commit()


cursor = conn.cursor()

conn.execute("SET IDENTITY_INSERT dbo.StressRidgePredictions ON")
for index,row in ridgeReggPredictionDF.iterrows():
    cursor.execute('''
    INSERT INTO StressRidgePredictions(ID, PersonID,Humidity,Temperature,StepCount,PredictedStressLevel)
    VALUES (?,?,?,?,?,?)
    ''', row["PersonID"],row["PersonID"],row["Humidity"],row["Temperature"],row["StepCount"],row["PredictedStressLevel"])
    

cursor.commit()
conn.execute(
'''DECLARE @epsilon FLOAT = 0.0001;

UPDATE p
SET p.Accuracy = CASE 
                    WHEN p.PredictedStressLevel = 0 AND s.StressLevel = 0 THEN 100
                    WHEN p.PredictedStressLevel = s.StressLevel THEN 100
                    WHEN p.PredictedStressLevel - s.StressLevel  < 0.01 AND p.PredictedStressLevel - s.StressLevel  > -0.01 THEN 99
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.1 AND p.PredictedStressLevel - s.StressLevel  > -0.1 THEN 95
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.2 AND p.PredictedStressLevel - s.StressLevel  > -0.2 THEN 90
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.3 AND p.PredictedStressLevel - s.StressLevel  > -0.3 THEN 85
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.4 AND p.PredictedStressLevel - s.StressLevel  > -0.4 THEN 80
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.5 AND p.PredictedStressLevel - s.StressLevel  > -0.5 THEN 75
                    ELSE CAST((p.PredictedStressLevel * 100.0 / (s.StressLevel + @epsilon)) AS FLOAT)
                 END
FROM dbo.StressRidgePredictions p
JOIN dbo.StressMetrics s
ON p.PersonID = s.PersonID;
'''
) 

conn.commit()

# Lasso Regression Model

lassoPredictionDF = stressPredictions.drop(columns=["PredictedStressLevel"])
lasso_reg = Lasso(alpha=1.0)  
lasso_reg.fit(x_train, y_train)
lassoPredictions = lasso_reg.predict(lassoPredictionDF[["Humidity","Temperature","StepCount"]])
lassoPredictions = lassoPredictions.reshape(-1,1).astype(np.float64)
lassoPredictions[lassoPredictions < 0] = 0
lassoPredictionDF["PredictedStressLevel"] = lassoPredictions

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=(localdb)\\local;' 
    'DATABASE=StressData;'
    'Trusted_Connection=yes;'
)

# Creating table for dataframe with new predictions
conn.execute('''
CREATE TABLE StressLassoPredictions (
             ID int,
             PersonID int,
             Humidity float,
             Temperature float,
             StepCount int,
             PredictedStressLevel float,
             Accuracy int
             )
''')

# Creating foreign key for our new table with the people table

conn.execute('''
ALTER TABLE StressLassoPredictions
ADD CONSTRAINT FK_StressLassoPredictions_People
FOREIGN KEY (PersonID) REFERENCES People(ID)
'''             
             )
conn.commit()
# Creating cursor to input data from our dataframe

cursor = conn.cursor()
conn.execute("SET IDENTITY_INSERT dbo.StressLassoPredictions ON")
for index,value in lassoPredictionDF.iterrows():
    cursor.execute('''
INSERT INTO StressLassoPredictions(ID,PersonID,Humidity,Temperature,StepCount,PredictedStressLevel)
                   VALUES(?,?,?,?,?,?)                  
''',value["PersonID"],value["PersonID"],value["Humidity"],value["Temperature"],value["StepCount"],value["PredictedStressLevel"])
    
    cursor.commit()
    conn.commit()
conn.execute("SET IDENTITY_INSERT dbo.StressLassoPredictions OFF")

conn.execute(
'''DECLARE @epsilon FLOAT = 0.0001;

UPDATE p
SET p.Accuracy = CASE 
                    WHEN p.PredictedStressLevel = 0 AND s.StressLevel = 0 THEN 100
                    WHEN p.PredictedStressLevel = s.StressLevel THEN 100
                    WHEN p.PredictedStressLevel - s.StressLevel  < 0.01 AND p.PredictedStressLevel - s.StressLevel  > -0.01 THEN 99
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.1 AND p.PredictedStressLevel - s.StressLevel  > -0.1 THEN 95
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.2 AND p.PredictedStressLevel - s.StressLevel  > -0.2 THEN 90
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.3 AND p.PredictedStressLevel - s.StressLevel  > -0.3 THEN 85
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.4 AND p.PredictedStressLevel - s.StressLevel  > -0.4 THEN 80
                     WHEN p.PredictedStressLevel - s.StressLevel  < 0.5 AND p.PredictedStressLevel - s.StressLevel  > -0.5 THEN 75
                    ELSE CAST((p.PredictedStressLevel * 100.0 / (s.StressLevel + @epsilon)) AS FLOAT)
                 END
FROM dbo.StressLassoPredictions p
JOIN dbo.StressMetrics s
ON p.PersonID = s.PersonID;
'''
) 

conn.commit()
# Creating views for the powerBI connection
conn.execute(
'''
CREATE VIEW People_view as 
SELECT * 
FROM StressData.dbo.People
'''

)

conn.execute(
'''
CREATE VIEW XGBoost_view as 
SELECT * 
FROM StressData.dbo.StressXGBoostPredictions
'''

)

conn.execute(
'''
CREATE VIEW StressRidge_view as 
SELECT * 
FROM StressData.dbo.StressRidgePredictions
'''

)

conn.execute(
'''
CREATE VIEW StressRegressionPredictions_view as 
SELECT * 
FROM StressData.dbo.StressRegressionPredictions
'''

)

conn.execute(
'''
CREATE VIEW StressLasso_view as
SELECT * 
FROM StressData.dbo.StressLassoPredictions
'''

)

conn.execute(
'''
CREATE VIEW RandomForest_view as 
SELECT * 
FROM StressData.dbo.StressRandomForestPredictions
'''

)

conn.execute(
'''
CREATE VIEW StressMetrics_view as 
SELECT * 
FROM StressData.dbo.StressMetrics
'''

)

conn.commit()

# Creating view for overview visual in PowerBI

conn.execute('''
CREATE VIEW CombinedTables_view as
    SELECT sd.PersonID as ID, sd.StressLevel as Initial , pl.PredictedStressLevel as LassoPrediction,pr.PredictedStressLevel as RandomForestPrediction,prp.PredictedStressLevel as LinearPrediction,prd.PredictedStressLevel as RidgePrediction,px.PredictedStressLevel as XGBoostPrediction
             FROM  StressData.dbo.StressMetrics sd      
             JOIN StressLassoPredictions pl
             ON sd.PersonID = pl.PersonID
             JOIN StressRandomForestPredictions pr
             ON sd.PersonID = pr.PersonID
             JOIN StressRegressionPredictions prp
             ON sd.PersonID = prp.PersonID
             JOIN StressRidgePredictions prd
             ON sd.PersonID = prd.PersonID
             JOIN StressXGBoostPredictions px
             ON sd.PersonID = px.PersonID
''')
#  Adding foreign key
conn.commit()
