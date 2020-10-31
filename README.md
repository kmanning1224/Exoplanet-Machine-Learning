# Machine Learning Homework - Exoplanet Exploration

![exoplanets.jpg](Images/exoplanets.jpg)

### Reporting

Working with both Linear and Logistic Regression I took the cleaned data and returned these results:

* Set up the Dataframe for ML
```
df = pd.read_csv("exoplanet_data.csv")
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()

# Set features. This will also be used as your x values.
selected_features = df[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period','koi_period_err1','koi_period_err2',
                       'koi_time0bk','koi_time0bk_err1','koi_time0bk_err2','koi_impact','koi_impact_err1','koi_impact_err2','koi_duration','koi_duration_err1',
                       'koi_duration_err2','koi_depth','koi_depth_err1','koi_depth_err2','koi_prad','koi_prad_err1','koi_prad_err2','koi_teq','koi_insol',
                       'koi_insol_err1','koi_insol_err2','koi_model_snr','koi_tce_plnt_num','koi_steff','koi_steff_err1','koi_steff_err2','koi_slogg','koi_slogg_err1',
                       'koi_slogg_err2','koi_srad','koi_srad_err1','koi_srad_err2','ra','dec','koi_kepmag']]

y = df['koi_disposition']
X = selected_features       
                       
```
* Import Sklearn processes
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=30)

# Scale your data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
X_scaler = MinMaxScaler().fit(X_train)
X_train_scale = X_scaler.transform(X_train)
X_test_scale = X_scaler.transform(X_test)

* Linear Modeling
from sklearn.svm import SVC 
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test)

* Logistic Modeling
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2')
model2 = classifier.fit(X_train_scaled, y_train)
```
* Print out the Scores

```
* Linear Test Scores
print(f"Training Data Score: {model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model.score(X_test_scaled, y_test)}")

Training Data Score: 0.8373068853709709
Testing Data Score: 0.8558352402745996

* Logistic Test Scores
print(f"Training Data Score: {model2.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model2.score(X_test_scaled, y_test)}")

Training Data Score: 0.8458897577722678
Testing Data Score: 0.8621281464530892

```
* Begin Tuning

```
* For Linear
from sklearn.model_selection import GridSearchCV
param_grid = {'penalty': ['l1', 'l2'],
    'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid, verbose=3)

print(grid.best_params_)
print(grid.best_score_)

{'C': 50, 'gamma': 0.0001}
0.8781203836441831

* For Logistic
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(model, param_grid, verbose=3)

print(grid.best_params_)
print(grid.best_score_)

{'C': 100}
0.8701116657812966
```
* Finally, save the models using joblib to designated SAV files.
```
import joblib
filename = 'MODEL1.sav'
joblib.dump(model, filename)

import joblib
filename = 'MODEL2.sav'
joblib.dump(model2, filename)
```
- - -


## Conclusion

As we can see against each test/train score, the Logistic method produced somewhat more accurate results in comparison to Linear. 

- - -
## Resources

* [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

* [Scikit-Learn Tutorial Part 1](https://www.youtube.com/watch?v=4PXAztQtoTg)

* [Scikit-Learn Tutorial Part 2](https://www.youtube.com/watch?v=gK43gtGh49o&t=5858s)

* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)


