import pandas as pd
from pycaret.regression import *
from functions_code import *
import sweetviz as sv
from skopt.space import Real, Integer
train = pd.read_csv('train.csv').drop(columns =["Id", "PoolQC"]) # read in the data
test = pd.read_csv("test.csv").drop(columns = ["Id", "PoolQC"])
# comparison_report = sv.compare([train, "Train"], [test, "Test"], 'SalePrice')
# comparison_report.show_html("comparison_report.html")
updated_train = set_missing_to_na(train, ["MasVnrType", "Alley"]) # Example of manually setting the missing 
updated_test = set_missing_to_na(test, ["MasVnrType", "Alley"])
clf = setup(updated_train, target='SalePrice', session_id=666) # Setting up the model
top3 = compare_models(n_select = 3)
tuned_top3 = [tune_model(i) for i in top3] # tuning best 3 models
blender = blend_models(tuned_top3) #Create a blender model (Voting Regressor)
stacker = stack_models(tuned_top3) #Create a stacked model (Meta Model)
best_model = automl(optimize = 'RMSLE') #Get's best model trained so far in current session
preds = predict_model(best_model, data = updated_test).rename(columns={'prediction_label': 'SalePrice'})
preds['Id'] = range(1461, 1461 + len(preds))
to_submit = preds[["Id", "SalePrice"]].reset_index().drop(columns="index")
to_submit.to_csv("house_prices_predictions.csv") # Save data for submission

#LightGBM example
lightgbm = create_model('lightgbm')

params = {
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [20, 30, 40, 50],
    'max_depth': [-1, 5, 10, 15],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'n_estimators': [50, 100, 200, 300]
}



# Create custom hyper-parameter grid for Bayesian optimization
params = {
    'learning_rate': Real(0.01, 0.3),
    'num_leaves': Integer(10, 50),
    'max_depth': Integer(6, 25),
    'min_child_samples': Integer(10, 30),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'n_estimators': Integer(200, 1000),
    "alpha": Real(0.1,1)
}


tuned = tune_model(lightgbm, search_algorithm= "bayesian", search_library="scikit-optimize", n_iter = 100, optimize="RMSLE", custom_grid = params)
preds = predict_model(tuned, data = updated_test).rename(columns={'prediction_label': 'SalePrice'})
preds['Id'] = range(1461, 1461 + len(preds))
to_submit = preds[["Id", "SalePrice"]].reset_index().drop(columns="index")
to_submit.to_csv("house_prices_predictions_2.csv") # Save data for submission
