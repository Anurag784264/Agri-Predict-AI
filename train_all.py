import sys
sys.stdout.reconfigure(encoding='utf-8')

from preprocessing.data_loader import load_crop_production, load_fertilizer_data, load_soil_data
from models.crop_recommendation import train_and_evaluate_crop_models
from models.fertilizer_model import train_and_evaluate_fertilizer_models
from models.yield_prediction import train_and_evaluate_yield_models

print('=== 1. Training Crop Recommendation Models ===')
df_soil = load_soil_data()
res_crop = train_and_evaluate_crop_models(df_soil)
print('Optimal Model:', res_crop['best_model_name'])
for m, sc in res_crop['results'].items():
    print(f"  {m}: Accuracy = {sc['accuracy']}, F1-Weighted = {sc['f1_weighted']}")

print('\n=== 2. Training Fertilizer Recommendation Models ===')
df_fert = load_fertilizer_data()
res_fert = train_and_evaluate_fertilizer_models(df_fert)
print('Optimal Model:', res_fert['best_model_name'])
for m, sc in res_fert['results'].items():
    print(f"  {m}: Accuracy = {sc['accuracy']}, F1-Weighted = {sc['f1_weighted']}")

print('\n=== 3. Training Crop Yield Regression Models ===')
df_crop = load_crop_production(nrows=30000)
res_yield = train_and_evaluate_yield_models(df_crop)
print('Optimal Model:', res_yield['best_model_name'])
for m, sc in res_yield['results'].items():
    print(f"  {m}: R2 = {sc['r2_score']}, RMSE = {sc['rmse']}, MAE = {sc['mae']}")

print('\nAll model pipelines trained and persisted successfully!')
