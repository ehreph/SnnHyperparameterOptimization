import pandas as pd

hp_set = pd.read_csv('res/configuration/snn_hyperparameter_config.csv').dropna()
hp_set = hp_set.sort_values('dataset', ascending=True)
res1 = pd.read_csv('analytics/model_results/result_accuracy_1.csv')
res2 = pd.read_csv('analytics/model_results/result_accuracy_2.csv')
res3 = pd.read_csv('analytics/model_results/result_accuracy_3.csv')
res4 = pd.read_csv('analytics/model_results/result_accuracy_4.csv')

res1 = res1.append(res2)
res1 = res1.append(res3)
res1 = res1.append(res4)
res1 = res1.reset_index(drop=True)
res1 = res1.astype({"test_accuracy": float, "test_score": float})

res1 = res1.groupby(['dataset'], as_index=False).agg(
    {'test_accuracy': ['max', 'std']})

res1['paper_testAccuracy'] = hp_set['testAccuracy']
res1 = res1.astype({"paper_testAccuracy": float})

res1.to_csv('analytics/model_results/results_combined_final.csv', index_label=False, index=False, float_format='%.3f')
