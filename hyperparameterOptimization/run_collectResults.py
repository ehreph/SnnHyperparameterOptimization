import pathlib
import pandas as pd
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
results_path = ROOT_DIR + "/target/results/"
analytics_path = ROOT_DIR + "/analytics/model_results/"
target_file = "result_accuracy.csv"

collected_results = pd.DataFrame()

for path in pathlib.Path(results_path).iterdir():
    if path.is_file():
        file_content = pd.read_csv(path)
        collected_results = collected_results.append(file_content.head(n=1))

analytics_dir = pathlib.Path(analytics_path)
print(collected_results)

os.chdir(analytics_path)
collected_results.to_csv(target_file, index_label=False, index=False)
