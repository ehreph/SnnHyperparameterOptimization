import pathlib
import pandas as pd

uci_dir = "PublishedResults/UCI/SNN"

snn_configurations = pd.DataFrame()

for path in pathlib.Path(uci_dir).iterdir():
    if path.is_file():
        file_content = pd.read_csv(path)
        file_content = file_content.sort_values(['testAccuracy'], ascending=False)
        snn_configurations = snn_configurations.append(file_content.head(n=1))

snn_configurations.to_csv("snn_hyperparameter_configuration.csv",index_label=False, index=False)
