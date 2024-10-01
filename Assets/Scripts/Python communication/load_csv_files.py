import pandas as pd

import os

main_path = "~/Unity_projects/FetalUltrasoundSimulator_test1/AcquiredData/Poses1/"
file_path = "pos_unity.csv"
df = pd.read_csv(os.path.join(main_path, file_path))

print("df ", df.shape)
# Display the first few rows of the DataFrame
print(df.head())