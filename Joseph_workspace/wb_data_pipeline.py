import wbgapi as wb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.impute import KNNImputer

class WbDataPipeline():

    def __init__(self, indicators, year, impute=True):
        self.indicators = indicators
        self.year = year
        self.happiness_data = pd.read_csv('../data/happiness/happiness.csv').drop('Country', axis=1)
        self.valid_countries = self.happiness_data['ISO_A3'].unique()

        wb.db = 2
        self.data = self.pull_data()
        if impute:
            self.data = self.impute_numeric_data()

    def pull_wb_data(self):
        data = wb.data.DataFrame(self.indicators, time=self.year)
        return data
    
    def pull_data(self):
        # print(f"Pulling {len(self.indicators)} indicators from World Bank data...")
        features = self.pull_wb_data()
        # print("Done")
        features = features.reset_index()
        features = features.infer_objects()
        features = features.rename(columns={features.columns[0]: 'ISO_A3'})
        features = features[features['ISO_A3'].isin(self.valid_countries)]
        merged = pd.merge(features, self.happiness_data, on='ISO_A3')
        return merged
    
    def impute_numeric_data(self) -> pd.DataFrame:
        '''This code fills in missing numerical data with the mean of its 5 nearest neighbors
        as determined by its nonmissing numerical data. No categorical features are 
        touched.
        
        '''
        numeric_data = self.data.select_dtypes(include=['float64', 'int'])
        imputed_data = KNNImputer().fit_transform(numeric_data.T).T
        self.data[numeric_data.columns] = imputed_data
        return self.data
    
    def get_data(self): 
        if self.data is None:
            self.data = self.pull_data()
        return self.data

    def check_missing(self, threshold):
        nans = self.data.isna().sum()
        threshold = 1 - threshold
        nthreshold = np.round(self.data.shape[0] * threshold)
        cols = nans[nans > nthreshold]
        print(f"The following features are less than {100*(1-threshold)}% complete:")
        for col in cols.index:
            pcomplete = 1 - (nans[col] / self.data.shape[0])
            print(f"   {col}: {pcomplete*100}% complete") 
        return cols
    
    def get_percent_complete(self):
        nans = self.data.isna().sum()
        return 1 - (nans / self.data.shape[0])
    
def clean_features():
    entries = []
    indicators = wb.series.list()
    ids = [(str(indicator['id']), str(indicator['value'])) for indicator in indicators]
    pbar = tqdm(total=len(ids), position=0, leave=True)
    for id, value in ids:
        dp = WbDataPipeline([id], 2022, impute=False)
        complete_percent = dp.get_percent_complete()[id]
        entries.append([id, value, complete_percent])
        pbar.update()
    pbar.close()

    entries = pd.DataFrame(entries, columns=['id', 'value', 'complete_percent'])
    return entries

def generate_wb_dataset(complete_percent=1.0, write_csv=None):
    feature_data = pd.read_csv('feature_data.csv')
    feature_data = feature_data.infer_objects()
    features = feature_data[feature_data['complete_percent'] >= complete_percent]['id']
    dp = WbDataPipeline(features, 2022)

    if write_csv is not None:
        dp.get_data().to_csv(write_csv)

    return dp.get_data()

if __name__ == "__main__":
    # entries = clean_features()
    # entries.to_csv('feature_data.csv')

    generate_wb_dataset(write_csv='no_missing.csv')