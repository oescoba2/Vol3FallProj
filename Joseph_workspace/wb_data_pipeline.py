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
        print(f"Pulling {len(self.indicators)} indicators from World Bank data...")
        features = self.pull_wb_data()
        print("Done")
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
    
    def get_missing_percentages(self):
        nans = self.data.isna().sum()
        return nans / self.data.shape[0]
    
def clean_features():
    entries = []
    indicators = wb.series.list()
    ids = [(str(indicator['id']), str(indicator['value'])) for indicator in indicators]
    pbar = tqdm(total=len(ids), position=0, leave=True)
    for id, value in ids:
        dp = WbDataPipeline([id], 2022)
        missing_percent = dp.get_missing_percentages()[id]
        entries.append([id, value, missing_percent])
        pbar.update()
    pbar.close()

    entries = pd.DataFrame(entries, columns=['id', 'value', 'missing_percent'])
    return entries