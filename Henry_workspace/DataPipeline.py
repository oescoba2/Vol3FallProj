import pycountry
import pandas as pd

class DataPipeline():
    '''
    A class for reading and writing data from a csv file.

    ...

    Attributes
    ----------
    data_path : str
        path of to the data file
    data : DataFrame
        dataframe containing the data
    
    Methods
    -------
    clean(pipeline)
        clean the data using the given pipeline
    '''

    def __init__(self, df=None, data_path=None):

        if data_path is not None:
            self.data_path = data_path
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = df

        self.transforms = {'name_change' : self.name_change}

    def clean(self, pipeline):
        
        df = self.df
        for transform in pipeline:
            df = self.transforms[transform]()
        return df
    
    def name_change(self):
        
        df = self.df
        name_changes = {'Bolivia (Plurinational State of)' : 'Bolivia',
                'Democratic Republic of the Congo' : 'Congo, The Democratic Republic of the',
                'Iran (Islamic Republic of)':'Iran',
                'Micronesia (Federated States of)' : 'Micronesia, Federated States of',
                'Republic of Korea' : 'Korea, Republic of',
                'Swaziland' : 'Eswatini',
                'The former Yugoslav republic of Macedonia' : 'North Macedonia',
                'Turkey' : 'Türkiye',
                'Venezuela (Bolivarian Republic of)' : 'Venezuela, Bolivarian Republic of',
                'Taiwan Province of China' : 'Taiwan',
                'Kosovo' : 'Serbia',
                'North Cyprus' : 'Cyprus',
                'Russia' : 'Russian Federation',
                'Hong Kong S.A.R. of China' : 'Hong Kong',
                'Ivory Coast' : 'CI',
                'Palestinian Territories' : 'PS',
                'Eswatini, Kingdom of' : 'SZ',}
        
        def get_country_code(country_name):
            if country_name[-1] == '*':
                country_name = country_name[:-1]
            if country_name in name_changes:
                country_name = name_changes[country_name]
            try:
                return pycountry.countries.get(country_name).alpha_3
            except:
                try:
                    return pycountry.countries.lookup(country_name).alpha_3
                except:    
                    raise ValueError(f"No ISO code associated with country {country_name}")
        
        df['ISO_A3'] = df['Country'].apply(get_country_code)
        return self.df
    
    def fill_null_with_time():
        pass