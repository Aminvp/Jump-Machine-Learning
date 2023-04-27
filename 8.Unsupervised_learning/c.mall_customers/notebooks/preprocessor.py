import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        
        self.Gender = {
            "Male" : 1 ,
            "Female" : 0
        }

        self.column_scaler = ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    def handle_missing_values (self) :
        self.df.fillna(0, inplace=True)

    """def handle_scaler(self):
        scaler = MinMaxScaler()
        self.df[self.column_scaler] = scaler.fit_transform(self.df[self.column_scaler])"""
      
    def transform (self) : 
        self.handle_missing_values()
        self.df.loc[:, 'Gender'] = self.df.Gender.replace(self.Gender)
        #self.handle_scaler()
        return self.df
