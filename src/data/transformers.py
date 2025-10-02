"""
Custom transformers for data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Select specific columns from DataFrame"""

    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X[self.columns]
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


class Datatypefix(BaseEstimator, TransformerMixin):
    """Fix data types for specific columns"""
 
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            if col == 'Ram':
                X_copy['Ram'] = X_copy['Ram'].str.replace('GB', '').astype('int32')
            elif col == 'Weight':    
                X_copy['Weight'] = X_copy['Weight'].str.replace('kg', '').astype('float32')
            elif col == 'Inches':
                X_copy['Inches'] = pd.to_numeric(X_copy['Inches'], errors='coerce')
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)


class Extractionfeature(BaseEstimator, TransformerMixin):
    """Extract features from complex columns"""

    def __init__(self, cols=None):
        self.cols = cols
        
    def extract_memory(self, df):
        """Extract memory features from Memory column"""
        df['Memory'] = df['Memory'].str.replace('\.0', '', regex=True)
        df['Memory'] = df['Memory'].str.replace('GB', '')
        df['Memory'] = df['Memory'].str.replace('TB', '000')

        new = df['Memory'].str.split("+", n=1, expand=True)
        df['first'] = new[0].str.strip()
        df['second'] = new[1]

        df["Layer1HDD"] = df['first'].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer1SSD"] = df['first'].apply(lambda x: 1 if "SSD" in x else 0)
        df["Layer1Hybrid"] = df['first'].apply(lambda x: 1 if "Hybrid" in x else 0)
        df["Layer1Flash_Storage"] = df['first'].apply(lambda x: 1 if "Flash Storage" in x else 0)

        df['first'] = df['first'].str.replace(r'\D', '', regex=True)
        df['second'].fillna("0", inplace=True)

        df["Layer2HDD"] = df['second'].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer2SSD"] = df['second'].apply(lambda x: 1 if "SSD" in x else 0)
        df["Layer2Hybrid"] = df['second'].apply(lambda x: 1 if "Hybrid" in x else 0)
        df["Layer2Flash_Storage"] = df['second'].apply(lambda x: 1 if "Flash Storage" in x else 0)
        df['second'] = df['second'].str.replace(r'\D', '', regex=True)

        df['first'] = df['first'].astype(int)
        df['second'] = df['second'].astype(int)

        df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
        df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
        df["Hybrid"] = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
        df["Flash_Storage"] = df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]
        return df

    def fetch_processor(self, text):
        """Categorize processor types"""
        if text in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
            return text
        elif text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

    def cat_os(self, x):
        """Categorize operating systems"""
        x_lower = x.lower()
        if "windows" in x_lower:
            return "Windows"
        elif "mac" in x_lower:
            return "Mac"
        else:
            return "Others/No OS/Linux"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            if col == 'ScreenResolution':
                X_copy['Touchscreen'] = X_copy['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
                X_copy['Ips'] = X_copy['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
                res = X_copy['ScreenResolution'].str.extract(r'(\d+x\d+)')
                res_split = res[0].str.split('x', n=1, expand=True)
                res_split.columns = ["X_res", "Y_res"]
                X_copy = pd.concat([X_copy, res_split], axis=1)
                X_copy['X_res'] = X_copy['X_res'].astype('int')
                X_copy['Y_res'] = X_copy['Y_res'].astype('int')
            elif col == 'Cpu':
                X_copy['Cpu Name'] = X_copy['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
                X_copy['Cpu brand'] = X_copy['Cpu Name'].apply(self.fetch_processor)
            elif col == 'Memory':
                X_copy = self.extract_memory(X_copy)
            elif col == 'Gpu':
                X_copy['Gpu brand'] = X_copy['Gpu'].apply(lambda x: x.split()[0])
            elif col == 'OpSys':
                X_copy['os'] = X_copy['OpSys'].apply(self.cat_os)

        # Add Pixels Per Inch (PPI) feature
        if 'X_res' in X_copy.columns and 'Y_res' in X_copy.columns and 'Inches' in X_copy.columns:
            X_copy['ppi'] = (((X_copy['X_res']**2) + (X_copy['Y_res']**2))**0.5 / X_copy['Inches']).astype('float')
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)


class Transformation(BaseEstimator, TransformerMixin):
    """Apply log transformation to skewed columns"""
    
    def __init__(self, cols=None, skew_threshold=1):
        self.cols = cols
        self.skew_threshold = skew_threshold
        self.skewed_columns = [] 

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.select_dtypes(include=[np.number]).columns
        
        self.skewed_columns = [
            col for col in self.cols if abs(X[col].skew()) > self.skew_threshold
        ]
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.skewed_columns:
            # Apply log transformation
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ScalingTransform(BaseEstimator, TransformerMixin):
    """Scale features using StandardScaler or MinMaxScaler"""

    def __init__(self, cols, scaling_method):
        self.cols = cols
        self.scaler_ = None
        self.scaling_method = scaling_method

    def fit(self, X, y=None):
        if self.scaling_method == "std_scale":
            self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        elif self.scaling_method == "min_max_scale":
            self.scaler_ = MinMaxScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class OneHotEncodeColumns(BaseEstimator, TransformerMixin):
    """One-hot encode categorical columns"""
    
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None
        self.column_names = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop='first')
        self.encoder.fit(X[self.cols])
        self.column_names = self.encoder.get_feature_names_out(self.cols)
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_data = self.encoder.transform(X_copy[self.cols])
        encoded_df = pd.DataFrame(encoded_data, columns=self.column_names, index=X_copy.index)
        
        X_copy = X_copy.drop(columns=self.cols)
        X_copy = pd.concat([X_copy, encoded_df], axis=1)

        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    """Label encode categorical columns"""

    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop specified columns"""

    def __init__(self, cols=None):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.cols is None:
            return X
        else:
            return X.drop(self.cols, axis=1)


class FullPipeline1:
    """Complete preprocessing pipeline"""
    
    def __init__(self, feats_min_max_scale=None, num_cols=None):
        self.all_cols = ['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram',
                         'Memory', 'Gpu', 'OpSys', 'Weight']

        self.drop_cols = ['ScreenResolution', 'X_res', 'Y_res', 'Inches', 'Cpu', 'Cpu Name', 'first', 'second', 
                          'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 
                          'Layer2Hybrid', 'Layer2Flash_Storage', 'Hybrid','Flash_Storage', 'Memory', 'Gpu', 'OpSys']

        self.encode_cols = ['TypeName', 'Company', 'Cpu brand', 'Gpu brand', 'os']
        
        # Default scaling columns if not provided
        if feats_min_max_scale is None:
            feats_min_max_scale = ['Weight', 'Ram', 'ppi', 'HDD', 'SSD']
        if num_cols is None:
            num_cols = ['Weight', 'Ram', 'ppi', 'HDD', 'SSD', 'Price']
            
        self.scale_cols = feats_min_max_scale

        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.all_cols)),
            ('data_fix', Datatypefix(cols=self.all_cols)),
            ('extract_feature', Extractionfeature(cols=self.all_cols)),
            ('power_transformation', Transformation(cols=list(set(num_cols) - set(['Price'])))),
            ('label_encode', OneHotEncodeColumns(cols=self.encode_cols)),
            ('scaling', ScalingTransform(cols=list(set(self.scale_cols)-set(['Price'])),
                                         scaling_method="min_max_scale")),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols))
        ])

        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Price'])),
            ('power_transformation', Transformation(cols=['Price'])),
            ('scaling', ScalingTransform(cols=['Price'], scaling_method="std_scale"))
        ])

    def fit_transform(self, X_train, y_train):
        """Fit the pipeline and transform training data"""
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train

    def transform(self, X_test, y_test):
        """Transform test data using fitted pipeline"""
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test