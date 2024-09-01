
class FeatureFilter


class FeaureEngineer
    def __init__(self, df):
        self.feature_for_dv: list[str] = []
        self.filter: FeatureFilter = FeatureFilter()

    def derive_feature(self):
        // input df output df

    def filter_feature(self):
        // input df output df

class Grouper
    def __init__(self, df):
        self.groupingparams: dict = {}
    
    def groupdata(self):
        // input df output df

class ModelTrainer

    def __init__(self, df):
        self.featureengineer: FeatureEngineer = FeatureEngineer(df)
        self.grouper: Grouper = Grouper(df)

