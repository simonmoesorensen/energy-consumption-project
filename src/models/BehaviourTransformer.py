import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BehaviourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rolling_mean='10min', rolling_var='3min'):
        self.rolling_mean = rolling_mean
        self.rolling_var = rolling_var

    def adjust_acceleration(self, X, columns):
        X_adjusted = X.copy()

        trips = X_adjusted.trip.unique()

        self.adjusted_cols = []
        self.var_cols = []
        # Calculate rolling mean
        for col in columns:
            for trip in trips:
                X_adjusted.loc[X_adjusted.trip == trip, f'{col}_mean'] = X_adjusted.query(f'trip == {trip}')[col].rolling(
                    self.rolling_mean).mean()

            adjusted_col = f"adjusted_{col}"
            X_adjusted[adjusted_col] = X_adjusted[col] - X_adjusted[f'{col}_mean']

            self.adjusted_cols.append(adjusted_col)

        # Calculate rolling variance
        for col in columns:
            for trip in trips:
                X_adjusted.loc[X_adjusted.trip == trip, f'{col}_var'] = X_adjusted.query(f'trip == {trip}')[f'adjusted_{col}'].rolling(
                    self.rolling_var).var()
            self.var_cols.append(f'{col}_var')

        return X_adjusted[self.adjusted_cols + self.var_cols]

    def fit(self, X, y=None):
        if self.adjusted_cols is None:
            raise ValueError('Adjust acceleration (.adjust_acceleration(X, columns)) before fitting')

        # Quantiles for turning
        acceleration_x = list(filter(lambda x: 'x' in x, self.adjusted_cols))[0]

        X_acc = X[acceleration_x]

        # Hard quantiles
        self.q_hard_upper = X_acc.quantile(0.99)
        self.q_hard_lower = X_acc.quantile(0.01)

        # Medium quantiles
        self.q_medium_upper = X_acc.quantile(0.975)
        self.q_medium_lower = X_acc.quantile(0.025)

        # Soft quantiles
        self.q_soft_upper = X_acc.quantile(0.9)
        self.q_soft_lower = X_acc.quantile(0.1)

        # Acceleration

        # Car 0-50 km/h in seconds
        # https://www.renaultgroup.com/en/news-on-air/news/the-renault-zoe-motor-energy-efficiency-and-power/#:~:text=It%20has%20a%20torque%20of,h%20takes%20just%206.5%20seconds.
        zero_to_50_s = 3.5

        meters = 50 * 1000
        # m / s^2 = (m / s) * (1 / zero_to_100)
        max_acceleration = (meters / 3600) / zero_to_50_s

        # Acceleration in g's
        self.max_acc_gs = max_acceleration / 9.82

    def transform(self, X, y=None):
        if self.adjusted_cols is None:
            raise ValueError('Run fit before transform')

        # Get column with 'y' axis
        acceleration_y = list(filter(lambda x: 'y' in x, self.adjusted_cols))[0]

        # Get column with 'x' axis
        acceleration_x = list(filter(lambda x: 'x' in x, self.adjusted_cols))[0]

        value_columns = ["velocity", "speed_limit"] + self.adjusted_cols + self.var_cols
        X_transform = X.copy()[value_columns]

        # Speeding if exceeds by 5%
        X_transform['is_speeding'] = X_transform['velocity'] > (X_transform['speed_limit'] * 1.05)

        # Define hard acceleration as anything equal or above 80% of max
        X_transform['is_hard_acceleration'] = X_transform[acceleration_y] >= (self.max_acc_gs * 0.80)

        # Define medium acceleration as anything equal or above 50% of max and below 80%
        X_transform['is_medium_acceleration'] = (X_transform[acceleration_y] >= (self.max_acc_gs * 0.50)) & (
                X_transform[acceleration_y] < (self.max_acc_gs * 0.8))

        # Define easy acceleration as anything below 50% of max and above 20% of max
        X_transform['is_soft_acceleration'] = (X_transform[acceleration_y] < (self.max_acc_gs * 0.5)) & (
                X_transform[acceleration_y] > (self.max_acc_gs * 0.2))

        # Define hard braking as anything equal or above 80% of max
        X_transform['is_hard_braking'] = -X_transform[acceleration_y] >= (self.max_acc_gs * 0.80)

        # Define medium braking as anything equal or above 50% of max and below 80%
        X_transform['is_medium_braking'] = (-X_transform[acceleration_y] >= (self.max_acc_gs * 0.50)) & (
                -X_transform[acceleration_y] < (self.max_acc_gs * 0.8))

        # Define easy braking as anything below 50% of max and above 20% of max
        X_transform['is_soft_braking'] = (-X_transform[acceleration_y] < (self.max_acc_gs * 0.5)) & (
                -X_transform[acceleration_y] > (self.max_acc_gs * 0.2))

        X_acc = X_transform[acceleration_x]


        X_transform['is_hard_turning'] = (X_acc >= self.q_hard_upper) | (X_acc <= self.q_hard_lower)
        X_transform['is_medium_turning'] = ((X_acc < self.q_hard_upper) & (X_acc >= self.q_medium_upper)) | (
                (X_acc > self.q_hard_lower) & (X_acc <= self.q_medium_lower))
        X_transform['is_soft_turning'] = ((X_acc < self.q_medium_upper) & (X_acc >= self.q_soft_upper)) | (
                (X_acc > self.q_medium_lower) & (X_acc <= self.q_soft_lower))

        self.behavioural_columns = X_transform.drop(value_columns, axis=1).columns.tolist()
        return X_transform[self.behavioural_columns]


if __name__ == '__main__':
    df = pd.read_parquet('../../data/processed/all_data.parquet')
    transformer = BehaviourTransformer()
    transformer.fit(df, columns=['acceleration_x', 'acceleration_y'])
    out = transformer.transform(df)
    print(out)