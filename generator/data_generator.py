import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class MiningDataGenerator:
    def __init__(self):
        self.mines = ["LV-426", "Origae-6", "Fiorina 161"]
        self.start_date = datetime(2099, 11, 2)
        self.days = 40
        
        
        self.base_means = {"LV-426": 50, "Origae-6": 50, "Fiorina 161": 50}
        self.base_stds = {"LV-426": 20, "Origae-6": 20, "Fiorina 161": 20}
        self.correlation = 0.2
        self.daily_growth = 0.02  
        
        
        self.dow_factors = {6: 0.6} 
        
        
        self.events = [
            {"date": datetime(2099, 11, 10), "duration": 3, "factor": 0.4, "prob": 1.0},
            {"date": datetime(2099, 11, 20), "duration": 1, "factor": 1.4, "prob": 0.33},
        ]

    def generate(self) -> pd.DataFrame:
        dates = [self.start_date + timedelta(days=i) for i in range(self.days)]
        df = pd.DataFrame(index=dates)
        df.index.name = "Date"

        
        n = len(dates)
        mean_vec = [self.base_means[m] for m in self.mines]
        cov_matrix = self._create_covariance_matrix(mean_vec)

        random_part = np.random.multivariate_normal(mean_vec, cov_matrix, size=n)

        for i, mine in enumerate(self.mines):
            series = random_part[:, i]

            
            trend = np.power(1 + self.daily_growth, np.arange(n))

            
            dow_effect = np.array([self.dow_factors.get(d.weekday(), 1.0) for d in dates])

            
            df[mine] = series * trend * dow_effect

        
        df = self._apply_events(df)

        df = df.round(2)
        return df

    def _create_covariance_matrix(self, means: List[float]) -> np.ndarray:
        n = len(means)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov[i, j] = self.base_stds[self.mines[i]] ** 2
                else:
                    cov[i, j] = self.correlation * self.base_stds[self.mines[i]] * self.base_stds[self.mines[j]]
        return cov

    def _apply_events(self, df: pd.DataFrame) -> pd.DataFrame:
        for event in self.events:
            if np.random.random() > event["prob"]:
                continue
            start = event["date"]
            duration = event["duration"]
            factor = event["factor"]

            mask = (df.index >= start) & (df.index < start + timedelta(days=duration))
            if mask.any():
               
                center_idx = mask.argmax()
                distances = np.abs(np.arange(len(df)) - (df.index.get_loc(start) + duration // 2))
                bell = np.exp(-distances**2 / (2 * (duration/3)**2))
                df.loc[mask] *= (1 + (factor - 1) * bell[mask])
        return df