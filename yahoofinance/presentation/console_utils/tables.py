"""Table rendering utilities"""
import pandas as pd
from tabulate import tabulate

class TableRenderer:
    @staticmethod
    def render_dataframe(df: pd.DataFrame, headers="keys") -> str:
        return tabulate(df, headers=headers, tablefmt="fancy_grid", showindex=True)
