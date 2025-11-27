"""Utility functions for the multimodal RAG system"""
import pandas as pd

def load_excel(path):
    """Load Excel file and return DataFrame"""
    return pd.read_excel(path)

def get_product_by_id(df, row_id):
    """Get product information by row ID"""
    return df.iloc[int(row_id)].to_dict()
