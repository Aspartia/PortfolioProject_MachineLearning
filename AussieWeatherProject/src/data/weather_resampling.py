from pandas.api.types import is_numeric_dtype
import pandas as pd


# -------------------------------------------------------------------
# Resampling data
# -------------------------------------------------------------------

def resample_weather_data(weather_df, freq='D'):
    weather_df=weather_df.set_index(['Date'])
    # # MultiIndex: weather_df=weather_df.set_index(['Date', 'Location'])
    # location = weather_df.groupby('Location').resample('D', level='Date').sum()

    sampling = {}
    
    for col in weather_df.columns:
        if is_numeric_dtype(weather_df[col]): 
            sampling[col] = 'mean'
        else:  
            sampling[col] = 'last'


    resampled_df = weather_df.resample(freq).apply(sampling)
    
    return resampled_df