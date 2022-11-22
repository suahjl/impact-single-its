import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.tsa.api as smt
import paneleventstudy as es
from ceic_api_client.pyceic import Ceic
import telegram_send
import dataframe_image as dfi
from tabulate import tabulate
from tqdm import tqdm
import time
import glob
import os

time_start = time.time()

# 0 --- Main settings
path_transactions = 'D:/Users/ECSUAH/Desktop/Quant/'
tel_config = 'EcMetrics_Config_GeneralFlow.conf'  # EcMetrics_Config_GeneralFlow # EcMetrics_Config_RMU
# Ceic.login('suahjinglian@bnm.gov.my', 'J#D!H@MST#r1995')
redownload_gmob = False  # large csv file
run_charts = True  # slow

# I --- Functions


def telsendimg(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])


def telsendfiles(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])


def telsendmsg(conf='', msg=''):
    telegram_send.send(conf=conf,
                       messages=[msg])


def ceic2pandas_ts(input, start_date):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try: input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except: print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=start_date)  # retrieves ceicseries
        y = series_result.data
        series_name = y[0].metadata.name  # retrieves name of series
        time_points_dict = dict((tp.date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
        series = pd.Series(time_points_dict)  # convert into pandas series indexed to timepoints
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol = frame_consol.rename(columns={0: series_name})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame = frame.rename(columns={0: series_name})
            frame_consol = pd.concat([frame_consol, frame], axis=1)  # left-right concat on index (time)
        elif k < 1:
            raise NotImplementedError
        k += 1
        frame_consol = frame_consol.sort_index()
    return frame_consol


def multiEventTSplot(
        data,
        x_col,
        y_cols,
        titles,
        event_col,
        main_title,
        y_title,
        maxrows,
        maxcols,
        output_name
):
    d = data.copy()  # deep copy
    maxr = maxrows
    maxc = maxcols
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=titles)
    nr = 1
    nc = 1
    for y in tqdm(y_cols):
        # Add selected series
        fig.add_trace(
            go.Scatter(
                x=d[x_col].astype('str'),
                y=d[y],
                mode='lines',
                line=dict(width=2)
            ),
            row=nr,
            col=nc
        )
        # Add event indicator
        if d[event_col].max() < 1:
            pass
        elif d[event_col].max() == 1:
            event_onset_index = d[d[event_col] == 1].index[0]  # backs out the first row when
            event_onset_date = d.loc[d.index == event_onset_index, 'date'].reset_index(drop=True)[0]
            fig.add_vline(
                x=event_onset_date,
                line_width=1,
                line_dash='solid',
                line_color='black',
                row=nr,
                col=nc
            )
        # Move to next subplot
        nc += 1
        if nr > maxr:
            raise NotImplementedError('More subplots than allowed by dimension of main plot!')
        if nc > maxc:
            nr += 1  # next row
            nc = 1  # reset column
    for annot in fig['layout']['annotations']:
        annot['font'] = dict(size=11, color='black')  # subplot title font size
    fig.update_layout(
        title=main_title,
        yaxis_title=y_title,
        plot_bgcolor='white',
        hovermode='x',
        font=dict(color='black', size=12),
        showlegend=False,
        height=768,
        width=1366
    )
    fig.write_html(output_name + '.html')
    fig.write_image(output_name + '.png')

    return fig


# II --- Data
# Daily headline transactions
list_transactions = glob.glob(path_transactions + 'transactions_headline_*.xlsx')  # all files in the folder
latest_transactions = max(list_transactions, key=os.path.getctime)  # find the latest file
df = pd.read_excel(latest_transactions, sheet_name='daily_raw')  # NOT seasonally adjusted
df = df.rename(columns={'Unnamed: 0': 'date'})  # blank excel column label
df['date'] = df['date'].dt.date  # from datetime to date format
# df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'])
df = df.set_index('date')
# Scale transactions data to reference time
t_ref = date(2020, 1, 1)
for i in list(df.columns):
    df[i] = 100 * (df[i] / df.loc[df.index == t_ref, i].reset_index(drop=True)[0])
# Google mobility
if redownload_gmob:
    # Speed up by defining dtypes
    dict_gmob_dtype = {'country_region_code': 'str',
                       'country_region': 'str',
                       'sub_region_1': 'str',
                       'sub_region_2': 'str',
                       'metro_area': 'str',
                       'iso_3166_2_code': 'str',
                       'census_fips_code': 'str',
                       'place_id': 'str',
                       'date': 'str',
                       'retail_and_recreation_percent_change_from_baseline': 'float',
                       'grocery_and_pharmacy_percent_change_from_baseline': 'float',
                       'parks_percent_change_from_baseline': 'float',
                       'transit_stations_percent_change_from_baseline': 'float',
                       'workplaces_percent_change_from_baseline': 'float',
                       'residential_percent_change_from_baseline': 'float'}
    df_gmob = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',
                          dtype=dict_gmob_dtype)
    # Keep relevant rows
    df_gmob = df_gmob[df_gmob['country_region'] == 'Malaysia']  # keep only Malaysia
    df_gmob = df_gmob[(df_gmob['sub_region_1'].isna())]  # keep only national rows
    df_gmob['date'] = pd.to_datetime(df_gmob['date']).dt.date  # datetime format
    df_gmob = df_gmob.rename(columns={'sub_region_1': 'state',
                                      'retail_and_recreation_percent_change_from_baseline': 'mob_retl',
                                      'grocery_and_pharmacy_percent_change_from_baseline': 'mob_groc',
                                      'parks_percent_change_from_baseline': 'mob_park',
                                      'transit_stations_percent_change_from_baseline': 'mob_tran',
                                      'workplaces_percent_change_from_baseline': 'mob_work',
                                      'residential_percent_change_from_baseline': 'mob_resi'})
    df_gmob = df_gmob[['date', 'mob_retl', 'mob_groc', 'mob_park', 'mob_tran', 'mob_work', 'mob_resi']]
    # Reindex to 100 = baseline
    list_mob = ['mob_retl', 'mob_groc', 'mob_park', 'mob_tran', 'mob_work', 'mob_resi']
    for i in list_mob:
        df_gmob[i] = df_gmob[i] + 100
    # Convert into 7DMA
    # df_gmob = df_gmob.reset_index(drop=True)
    # for i in list_mob:
    #     df_gmob[i] = df_gmob[i].rolling(7, min_periods=7).mean().reset_index(drop=True)
    # save interim file
    df_gmob['date'] = df_gmob['date'].astype('str')
    df_gmob.to_parquet('gmob_cleaned_NSA.parquet', index=False)
# Read interim gmob file save in local directory
df_gmob = pd.read_parquet('gmob_cleaned_NSA.parquet')
df_gmob['date'] = pd.to_datetime(df_gmob['date']).dt.date
# Merge
df = df.merge(df_gmob, on='date', how='outer')
del df_gmob
# Set index
df = df.set_index('date')

# New column for composite mobility
df['mob_retgro'] = (df['mob_retl'] + df['mob_groc']) / 2

# New columns for lags AR(n) processes
col_og = list(df.columns)
maxlag = 1
for h in range(1, maxlag + 1):
    for i in col_og:
        df[i + '_lag' + str(h)] = df[i].shift(h)
# New columns for natural logs
col_ogpluslag = list(df.columns)
for i in col_ogpluslag:
    df[i + '_ln'] = np.log(df[i])  # ln(x)
# New columns for log-difference
for i in col_ogpluslag:
    df[i + '_ln_d'] = df[i + '_ln'] - df[i + '_ln'].shift(1)  # ln(x_{t}/x_{t-1})
# Event onsets
df.loc[df.index == date(2020, 7, 20), 'handout'] = 1  # 13, 1, 20, 24
for i in ['handout']:
    df.loc[df[i].isna(), i] = 0  # vectorised fillna
# Restrict to vicinity
T_lb = date(2020, 6, 1)  # Lockdowns effectively relaxed
T_ub = date(2020, 8, 3)  # Before the post-booster speed relaxation
df = df[((df.index >= T_lb) &
         (df.index <= T_ub))]
# Settings
event_choice = 'handout'

# III --- Time series plots
if run_charts:
    df_forchart = df.reset_index()
    # Level
    fig_level = multiEventTSplot(
        data=df_forchart,
        x_col='date',
        y_cols=['total',
                'cashless', 'cash',
                'online', 'physical',
                'fpx', 'mydebit', 'jompay', 'atm',
                'credebit_physical', 'credebit_online', 'credebit',
                'mob_retl', 'mob_groc'],
        titles=['Total',
                'Cashless', 'Cash',
                'Online', 'Physical',
                'FPX', 'MyDebit', 'JomPAY', 'ATM Withdrawals',
                'Physical Card', 'Online Card', 'Card',
                'Retail & Recreation Mobility', 'Grocery and Pharmacy Mobility'],
        event_col=event_choice,
        maxrows=4,
        maxcols=4,
        main_title='Daily Transactions and Mobility During the BSH3 Cash Handouts',
        y_title='100 = (1 Jan 2020; or Jan-Feb 2020) ',
        output_name='Output/ImpactSingleITS_2020BSH3Handouts_TSPlots_NSA_Level'
    )
    telsendimg(conf=tel_config,
               path='Output/ImpactSingleITS_2020BSH3Handouts_TSPlots_NSA_Level.png',
               cap='Daily Transactions During the BSH3 Cash Handouts')
    # Log-difference
    fig_logdiff = multiEventTSplot(
        data=df_forchart,
        x_col='date',
        y_cols=['total_ln_d',
                'cashless_ln_d', 'cash_ln_d',
                'online_ln_d', 'physical_ln_d',
                'fpx_ln_d', 'mydebit_ln_d', 'jompay_ln_d', 'atm_ln_d',
                'credebit_physical_ln_d', 'credebit_online_ln_d', 'credebit_ln_d',
                'mob_retl_ln_d', 'mob_groc_ln_d'],
        titles=['Total',
                'Cashless', 'Cash',
                'Online', 'Physical',
                'FPX', 'MyDebit', 'JomPAY', 'ATM Withdrawals',
                'Physical Card', 'Online Card', 'Card',
                'Retail & Recreation Mobility', 'Grocery and Pharmacy Mobility'],
        event_col=event_choice,
        maxrows=4,
        maxcols=4,
        main_title='Log Difference of Daily Transactions and Mobility During the BSH3 Cash Handouts',
        y_title='Growth / 100',
        output_name='Output/ImpactSingleITS_2020BSH3Handouts_TSPlots_NSA_LogDiff'
    )
    telsendimg(conf=tel_config,
               path='Output/ImpactSingleITS_2020BSH3Handouts_TSPlots_NSA_LogDiff.png',
               cap='Log Difference of Daily Transactions During the BSH3 Cash Handouts')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
