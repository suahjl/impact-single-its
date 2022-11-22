import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
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
path_transactions = ''
tel_config = 'EcMetrics_Config_GeneralFlow.conf'  # EcMetrics_Config_GeneralFlow # EcMetrics_Config_RMU
# Ceic.login('suahjinglian@bnm.gov.my', 'J#D!H@MST#r1995')
redownload_gmob = False  # large csv file
run_prechecks = True  # spam
show_ci = True  # VAR CIs diverge
quantum_handout_myr = 3000000000  # MYR3 billion (MoF press coverage)
# https://www.astroawani.com/berita-malaysia/bayaran-bsh-dibuat-sebelum-akhir-bulan-julai-menteri-kewangan-249268

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
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=y_cols)
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
df = pd.read_excel(latest_transactions, sheet_name='daily_sa')  # seasonally adjusted
df = df.rename(columns={'Unnamed: 0': 'date'})  # blank excel column label
df['date'] = df['date'].dt.date  # from datetime to date format
df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'])
df = df.set_index('date')
# Scale transactions data to reference time
t_ref = date(2020, 1, 1)
myr_ref = df[df.index == t_ref].iloc[0]  # Store MYR value during t_ref
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
    df_gmob = df_gmob.reset_index(drop=True)
    for i in list_mob:
        df_gmob[i] = df_gmob[i].rolling(7, min_periods=7).mean().reset_index(drop=True)
    # save interim file
    df_gmob['date'] = df_gmob['date'].astype('str')
    df_gmob.to_parquet('gmob_cleaned.parquet', index=False)
# Read interim gmob file save in local directory
df_gmob = pd.read_parquet('gmob_cleaned.parquet')
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
df.loc[df.index == date(2020, 7, 20), 'handout'] = 1  # 13 (OG), 1, 20, 24 (a few announcements)
for i in ['handout']:
    df.loc[df[i].isna(), i] = 0  # vectorised fillna
# Settings
event_choice = 'handout'
event_date = date(2020, 7, 20)  # 13 (OG in March), 1, 20, 24
# Restrict to vicinity
T_lb = date(2020, 6, 1)
T_ub = date(2020, 8, 3)
df = df[((df.index >= T_lb) &
         (df.index <= T_ub))]
postevent_length = (T_ub - event_date).days

# III --- Time series plots

# IV --- Setup data for analysis
# Generate relative time column
df = es.dropmissing(data=df, event=event_choice)  # the May handout coincided with some freedom of movement
df['ct'] = np.arange(len(df))  # numeric calendar time
df['reltime'] = df['ct'] - df.loc[df[event_choice] == 1, 'ct'].reset_index(drop=True)[0]

# V --- Pre-Checks for interrupted time series (forecast counterfactual using pre-data)
target_endog = ['total_ln_d']
loglevels_endog = ['total_ln', 'mob_retgro_ln', 'mob_retl_ln', 'mob_groc_ln']
logdiff_endog = ['total_ln_d', 'mob_retgro_ln_d', 'mob_retl_ln_d', 'mob_groc_ln_d']
dict_raw_to_nice = {'total_ln': 'ln(Total)',
                    'mob_retgro_ln': 'ln(Retail & Grocery Mobility)',
                    'mob_retl_ln': 'ln(Retail Mobility)',
                    'mob_groc_ln': 'ln(Grocery Mobility)',
                    'total_ln_d': '\u0394ln(Total)',
                    'mob_retgro_ln_d': '\u0394ln(Retail & Grocery Mobility)',
                    'mob_retl_ln_d': '\u0394ln(Retail Mobility)',
                    'mob_groc_ln_d': '\u0394ln(Grocery Mobility)'}
preevent_only = True  # keep only pre-event period
if preevent_only:
    d = df[df['reltime'] < 0]
elif not preevent_only:
    d = df.copy()
if run_prechecks:
    # Check if VECM should be used
    mackinnonp_consol = pd.DataFrame(columns=['RHS_Variable', 'pval'])
    for i in tqdm(loglevels_endog[1:]):  # exclude the first one from loop
        mackinnonp = pd.DataFrame([smt.coint(y0=d['total_ln'], y1=d[i], return_results=False)[1]], columns=['pval'])
        mackinnonp['RHS_Variable'] = i
        mackinnonp_consol = pd.concat([mackinnonp_consol, mackinnonp], axis=0)
    tab_mackinnonp = tabulate(mackinnonp_consol, showindex=False, headers='keys', tablefmt='github')
    print(tab_mackinnonp)
    telsendmsg(conf=tel_config,
               msg=tab_mackinnonp)
    # Check for unit roots
    urtest_consol = pd.DataFrame(columns=['variable', 'cons_trend', 'pval'])
    for i in tqdm(loglevels_endog + logdiff_endog):
        for cons in ['n', 'c', 'ct', 'ctt']:
            pval_null_uroot = smt.adfuller(x=d[i],
                                           autolag='BIC',
                                           regression=cons)[1]
            urtest = pd.DataFrame([pval_null_uroot], columns=['pval'])
            urtest['variable'] = i
            urtest['cons_trend'] = cons
            urtest_consol = pd.concat([urtest_consol, urtest], axis=0)
    urtest_consol.loc[urtest_consol['pval'] > 0.05, 'UnitRoot'] = 1
    urtest_consol.loc[urtest_consol['pval'] <= 0.05, 'UnitRoot'] = 0
    urtest_consol['cons_trend'] = urtest_consol['cons_trend'].replace({'n': 'None',
                                                                       'c': 'Constant',
                                                                       'ct': 'Constant & Trend',
                                                                       'ctt': 'Constant & Quad Trend'})
    urtest_consol['UnitRoot'] = urtest_consol['UnitRoot'].astype('int')
    urtest_consol['UnitRoot'] = urtest_consol['UnitRoot'].replace({0: 'No',
                                                                   1: 'Yes'})
    urtest_consol = urtest_consol.round(3)
    tab_urtest = tabulate(urtest_consol, showindex=False, headers='keys', tablefmt='github')
    print(tab_urtest)
    urtest_consol = urtest_consol.reset_index(drop=True)
    urtest_consol['variable'] = urtest_consol['variable'].replace(dict_raw_to_nice)
    dfi.export(urtest_consol, 'Output/ImpactSingleITS_2020BSH3Handouts_URoot.png')
    telsendimg(conf=tel_config,
               path='Output/ImpactSingleITS_2020BSH3Handouts_URoot.png',
               cap='Unit root tests')

# VI --- Single entity interrupted time series
# Setup
list_target_endog = ['total_ln_d']
list_target_endog_level = ['total_ln']
list_target_endog_myr = ['total']
list_loglevels_endog = ['total_ln']  # ensure no overlaps
list_logdiff_endog = ['total_ln_d']  # ensure no overlaps
list_loglevels_exog = ['mob_retl_ln', 'mob_groc_ln']  # 'mob_retl_ln', 'mob_groc_ln'
list_logdiff_exog = ['mob_retl_ln_d', 'mob_groc_ln_d']  # 'mob_retl_ln_d', 'mob_groc_ln_d'


def single_its_arx(data=df,
                   logdiff_endog=list_logdiff_endog,
                   logdiff_exog=list_logdiff_exog,
                   target_endog=list_target_endog,
                   target_endog_level=list_target_endog_level,
                   target_endog_myr=list_target_endog_myr,
                   handout_myr=quantum_handout_myr,
                   exog_quad=False,
                   exog_cubic=False,
                   alpha='n'):
    # Preliminaries
    d_full = data.copy()
    if (exog_quad is True) & (exog_cubic is True):
        raise NotImplementedError('Make up your mind, only one of exog_quad and exog_cubic can be True')
    if exog_quad is True:
        for i in logdiff_exog:
            d_full[i + '_quad'] = d_full[i] ** 2
        logdiff_exog = logdiff_exog + [i + '_quad' for i in logdiff_exog]
    elif exog_cubic is True:
        for i in logdiff_exog:
            d_full[i + '_quad'] = d_full[i] ** 2
            d_full[i + '_cubic'] = d_full[i] ** 3
        exog_quad_list = [i + '_quad' for i in logdiff_exog]
        exog_cubic_list = [i + '_cubic' for i in logdiff_exog]
        logdiff_exog = logdiff_exog + exog_quad_list + exog_cubic_list

    # Trim dataframe copy
    d = d_full[d_full['reltime'] < 0]  # only keep PRE-event period
    d_post = d_full[d_full['reltime'] >= 0]  # only keep POST-event period

    # Estimation
    ar_order = ar_select_order(d[logdiff_endog], maxlag=7, trend=alpha, ic='hqic').ar_lags
    if ar_order is None: ar_order = 1  # fail-safe, defaults to AR(1)-X if lag selection is indifferent
    elif ar_order is not None: ar_order = ar_order[0]
    est_arx = smt.AutoReg(endog=d[logdiff_endog],
                          exog=d[logdiff_exog],
                          trend=alpha,
                          lags=range(1, ar_order + 1))
    res_arx = est_arx.fit(cov_type='HAC', cov_kwds={'maxlags': ar_order})

    # In-sample prediction (pre-event period; with interval)
    pred = res_arx.get_prediction(start=0,
                                  end=len(d) - 1,
                                  dynamic=False,
                                  exog=d[logdiff_exog].values)  # zero-indexed
    pred = pd.concat([pred.predicted_mean, pred.se_mean], axis=1)
    pred = pred.reset_index().rename(columns={'index': 'date',
                                              'predicted_mean': 'ptpred_' + target_endog[0]})
    pred['date'] = pred['date'].dt.date
    pred = pred.set_index('date')
    pred['lbpred_' + target_endog[0]] = pred['ptpred_' + target_endog[0]] - 1.96 * pred['mean_se']
    pred['ubpred_' + target_endog[0]] = pred['ptpred_' + target_endog[0]] + 1.96 * pred['mean_se']
    del pred['mean_se']

    # In-sample prediction: Extrapolate level in-sample prediction
    df_pred = pd.concat([d[loglevels_endog], pred],
                         axis=1)
    df_pred = df_pred.astype('float')
    for k in tqdm(range(len(d) + 1)):
        try:
            if k == 1:
                df_pred.loc[(df_pred.index == df_pred.index.min()),
                            'ptlvlpred_' + target_endog_level[0]] = df_pred[target_endog_level[0]]

                df_pred.loc[(df_pred.index == df_pred.index.min()),
                            'lblvlpred_' + target_endog_level[0]] = df_pred[target_endog_level[0]]

                df_pred.loc[(df_pred.index == df_pred.index.min()),
                            'ublvlpred_' + target_endog_level[0]] = df_pred[target_endog_level[0]]
            elif k >= 1:
                df_pred.loc[df_pred['ptlvlpred_' + target_endog_level[0]].isna(),
                            'ptlvlpred_' + target_endog_level[0]] = \
                    df_pred[target_endog_level[0]].shift(1) + \
                    df_pred['ptpred_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}

                df_pred.loc[df_pred['lblvlpred_' + target_endog_level[0]].isna(),
                            'lblvlpred_' + target_endog_level[0]] = \
                    df_pred[target_endog_level[0]].shift(1) + \
                    df_pred['lbpred_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}

                df_pred.loc[df_pred['ublvlpred_' + target_endog_level[0]].isna(),
                            'ublvlpred_' + target_endog_level[0]] = \
                    df_pred[target_endog_level[0]].shift(1) + \
                    df_pred['ubpred_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}
        except:
            print('No more missing values in pre-event period')

    # In-sample prediction: Bring in observed pre-event observations
    df_pred[loglevels_endog] = df_pred[loglevels_endog].combine_first(df[loglevels_endog])
    df_pred = pd.concat([df['reltime'], df_pred], axis=1)
    df_pred = df_pred.set_index('reltime')

    # In-sample prediction: Keep only log-levels
    df_pred = df_pred[loglevels_endog + ['ptlvlpred_' + target_endog_level[0],
                                         'lblvlpred_' + target_endog_level[0],
                                         'ublvlpred_' + target_endog_level[0]]]

    # Forecast (with interval)
    fcast = res_arx.get_prediction(start=len(d),  # zero-indexed
                                   end=len(d) + len(d_post) - 1,
                                   exog_oos=d_post[logdiff_exog].values)
    fcast = pd.concat([fcast.predicted_mean, fcast.se_mean], axis=1)
    fcast = fcast.reset_index().rename(columns={'index': 'date',
                                                'predicted_mean': 'ptfcast_' + target_endog[0]})
    fcast['date'] = fcast['date'].dt.date
    fcast = fcast.set_index('date')
    fcast['lbfcast_' + target_endog[0]] = fcast['ptfcast_' + target_endog[0]] - 1.96 * fcast['mean_se']
    fcast['ubfcast_' + target_endog[0]] = fcast['ptfcast_' + target_endog[0]] + 1.96 * fcast['mean_se']
    del fcast['mean_se']

    # Forecast: Extrapolate level forecast
    df_fcast = pd.concat([d[loglevels_endog], fcast],
                         axis=1)
    df_fcast = df_fcast.astype('float')
    for k in tqdm(range(postevent_length + 1)):
        try:
            if k == 0:  # use last observed value
                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) & (df_fcast.index == event_date)),
                             'ptcfact_' + target_endog_level[0]] = \
                    df_fcast[target_endog_level[0]].shift(1) + \
                    df_fcast['ptfcast_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}

                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) & (df_fcast.index == event_date)),
                             'lbcfact_' + target_endog_level[0]] = \
                    df_fcast[target_endog_level[0]].shift(1) + \
                    df_fcast['lbfcast_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}

                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) & (df_fcast.index == event_date)),
                             'ubcfact_' + target_endog_level[0]] = \
                    df_fcast[target_endog_level[0]].shift(1) + \
                    df_fcast['ubfcast_' + target_endog[0]]  # yhat_{t} = y_{t-1} + \Delta yhat_{t}

            elif k >= 1:  # use last dynamic forecast values
                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) &
                              (df_fcast.index == event_date + timedelta(days=k))),
                             'ptcfact_' + target_endog_level[0]] = \
                    df_fcast['ptcfact_' + target_endog_level[0]].shift(1) + \
                    df_fcast['ptfcast_' + target_endog[0]]  # yhat_{t} = yhat_{t-1} + \Delta yhat_{t}

                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) &
                              (df_fcast.index == event_date + timedelta(days=k))),
                             'lbcfact_' + target_endog_level[0]] = \
                    df_fcast['ptcfact_' + target_endog_level[0]].shift(1) + \
                    df_fcast['lbfcast_' + target_endog[0]]  # yhat_{t} = yhat_{t-1} + \Delta yhat_{t}

                df_fcast.loc[((df_fcast[target_endog_level[0]].isna()) &
                              (df_fcast.index == event_date + timedelta(days=k))),
                             'ubcfact_' + target_endog_level[0]] = \
                    df_fcast['ptcfact_' + target_endog_level[0]].shift(1) + \
                    df_fcast['ubfcast_' + target_endog[0]]  # yhat_{t} = yhat_{t-1} + \Delta yhat_{t}
        except:
            print('No more missing values in post-event period')

    # Forecast: Bring in observed post-event observations
    df_fcast[loglevels_endog] = df_fcast[loglevels_endog].combine_first(df[loglevels_endog])
    df_fcast = pd.concat([df['reltime'], df_fcast], axis=1)
    df_fcast = df_fcast.set_index('reltime')

    # Forecast: Keep only log-levels
    df_fcast = df_fcast[loglevels_endog + ['ptcfact_' + target_endog_level[0],
                                           'lbcfact_' + target_endog_level[0],
                                           'ubcfact_' + target_endog_level[0]]]

    # Forecast: Calculate "impact"
    df_fcast['pt_impact'] = df_fcast[target_endog_level[0]] - df_fcast['ptcfact_' + target_endog_level[0]]
    df_fcast['ub_impact'] = df_fcast[target_endog_level[0]] - df_fcast['lbcfact_' + target_endog_level[0]]  # flipped
    df_fcast['lb_impact'] = df_fcast[target_endog_level[0]] - df_fcast['ubcfact_' + target_endog_level[0]]  # flipped

    # Forecast: Calculate cumulative impact and convert into MYR
    df_fcast_myr = myr_ref[target_endog_myr[0]] * (np.exp(df_fcast) / 100)
    df_fcast_myr.columns = df_fcast_myr.columns.str.replace('_ln', '', regex=False)
    for i, j in zip(['pt', 'lb', 'ub'], ['pt', 'ub', 'lb']):  # flipped
        df_fcast_myr[i + '_impact'] = df_fcast_myr[target_endog_myr[0]] - df_fcast_myr[
            j + 'cfact_' + target_endog_myr[0]]
    for i in ['pt', 'lb', 'ub']:
        df_fcast_myr[i + '_cumulimpact'] = np.cumsum(df_fcast_myr[i + '_impact'])
    for i in ['pt', 'lb', 'ub']:
        df_fcast_myr[i + '_cumulimpact'] = np.cumsum(df_fcast_myr[i + '_impact'])
    for i in ['pt', 'lb', 'ub']:
        df_fcast_myr[i + '_multiplier'] = df_fcast_myr[i + '_cumulimpact'] / handout_myr

    # Output
    return df_fcast, df_fcast_myr, ar_order, df_pred


# Execute
df_fcast, df_fcast_myr, ar_order, df_pred = single_its_arx(data=df, exog_quad=False, exog_cubic=False, alpha='n')
df_fcast_quad, df_fcast_myr_quad, ar_order_quad, df_pred_quad = single_its_arx(data=df, exog_quad=True, exog_cubic=False, alpha='c')
df_fcast_cubic, df_fcast_myr_cubic, ar_order_cubic, df_pred_cubic = single_its_arx(data=df, exog_quad=False, exog_cubic=True, alpha='c')

# Plot impact estimates
# Keep only post-event period for impact est (no counterfactual for pre-period)
df_fcast_post = df_fcast[df_fcast.index >= 0]
fig_impact = go.Figure()
# Add point estimate
fig_impact.add_trace(
    go.Scatter(
        x=df_fcast_post.index.astype('str'),
        y=df_fcast_post['pt_impact'],
        name='Point Estimate',
        mode='lines',
        line=dict(color='black', width=3)
    )
)
# Add reference
df_fcast_post['ref'] = 0
fig_impact.add_trace(
    go.Scatter(
        x=df_fcast_post.index.astype('str'),
        y=df_fcast_post['ref'],
        name='Reference (Y=0)',
        mode='lines',
        line=dict(color='darkgray', width=1.5)
    )
)
del df_fcast_post['ref']
if show_ci:
    # Add lower bound
    fig_impact.add_trace(
        go.Scatter(
            x=df_fcast_post.index.astype('str'),
            y=df_fcast_post['lb_impact'],
            name='Lower Bound',
            mode='lines',
            line=dict(color='black', width=1.5, dash='dash')
        )
    )
    # Add upper bound
    fig_impact.add_trace(
        go.Scatter(
            x=df_fcast_post.index.astype('str'),
            y=df_fcast_post['ub_impact'],
            name='Upper Bound',
            mode='lines',
            line=dict(color='black', width=1.5, dash='dash')
        )
    )
# Layout
fig_impact.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Impact of BSH3 Cash Handouts on Daily Transactions',
    yaxis_title='% / 100 of Counterfactual',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)
# Export chart
fig_impact.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot.html')
fig_impact.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot.png',
           cap='Single Entity Interrupted Time Series: ' +
          'Impact of BSH3 Cash Handouts on Daily Transactions')

# Plot cumulative impact estimates (MYR)
# Keep only post-event period for impact est (no counterfactual for pre-period)
df_fcast_myr_post = df_fcast_myr[df_fcast_myr.index >= 0]
fig_impact_cumul_myr = go.Figure()
# Add point estimate
fig_impact_cumul_myr.add_trace(
    go.Scatter(
        x=df_fcast_myr_post.index.astype('str'),
        y=df_fcast_myr_post['pt_cumulimpact'],
        name='Point Estimate',
        mode='lines',
        line=dict(color='black', width=3)
    )
)
# Add reference
df_fcast_myr_post['ref'] = 0
fig_impact_cumul_myr.add_trace(
    go.Scatter(
        x=df_fcast_myr_post.index.astype('str'),
        y=df_fcast_myr_post['ref'],
        name='Reference (Y=0)',
        mode='lines',
        line=dict(color='darkgray', width=1.5)
    )
)
del df_fcast_myr_post['ref']
if show_ci:
    # Add lower bound
    fig_impact_cumul_myr.add_trace(
        go.Scatter(
            x=df_fcast_myr_post.index.astype('str'),
            y=df_fcast_myr_post['lb_cumulimpact'],
            name='Lower Bound',
            mode='lines',
            line=dict(color='black', width=1.5, dash='dash')
        )
    )
    # Add upper bound
    fig_impact_cumul_myr.add_trace(
        go.Scatter(
            x=df_fcast_myr_post.index.astype('str'),
            y=df_fcast_myr_post['ub_cumulimpact'],
            name='Upper Bound',
            mode='lines',
            line=dict(color='black', width=1.5, dash='dash')
        )
    )
# Layout
fig_impact_cumul_myr.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Cumulative MYR Impact of BSH3 Cash Handouts on Daily Transactions',
    yaxis_title='MYR',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)

# Export chart
fig_impact_cumul_myr.write_html('Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot.html')
fig_impact_cumul_myr.write_image('Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot.png',
           cap='Single Entity Interrupted Time Series: ' +
               'Cumulative MYR Impact of BSH3 Cash Handouts on Daily Transactions')

# Plot cumulative multiplier estimates (cumul impact / total handouts)
# Keep only post-event period for impact est (no counterfactual for pre-period)
# df_fcast_myr_post = df_fcast_myr[df_fcast_myr.index >= 0]
df_fcast_myr_end = df_fcast_myr[df_fcast_myr.index == df_fcast_myr.index.max()].reset_index(drop=True)
df_fcast_myr_end = df_fcast_myr_end.round(3)
df_fcast_myr_end.to_csv('multiplier_2020BSH3Handouts.txt', index=False, sep='|')
fig_multiplier = go.Figure()
# Add point estimate
fig_multiplier.add_trace(
    go.Bar(
        y=df_fcast_myr_end['pt_multiplier'],
        name='Point Estimate',
        error_y=dict(
            type='constant',
            value=(df_fcast_myr_end['ub_multiplier'] - df_fcast_myr_end['pt_multiplier']).reset_index(drop=True)[0],
            valueminus=(df_fcast_myr_end['pt_multiplier'] - df_fcast_myr_end['lb_multiplier']).reset_index(drop=True)[0]
        ),
        marker=dict(color='lightgray'),
        width=0.3,
        text=df_fcast_myr_end['pt_multiplier'],
        textposition='outside'
    )
)
# Layout
# fig_multiplier.update_yaxes(range=[0, 0.8])
fig_multiplier.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Spending Multiplier of BSH3 Cash Handouts on Daily Transactions (Cumulative Impact / Total Handouts)',
    yaxis_title='Spending Multiplier',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    uniformtext=dict(mode='show', minsize=22),
    height=768,
    width=1366
)
# Export chart
fig_multiplier.write_html('Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot.html')
fig_multiplier.write_image('Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot.png',
           cap='Single Entity Interrupted Time Series: ' +
               'Spending Multiplier of BSH3 Cash Handouts on Daily Transactions (Cumulative Impact / Total Handouts)')

# Plot observed + counterfactual series
fig_comp = go.Figure()
# Add point estimate
fig_comp.add_trace(
    go.Scatter(
        x=df_fcast.index.astype('str'),
        y=df_fcast[list_target_endog_level[0]],
        name='Observed',
        mode='lines',
        line=dict(color='black', width=2)
    )
)
# Add counterfactual point estimate
fig_comp.add_trace(
    go.Scatter(
        x=df_fcast.index.astype('str'),
        y=df_fcast['ptcfact_' + list_target_endog_level[0]],
        name='Counterfactual (Point Est)',
        mode='lines',
        line=dict(color='red', width=2, dash='dash')
    )
)
if show_ci:
    # Add counterfactual lower bound
    fig_comp.add_trace(
        go.Scatter(
            x=df_fcast.index.astype('str'),
            y=df_fcast['lbcfact_' + list_target_endog_level[0]],
            name='Counterfactual (Lower Bound)',
            mode='lines',
            line=dict(color='lightcoral', width=1.5, dash='dash')
        )
    )
    # Add counterfactual upper bound
    fig_comp.add_trace(
        go.Scatter(
            x=df_fcast.index.astype('str'),
            y=df_fcast['ubcfact_' + list_target_endog_level[0]],
            name='Counterfactual (Upper Bound)',
            mode='lines',
            line=dict(color='lightcoral', width=1.5, dash='dash')
        )
    )
# Layout
fig_comp.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Observed and Counterfactual Daily Transactions',
    yaxis_title='Natural Logarithm; ln(100) = 4.605 = ' + str(t_ref) ,
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)

# Plot observed + counterfactual + in-sample prediction series
fig_comp_withpred = go.Figure(fig_comp)
# Add pre-event prediction point estimate
fig_comp_withpred.add_trace(
    go.Scatter(
        x=df_pred.index.astype('str'),
        y=df_pred['ptlvlpred_' + list_target_endog_level[0]],
        name='Pre-Event Prediction (Point Est)',
        mode='lines',
        line=dict(color='blue', width=1.5, dash='dash')
    )
)
if show_ci:
    # Add pre-event prediction lower bound
    fig_comp_withpred.add_trace(
        go.Scatter(
            x=df_pred.index.astype('str'),
            y=df_pred['lblvlpred_' + list_target_endog_level[0]],
            name='Pre-Event Prediction (Lower Bound)',
            mode='lines',
            line=dict(color='lightblue', width=1, dash='dash')
        )
    )
    # Add pre-event prediction upper bound
    fig_comp_withpred.add_trace(
        go.Scatter(
            x=df_pred.index.astype('str'),
            y=df_pred['ublvlpred_' + list_target_endog_level[0]],
            name='Pre-Event Prediction (Upper Bound)',
            mode='lines',
            line=dict(color='lightblue', width=1, dash='dash')
        )
    )
# Layout
fig_comp_withpred.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Observed, Counterfactual, and Pre-Event Predicted Daily Transactions',
    yaxis_title='Natural Logarithm; ln(100) = 4.605 = ' + str(t_ref) ,
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)
# Export chart
fig_comp.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot.html')
fig_comp.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot.png',
           cap='Single Entity Interrupted Time Series: ' +
               'Observed and Counterfactual Daily Transactions (with 95% CIs)')

fig_comp_withpred.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot.html')
fig_comp_withpred.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot.png',
           cap='Single Entity Interrupted Time Series: ' +
               'Observed, Counterfactual, and Pre-Event Predicted Daily Transactions (with 95% CIs)')

# ------- Sensitivity charts

# Daily impact
# Keep only post-event period for impact est (no counterfactual for pre-period)
df_fcast_quad_post = df_fcast_quad[df_fcast_quad.index >= 0]
df_fcast_cubic_post = df_fcast_cubic[df_fcast_cubic.index >= 0]
fig_impact_sens = go.Figure()
# Add point estimate
fig_impact_sens.add_trace(
    go.Scatter(
        x=df_fcast_post.index.astype('str'),
        y=df_fcast_post['pt_impact'],
        name='Linear',
        mode='lines',
        line=dict(color='black', width=3)
    )
)
fig_impact_sens.add_trace(
    go.Scatter(
        x=df_fcast_quad_post.index.astype('str'),
        y=df_fcast_quad_post['pt_impact'],
        name='Quadratic',
        mode='lines',
        line=dict(color='crimson', width=3)
    )
)
fig_impact_sens.add_trace(
    go.Scatter(
        x=df_fcast_cubic_post.index.astype('str'),
        y=df_fcast_cubic_post['pt_impact'],
        name='Cubic',
        mode='lines',
        line=dict(color='darkblue', width=3)
    )
)
# Add reference
df_fcast_quad_post['ref'] = 0
fig_impact_sens.add_trace(
    go.Scatter(
        x=df_fcast_quad_post.index.astype('str'),
        y=df_fcast_quad_post['ref'],
        name='Reference (Y=0)',
        mode='lines',
        line=dict(color='darkgray', width=1.5)
    )
)
del df_fcast_quad_post['ref']
# Layout
fig_impact_sens.update_layout(
    title='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
          'Impact of CMCO Cash Handouts on Daily Transactions',
    yaxis_title='% / 100 of Counterfactual',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)
# Export chart
fig_impact_sens.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot_Sensitivity.html')
fig_impact_sens.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot_Sensitivity.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ImpactPlot_Sensitivity.png',
           cap='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
               'Impact of CMCO Cash Handouts on Daily Transactions')

# Cumulative impact
# Keep only post-event period for impact est (no counterfactual for pre-period)
df_fcast_myr_quad_post = df_fcast_myr_quad[df_fcast_myr_quad.index >= 0]
df_fcast_myr_cubic_post = df_fcast_myr_cubic[df_fcast_myr_cubic.index >= 0]
fig_impact_cumul_myr_sens = go.Figure()
# Add point estimate
fig_impact_cumul_myr_sens.add_trace(
    go.Scatter(
        x=df_fcast_myr_post.index.astype('str'),
        y=df_fcast_myr_post['pt_cumulimpact'],
        name='Linear',
        mode='lines',
        line=dict(color='black', width=3)
    )
)
fig_impact_cumul_myr_sens.add_trace(
    go.Scatter(
        x=df_fcast_myr_quad_post.index.astype('str'),
        y=df_fcast_myr_quad_post['pt_cumulimpact'],
        name='Quadratic',
        mode='lines',
        line=dict(color='crimson', width=3)
    )
)
fig_impact_cumul_myr_sens.add_trace(
    go.Scatter(
        x=df_fcast_myr_cubic_post.index.astype('str'),
        y=df_fcast_myr_cubic_post['pt_cumulimpact'],
        name='Cubic',
        mode='lines',
        line=dict(color='darkblue', width=3)
    )
)
# Add reference
df_fcast_myr_post['ref'] = 0
fig_impact_cumul_myr_sens.add_trace(
    go.Scatter(
        x=df_fcast_myr_post.index.astype('str'),
        y=df_fcast_myr_post['ref'],
        name='Reference (Y=0)',
        mode='lines',
        line=dict(color='darkgray', width=1.5)
    )
)
del df_fcast_myr_post['ref']
# Layout
fig_impact_cumul_myr_sens.update_layout(
    title='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
          'Cumulative MYR Impact of CMCO Cash Handouts on Daily Transactions',
    yaxis_title='MYR',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)
# Export chart
fig_impact_cumul_myr_sens.write_html('Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot_Sensitivity.html')
fig_impact_cumul_myr_sens.write_image('Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot_Sensitivity.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot_Sensitivity.png',
           cap='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
               'Cumulative MYR Impact of CMCO Cash Handouts on Daily Transactions')

# Multiplier
# Keep only post-event period for impact est (no counterfactual for pre-period)
# df_fcast_myr_post = df_fcast_myr[df_fcast_myr.index >= 0]
df_fcast_myr_end_quad = df_fcast_myr_quad[df_fcast_myr_quad.index == df_fcast_myr_quad.index.max()].reset_index(drop=True)
df_fcast_myr_end_quad = df_fcast_myr_end_quad.round(3)
df_fcast_myr_end_cubic = df_fcast_myr_cubic[df_fcast_myr_cubic.index == df_fcast_myr_cubic.index.max()].reset_index(drop=True)
df_fcast_myr_end_cubic = df_fcast_myr_end_cubic.round(3)
fig_multiplier_sens = go.Figure()
# Add point estimate
fig_multiplier_sens.add_trace(
    go.Bar(
        y=df_fcast_myr_end['pt_multiplier'],
        name='Linear',
        error_y=dict(
            type='constant',
            value=(df_fcast_myr_end['ub_multiplier'] - df_fcast_myr_end['pt_multiplier']).reset_index(drop=True)[0],
            valueminus=(df_fcast_myr_end['pt_multiplier'] - df_fcast_myr_end['lb_multiplier']).reset_index(drop=True)[0]
        ),
        marker=dict(color='lightgray'),
        width=0.3,
        text=df_fcast_myr_end['pt_multiplier'],
        textposition='outside'
    )
)
fig_multiplier_sens.add_trace(
    go.Bar(
        y=df_fcast_myr_end_quad['pt_multiplier'],
        name='Quadratic',
        error_y=dict(
            type='constant',
            value=(df_fcast_myr_end_quad['ub_multiplier'] - df_fcast_myr_end_quad['pt_multiplier']).reset_index(drop=True)[0],
            valueminus=(df_fcast_myr_end_quad['pt_multiplier'] - df_fcast_myr_end_quad['lb_multiplier']).reset_index(drop=True)[0]
        ),
        marker=dict(color='lightcoral'),
        width=0.3,
        text=df_fcast_myr_end_quad['pt_multiplier'],
        textposition='outside'
    )
)
fig_multiplier_sens.add_trace(
    go.Bar(
        y=df_fcast_myr_end_cubic['pt_multiplier'],
        name='Cubic',
        error_y=dict(
            type='constant',
            value=(df_fcast_myr_end_cubic['ub_multiplier'] - df_fcast_myr_end_cubic['pt_multiplier']).reset_index(drop=True)[0],
            valueminus=(df_fcast_myr_end_cubic['pt_multiplier'] - df_fcast_myr_end_cubic['lb_multiplier']).reset_index(drop=True)[0]
        ),
        marker=dict(color='lightblue'),
        width=0.3,
        text=df_fcast_myr_end_cubic['pt_multiplier'],
        textposition='outside'
    )
)
# Layout
# fig_multiplier.update_yaxes(range=[0, 0.8])
fig_multiplier_sens.update_layout(
    title='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
          'Spending Multiplier of CMCO Cash Handouts on Daily Transactions (Cumulative Impact / Total Handouts)',
    yaxis_title='Spending Multiplier',
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    uniformtext=dict(mode='show', minsize=22),
    height=768,
    width=1366
)
# Export chart
fig_multiplier_sens.write_html('Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot_Sensitivity.html')
fig_multiplier_sens.write_image('Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot_Sensitivity.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_MultiplierPlot_Sensitivity.png',
           cap='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
               'Spending Multiplier of CMCO Cash Handouts on Daily Transactions (Cumulative Impact / Total Handouts)')

# Actual & counterfactual
fig_comp_sens = go.Figure()
# Add point estimate
fig_comp_sens.add_trace(
    go.Scatter(
        x=df_fcast.index.astype('str'),
        y=df_fcast[list_target_endog_level[0]],
        name='Observed',
        mode='lines',
        line=dict(color='black', width=2)
    )
)
# Add counterfactual point estimate
fig_comp_sens.add_trace(
    go.Scatter(
        x=df_fcast.index.astype('str'),
        y=df_fcast['ptcfact_' + list_target_endog_level[0]],
        name='Counterfactual (Linear)',
        mode='lines',
        line=dict(color='darkgray', width=2, dash='dash')
    )
)
fig_comp_sens.add_trace(
    go.Scatter(
        x=df_fcast_quad.index.astype('str'),
        y=df_fcast_quad['ptcfact_' + list_target_endog_level[0]],
        name='Counterfactual (Quadratic)',
        mode='lines',
        line=dict(color='crimson', width=2, dash='dash')
    )
)
fig_comp_sens.add_trace(
    go.Scatter(
        x=df_fcast_cubic.index.astype('str'),
        y=df_fcast_cubic['ptcfact_' + list_target_endog_level[0]],
        name='Counterfactual (Cubic)',
        mode='lines',
        line=dict(color='darkblue', width=2, dash='dash')
    )
)
# Layout
fig_comp_sens.update_layout(
    title='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
          'Observed and Counterfactual Daily Transactions',
    yaxis_title='Natural Logarithm; ln(100) = 4.605 = ' + str(t_ref) ,
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)

# Plot observed + counterfactual + in-sample prediction series
fig_comp_withpred_sens = go.Figure(fig_comp_sens)
# Add pre-event prediction point estimate
fig_comp_withpred_sens.add_trace(
    go.Scatter(
        x=df_pred.index.astype('str'),
        y=df_pred['ptlvlpred_' + list_target_endog_level[0]],
        name='Pre-Event Prediction (Point Est)',
        mode='lines',
        line=dict(color='gray', width=1.5, dash='dash')
    )
)
# Add pre-event prediction point estimate
fig_comp_withpred_sens.add_trace(
    go.Scatter(
        x=df_pred_quad.index.astype('str'),
        y=df_pred_quad['ptlvlpred_' + list_target_endog_level[0]],
        name='Pre-Event Prediction (Point Est)',
        mode='lines',
        line=dict(color='coral', width=1.5, dash='dash')
    )
)
# Add pre-event prediction point estimate
fig_comp_withpred_sens.add_trace(
    go.Scatter(
        x=df_pred_cubic.index.astype('str'),
        y=df_pred_cubic['ptlvlpred_' + list_target_endog_level[0]],
        name='Pre-Event Prediction (Point Est)',
        mode='lines',
        line=dict(color='lightblue', width=1.5, dash='dash')
    )
)
# Layout
fig_comp_withpred_sens.update_layout(
    title='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
          'Observed, Counterfactual, and Pre-Event Predicted Daily Transactions',
    yaxis_title='Natural Logarithm; ln(100) = 4.605 = ' + str(t_ref) ,
    plot_bgcolor='white',
    hovermode='x',
    font=dict(color='black', size=12),
    showlegend=False,
    height=768,
    width=1366
)
# Export charts
fig_comp_sens.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot_Sensitivity.html')
fig_comp_sens.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot_Sensitivity.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot_Sensitivity.png',
           cap='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
               'Observed and Counterfactual Daily Transactions (with 95% CIs)')

fig_comp_withpred_sens.write_html('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot_Sensitivity.html')
fig_comp_withpred_sens.write_image('Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot_Sensitivity.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot_Sensitivity.png',
           cap='(Sensitivity Check) Single Entity Interrupted Time Series: ' +
               'Observed, Counterfactual, and Pre-Event Predicted Daily Transactions')


# ------------- Messages on lag selection
# Describe the analysis (Main)
text_mod = 'Counterfactual is estimated with AR-X(' + str(ar_order) + \
           ') using log-difference transformation; lag order was selected using HQIC'
text_target = '\n\nTarget variable: ' + list_target_endog_level[0]
text_exog = '\nExogenous variables: ' + ', '.join(list_loglevels_exog)
full_text = text_mod + text_target + text_exog
telsendmsg(conf=tel_config,
           msg=full_text)

# Describe the analysis (Quadratic)
text_mod_quad = '(Quadratic) Counterfactual is estimated with AR-X(' + str(ar_order_quad) + \
           ') using log-difference transformation; lag order was selected using HQIC'
text_target_quad = '\n\nTarget variable: ' + list_target_endog_level[0]
text_exog_quad = '\nExogenous variables: ' + ', '.join(list_loglevels_exog)
full_text_quad = text_mod_quad + text_target_quad + text_exog_quad
telsendmsg(conf=tel_config,
           msg=full_text_quad)

# Describe the analysis (Cubic)
text_mod_cubic = '(Cubic) Counterfactual is estimated with AR-X(' + str(ar_order_cubic) + \
           ') using log-difference transformation; lag order was selected using HQIC'
text_target_cubic = '\n\nTarget variable: ' + list_target_endog_level[0]
text_exog_cubic = '\nExogenous variables: ' + ', '.join(list_loglevels_exog)
full_text_cubic = text_mod_cubic + text_target_cubic + text_exog_cubic
telsendmsg(conf=tel_config,
           msg=full_text_cubic)

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
