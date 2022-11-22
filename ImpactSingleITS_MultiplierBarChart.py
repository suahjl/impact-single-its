import pandas as pd
import numpy as np
import plotly.graph_objects as go
import telegram_send
import time
from tqdm import tqdm

time_start = time.time()

# 0 --- Main settings
tel_config = 'EcMetrics_Config_GeneralFlow.conf'  # EcMetrics_Config_GeneralFlow # EcMetrics_Config_RMU

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


# II --- Set up data frame
episodes = ['2020CMCOHandouts', '2020BSH3Handouts', '2021BPR3Handouts']
episodes_nice = ['CMCO (May 2020)', 'BSH 3 (Jul 2020)', 'BPR 3 (Sep 2021)']
k = 1
for e, enice in zip(episodes, episodes_nice):
    if k == 1:
        df = pd.read_csv('multiplier_' + e + '.txt', sep='|')
        df['Episode'] = enice
    elif k > 1:
        d = pd.read_csv('multiplier_' + e + '.txt', sep='|')
        d['Episode'] = enice
        df = pd.concat([df, d], axis=0)  # top-bottom concat
    k += 1
df_cumul = df[['Episode', 'pt_cumulimpact', 'lb_cumulimpact', 'ub_cumulimpact']]
df = df[['Episode', 'pt_multiplier', 'lb_multiplier', 'ub_multiplier']]
# Export data frame
df_cumul.to_csv('Output/ImpactSingleITS_ConsolCumulImpactTable.csv', index=False)
df.to_csv('Output/ImpactSingleITS_ConsolMultiplierTable.csv', index=False)

# III --- Chart
fig = go.Figure()
# Add estimates
fig.add_trace(
    go.Bar(
        y=df['pt_multiplier'],
        x=df['Episode'],
        name='Point Estimate',
        error_y=dict(
            type='data',
            symmetric=False,
            array=(df['ub_multiplier'] - df['pt_multiplier']).reset_index(drop=True),
            arrayminus=(df['pt_multiplier'] - df['lb_multiplier']).reset_index(drop=True)
        ),
        marker=dict(color='lightgray'),
        width=0.4,
        text=df['pt_multiplier'],
        textposition='outside'
    )
)
# Layout
fig.update_layout(
    title='Single Entity Interrupted Time Series: ' +
          'Spending Multipliers on Daily Transactions (Cumulative Impact / Total Handouts)',
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
fig.write_html('Output/ImpactSingleITS_ConsolMultiplierPlot.html')
fig.write_image('Output/ImpactSingleITS_ConsolMultiplierPlot.png')
telsendimg(conf=tel_config,
           path='Output/ImpactSingleITS_ConsolMultiplierPlot.png',
           cap='[CONSOLIDATED] Single Entity Interrupted Time Series: ' +
               'Spending Multiplier on Daily Transactions (Cumulative Impact / Total Handouts)')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')