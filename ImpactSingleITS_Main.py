# Executes all scripts (useful for refreshing estimates)

import telegram_send
import time

time_start = time.time()

# 0 --- Main settings
tel_config = 'test.conf'  # add own telegram bot channel config

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


# II --- Execute scripts
import ImpactSingleITS_2020CMCOHandouts_TSPlots
import ImpactSingleITS_2020CMCOHandouts
import ImpactSingleITS_2020CMCOHandouts_StructuralBreak
time.sleep(30)  # to prevent flooding

import ImpactSingleITS_2020BSH3Handouts_TSPlots
import ImpactSingleITS_2020BSH3Handouts
import ImpactSingleITS_2020BSH3Handouts_StructuralBreak
time.sleep(30)  # to prevent flooding

import ImpactSingleITS_2021BPR3Handouts_TSPlots
import ImpactSingleITS_2021BPR3Handouts
import ImpactSingleITS_2021BPR3Handouts_StructuralBreak
time.sleep(30)  # to prevent flooding

import ImpactSingleITS_MultiplierBarChart
import ImpactSingleITS_CompileOutput

# End
runtime_message = '\n----- All scripts in ImpactSingleITS ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----'
print(runtime_message)
telsendmsg(conf=tel_config,
           msg=runtime_message)
