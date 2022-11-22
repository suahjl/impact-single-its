import telegram_send
from datetime import date, timedelta
import time
from PIL import Image

import os

time_start = time.time()

# 0 --- Main settings
tel_config = 'EcMetrics_Config_GeneralFlow.conf'  # EcMetrics_Config_GeneralFlow EcMetrics_Config_RMU

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


def pil_img2pdf(list_images, extension='png', img_path='Output/', pdf_name='ImpactSingleITS_AllCharts'):
    seq = list_images.copy()  # deep copy
    list_img = []
    file_pdf = img_path + pdf_name + '.pdf'
    run = 0
    for i in seq:
        img = Image.open(img_path + i + '.' + extension)
        img = img.convert('RGB')  # PIL cannot save RGBA files as pdf
        if run == 0:
            first_img = img.copy()
        elif run > 0:
            list_img = list_img + [img]
        run += 1
    first_img.save(file_pdf,
                   'PDF',
                   resolution=100.0,
                   save_all=True,
                   append_images=list_img)

# II --- Compile output files into a single pdf
# 2020CMCO
seq_output = ['ImpactSingleITS_2020CMCOHandouts_TSPlots_Level',
              'ImpactSingleITS_2020CMCOHandouts_TSPlots_LogDiff',
              'ImpactSingleITS_2020CMCOHandouts_URoot',
              'ImpactSingleITS_2021BPR3Handouts_ObsCFactPlot',
              'ImpactSingleITS_2021BPR3Handouts_ObsCFactPredPlot',
              'ImpactSingleITS_2020CMCOHandouts_ImpactPlot',
              'ImpactSingleITS_2020CMCOHandouts_CumulMYRImpactPlot',
              'ImpactSingleITS_2020CMCOHandouts_MultiplierPlot',
              'ImpactSingleITS_2020CMCOHandouts_StructuralBreak_Spending',
              'ImpactSingleITS_2020CMCOHandouts_StructuralBreak_MobRetl',
              'ImpactSingleITS_2020CMCOHandouts_StructuralBreak_MobGroc']
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='ImpactSingleITS_2020CMCOHandouts_AllCharts')
telsendfiles(conf=tel_config,
             path='Output/ImpactSingleITS_2020CMCOHandouts_AllCharts.pdf',
             cap='All charts from the single entity ITS impact estimates project (2020 CMCO)')

# 2020BSH3
seq_output = ['ImpactSingleITS_2020BSH3Handouts_TSPlots_Level',
              'ImpactSingleITS_2020BSH3Handouts_TSPlots_LogDiff',
              'ImpactSingleITS_2020BSH3Handouts_URoot',
              'ImpactSingleITS_2020BSH3Handouts_ObsCFactPlot',
              'ImpactSingleITS_2020BSH3Handouts_ObsCFactPredPlot',
              'ImpactSingleITS_2020BSH3Handouts_ImpactPlot',
              'ImpactSingleITS_2020BSH3Handouts_CumulMYRImpactPlot',
              'ImpactSingleITS_2020BSH3Handouts_MultiplierPlot',
              'ImpactSingleITS_2020BSH3Handouts_StructuralBreak_Spending',
              'ImpactSingleITS_2020BSH3Handouts_StructuralBreak_MobRetl',
              'ImpactSingleITS_2020BSH3Handouts_StructuralBreak_MobGroc']
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='ImpactSingleITS_2020BSH3Handouts_AllCharts')
telsendfiles(conf=tel_config,
             path='Output/ImpactSingleITS_2020BSH3Handouts_AllCharts.pdf',
             cap='All charts from the single entity ITS impact estimates project (2020 BSH3)')

# 2021BPR3
seq_output = ['ImpactSingleITS_2021BPR3Handouts_TSPlots_Level',
              'ImpactSingleITS_2021BPR3Handouts_TSPlots_LogDiff',
              'ImpactSingleITS_2021BPR3Handouts_URoot',
              'ImpactSingleITS_2021BPR3Handouts_ObsCFactPlot',
              'ImpactSingleITS_2021BPR3Handouts_ObsCFactPredPlot',
              'ImpactSingleITS_2021BPR3Handouts_ImpactPlot',
              'ImpactSingleITS_2021BPR3Handouts_CumulMYRImpactPlot',
              'ImpactSingleITS_2021BPR3Handouts_MultiplierPlot',
              'ImpactSingleITS_2021BPR3Handouts_StructuralBreak_Spending',
              'ImpactSingleITS_2021BPR3Handouts_StructuralBreak_MobRetl',
              'ImpactSingleITS_2021BPR3Handouts_StructuralBreak_MobGroc']
pil_img2pdf(list_images=seq_output,
            img_path='Output/',
            extension='png',
            pdf_name='ImpactSingleITS_2021BPR3Handouts_AllCharts')
telsendfiles(conf=tel_config,
             path='Output/ImpactSingleITS_2021BPR3Handouts_AllCharts.pdf',
             cap='All charts from the single entity ITS impact estimates project (2021 BPR3)')


# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
