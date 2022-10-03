# Bai-Perron structural break test

library(strucchange)
library(tseries)
library(forecast)

### 0 --- Preliminares
# setwd('')
file_input <- 'interim.csv'
df <- read.csv(file_input)

# T_lb = date(2020, 3, 1)
# T_ub = date(2020, 6, 30)  # RMCO phased relaxation
dates <- seq(as.Date("2020-03-01"), as.Date("2020-05-31"), by="day")
n_T <- length(dates)
spending <- ts(data=df$total_ln)
mob_retl <- ts(data=df$mob_retl_ln)
mob_groc <- ts(data=df$mob_groc_ln)

# I --- Break point test
# Spending
bp <- breakpoints(spending ~ 1)
png('Output/ImpactSingleITS_2020CMCOHandouts_StructuralBreak_Spending.png')
summary(bp)
ci <- confint(bp)
plot(spending, xaxt = 'n', main='Bai-Perron Structural Breaks: Spending')
axis(1,
     at= seq(1, length(dates), by=28),
     labels=dates[seq(1, length(dates), by=28)])
lines(bp)
lines(ci)
dev.off()

# Retail mobility
bp <- breakpoints(mob_retl ~ 1)
png('Output/ImpactSingleITS_2020CMCOHandouts_StructuralBreak_MobRetl.png')
summary(bp)
ci <- confint(bp)
plot(mob_retl, xaxt = 'n', main='Bai-Perron Structural Breaks: Mobility (Retail & Recreation)')
axis(1,
     at= seq(1, length(dates), by=28),
     labels=dates[seq(1, length(dates), by=28)])
lines(bp)
lines(ci)
dev.off()

# Grocery mobility
bp <- breakpoints(mob_groc ~ 1)
png('Output/ImpactSingleITS_2020CMCOHandouts_StructuralBreak_MobGroc.png')
summary(bp)
ci <- confint(bp)
plot(mob_groc, xaxt = 'n', main='Bai-Perron Structural Breaks: Mobility (Grocery & Pharmacy)')
axis(1,
     at= seq(1, length(dates), by=28),
     labels=dates[seq(1, length(dates), by=28)])
lines(bp)
lines(ci)
dev.off()
