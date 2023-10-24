library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
maxCol = strtoi(args[2])/100

ecov = DDCWcov(data, maxCol)

wname = sub('.csv', '', args[1])

res = c(ecov$mu, ecov$cov)
write.csv(isOutlier, paste(wname, '_res.csv', sep=''))
