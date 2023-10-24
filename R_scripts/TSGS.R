library(MASS)
library(Rcpp)
library(GSE)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
filter = args[2]
partial_inpute = args[3]
tol = as.double(args[4])
maxiter = strtoi(args[5])
method = args[6]
init = args[7]
params = c(data, filter, partial_inpute, tol, maxiter,method, init)
if (length(args)< 8){
  TSGSdata = TSGS(data, filter, partial_inpute, tol, maxiter, method, init)
}
if (length(args) == 8){
  mu0 = as.double(args[8])
  params.append(mu0)
  
  TSGSdata = TSGS(data, filter, partial_inpute, tol, maxiter, method, init, mu0)
}
if (length(args) > 8){
  mu0 = as.double(args[8])
  params.append(mu0)
  S0 = as.matrix(args[9])
  params.append(S0)
  
  TSGSdata = TSGS(data, filter, partial_inpute, tol, maxiter, method, init, mu0, S0)
}
wname = sub('.csv', '', args[1])
write.csv(TSGSdata@mu, paste(wname, '_res_mu.csv', sep=''), row.names=FALSE)
write.csv(TSGSdata@S, paste(wname, '_res_S.csv', sep=''), row.names=FALSE)
write.csv(TSGSdata@xf, paste(wname, '_res_filtered.csv', sep=''), row.names=FALSE)