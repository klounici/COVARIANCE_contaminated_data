library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
init_est <- args[2]
crit <- as.double(args[3])
maxits <- as.integer(args[4])
quant <- as.double(args[5])
max_col <- as.double(args[6])

pars_list <- list(FALSE, 5, 0.15, 1e-12, TRUE)
names(pars_list) <- c("coreOnly", "numDiscrete", "fracNA",
                      "precScale", "silent")

didata <- DI(data, init_est, crit, maxits, quant, max_col, checkPars = pars_list)

wname <- sub(".csv", "", args[1])
write.csv(didata$center, paste(wname, "_res_mu.csv", sep = ""),
                        row.names = FALSE)
write.csv(didata$cov, paste(wname, "_res_S.csv", sep = ""), row.names = FALSE)