library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
quant <- as.double(args[2])
return_res <- as.logical(args[3])

if (ncol(data) < 250) {
    pars_list <- list(1., 0, 0., "automatic", 0.99, 0.5, "wmean", FALSE, TRUE, 25000, FALSE, "1stepM", "gkwls", "wrap", 100)
    names(pars_list) <- c("fracNA", "numDiscrete", "precScale", "cleanNAfirst", "tolProb","corrlim", "combineRule", "returnBigXimp", "silent", "nLocScale", "fastDDC", "standType", "corrType", "transFun", "nbngbrs")
}else {
    pars_list <- list(1., 0, 0., "automatic", 0.99, 0.5, "wmean", FALSE, TRUE, 25000, TRUE, "1stepM", "gkwls", "wrap", 100)
    names(pars_list) <- c("fracNA", "numDiscrete", "precScale", "cleanNAfirst", "tolProb","corrlim", "combineRule", "returnBigXimp", "silent", "nLocScale", "fastDDC", "standType", "corrType", "transFun", "nbngbrs")
}
ddc_data <- DDC(data, DDCpars = pars_list)

wname <- sub(".csv", "", args[1])

if (return_res) {
    write.csv(ddc_data, paste(wname, "_res.csv", sep = ""), row.names = FALSE)
} else {
    # Outlier test using chi-squared quantiles
    is_outlier <- ddc_data$stdResid > sqrt(qchisq(quant, 1))
    write.csv(is_outlier, paste(wname, "_res.csv", sep = ""), row.names = FALSE)
}

