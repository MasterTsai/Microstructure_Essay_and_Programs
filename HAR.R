library("timeSeries")
library("xts")
library("highfrequency")
library("chron")
library("zoo")

p.csv = read.csv("D:\\HNU\\Market Microstructure\\hyjt.csv")
p.zoo = read.zoo("D:\\HNU\\Market Microstructure\\hyjt.csv",header=TRUE,sep=',',index.column=2,FUN=as.chron,format='%Y/%m/%d %H:%M')
p = as.xts(p.zoo)


p = p*100
p_rt = makeReturns(p$close)
p_rv = rBPCov(p$close,makeReturns=T)

HARRVa <- harModel(data=q5,periods=c(1,5,20),Rvest=c("rCov"),type="HARRV",h=1)
HARRVa_sqrt <- harModel(data=q5,periods=c(1,5,20),Rvest=c("rCov"),type="HARRV",transform="sqrt",h=1)
HARRVa_log <- harModel(data=q5,periods = c(1,5,20), Rvest=c("rCov"),type="HARRV",transform="log",h=1)
summary(HARRVa)
summary(HARRVa_sqrt)
summary(HARRVa_log)

MSE.HARRVa = mean(HARRVa$residuals*HARRVa$residuals)
MSE.HARRVa_sqrt = mean(HARRVa_sqrt$residuals*HARRVa_sqrt$residuals)
MSE.HARRVa_log = mean(HARRVa_log$residuals*HARRVa_log$residuals)