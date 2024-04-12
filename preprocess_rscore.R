setwd('/datapath')
library(devtools)
library(Seurat)
library(MAESTRO)
library(Signac)
library(reticulate)
data<-read.table("datapath/atac.csv",sep = ",",header = TRUE,row.names=1)#pbmc
use_python("/pythonpath/python", required = TRUE)
accp <- ATACCalculateGenescore(data,
                                   organism = "GRCm38",#GRCh38 GRCm38 
                                   decaydistance = 10000,
                                   model = "Enhanced")
x<-accp[which(rowSums(accp) > 0),]
write.csv(x,file="/outputpath/accpscore.csv")