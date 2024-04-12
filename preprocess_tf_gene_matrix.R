library('BITFAM')
setwd("datapath")
getwd()

data<-read.table("rna.csv",sep = " ")#matrix
name<-read.table("name.csv",sep = ",",header = TRUE)#gene
cellname<-read.table("cell.csv",sep = ",",header = TRUE)#barcode

r<-name['name']
c<-cellname['barcodes']
data<-as.matrix(data)
dimnames(data)[1]<-r
colnames(data)<-r[["name"]]
rownames(data)<-c[["barcodes"]]
install.packages("rstan", type = "source")

TF_targets_dir<-"human/"#
All_TFs <-system.file("extdata", paste0(TF_targets_dir, "all_TFs.txt"), package = "BITFAM")
All_TFs <- read.table(All_TFs, stringsAsFactors = F)$V1

data<-t(data)
TF_used <- rownames(data)[rownames(data) %in% All_TFs]#select TFs in dataset
rownames(data) <- toupper(rownames(data))

gene_list <- list()

for(i in TF_used){
  TF_targets_path <-system.file("extdata", paste0(TF_targets_dir, i), package = "BITFAM")
  tmp_gene <- read.table(TF_targets_path, stringsAsFactors = F)
  tmp_gene <- toupper(tmp_gene$V1)
  gene_list[[which(TF_used == i)]] <- rownames(data)[rownames(data) %in% tmp_gene]
}

TF_used <- TF_used[ unlist(lapply(gene_list, length)) > 10]

gene_list <- list()
for(i in TF_used){
  TF_targets_path <-system.file("extdata", paste0(TF_targets_dir, i), package = "BITFAM")
  tmp_gene <- read.table(TF_targets_path, stringsAsFactors = F)
  tmp_gene <- toupper(tmp_gene$V1)
  gene_list[[which(TF_used == i)]] <- rownames(data)[rownames(data) %in% tmp_gene]
}
X <- t(data)
chipseq_weight <- matrix(1, nrow = length(colnames(X)), ncol = length(TF_used))
for(i in 1:length(TF_used)){
  chipseq_weight[, i] <- ifelse(colnames(X) %in% gene_list[[i]], 1, 0)
}

write.csv(chipseq_weight,file="tf_gene.csv")#TF-gene matrix
write.csv(TF_used,file="TF_used.csv")#TF list