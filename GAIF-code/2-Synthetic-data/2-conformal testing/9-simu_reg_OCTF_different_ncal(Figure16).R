####### Simulation for online conformal selection with feedback



library(MASS)
library(ks)
library(kernlab)
library(randomForest)
library(foreach)
library(reshape2)
library(ggplot2)
library(glmnet)
library(caret)
library(nnet)
library(ggpubr)
library(tidyverse)
library(kedd)
library(pbmcapply)
library(onlineFDR)
library(parallel)
library(magrittr)
library(latex2exp)
library(ggsci)
library(lsa)
library(doSNOW)
library(doParallel)
library(dplyr)
library(purrr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("functions_OnSel.R")
source("algoclass_OnSel.R")
source("SAFFRON_feedback functions.R")









######### RF -----------

T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)
ncal_seq <- c(200,400,600)

algo<- new("RF") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm

########## without conformity score selection ----------

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_RF <- foreach(iter = 1:nr, .combine = "rbind",
                     .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                   "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                   "tidyverse","lsa","magrittr","doSNOW","pbmcapply"), 
                     .errorhandling = "remove", 
                     .options.snow = opts)%dopar% {
                       info<-data.frame()
                       
                       for(n_cal in ncal_seq){
                         
                         # generate data
                         data <- data_generation_regression2(N=T)
                         Value=list(type="<=A",v=quantile(data$y,0.5))
                         his_data <- data_generation_regression2(N=n)
                         
                         p <- ncol(his_data)-1 # dimension of covariates
                         
                         
                         
                         ### some data notations, and index for null data-----
                         datawork=DataSplit(his_data,n,0,n_cal)
                         data_train=datawork$data_train
                         
                         data_cal=datawork$data_cal
                         
                         
                         Null_cal=NullIndex(data_cal$y,Value)
                         
                         
                         X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                         Y_train=as.matrix(data_train$y)
                         X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                         Y_cal=as.matrix(data_cal$y)
                         
                         
                         
                         data_test=data
                         Null_test=NullIndex(data_test$y,Value)
                         Alter_test=setdiff(1:length(data_test$y),Null_test)
                         X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                         Y_test=as.matrix(data_test$y)
                         
                         
                         
                         #theta <- data_test$y
                         theta <- rep(1,T)
                         theta[Null_test]=0
                         
                         # model and estimating locfdr -----
                         
                         model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
                         W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
                         W_test=Pred(algo,model,X_test) #predict classfication score of test data
                         
                         #calculate p-values
                         
                         pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
                         
                         
                         
                         # pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)
                         
                         
                         rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                         res <- CiterionCompute(rej_feedback_LORD$R,theta,"LF")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                         res <- CiterionCompute(rej_feedback_LORD_conse$R,theta,"LFS")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_LORD=LORD(pval,alpha,version = "++")
                         res <- CiterionCompute(rej_LORD$R,theta,"LORD++")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                         res <- CiterionCompute(rej_SAFFRON$R,theta,"SAFFRON")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                         res <- CiterionCompute(rej_feedback_SA$R,theta,"SF")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                         res <- CiterionCompute(rej_feedback_SA_conse$R,theta,"SFS")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         rej_LOND=LOND(pval,alpha)
                         res=CiterionCompute(rej_LOND$R,theta,"LOND")
                         res["size"]=n_cal
                         res["Alg"]="RF"
                         info=rbind(info,res)
                         
                         
                         
                         
                       }
                       
                       
                       return(info)
                     }
close(pb)
stopCluster(cl)

Result <- Result_RF
Result$ndraw <- factor(Result$size, levels = c('200', '400', '600'))
head(Result_RF,30)
Result$Method=factor(Result$Method,levels = c("SF","LF","SFS","LFS","SAFFRON",
                                              "LORD++","LOND"))
Resultdraw <- data.frame(Value = c(Result$FDP, Result$Power), 
                         Method = c(Result$Method, Result$Method), 
                         ndraw = c(Result$ndraw, Result$ndraw), 
                         Type = c(rep('FDP', nrow(Result)), rep('Power', nrow(Result))), 
                         hline = c(rep(alpha, nrow(Result)), rep(NA, nrow(Result))))
head(Result)

P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = TRUE) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1


############ SVM ------------

T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)
ncal_seq <- c(200,400,600)

algo<- new("SVM-R") #algorithm used for classification or regression
lambda<- 1 #specific parameter for the algorithm
########## without conformity score selection ----------

nr=500
cl = makeCluster(15)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_SVM <- foreach(iter = 1:nr, .combine = "rbind",
                      .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                    "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                    "tidyverse","lsa","magrittr","doSNOW","pbmcapply"), 
                      .errorhandling = "remove", 
                      .options.snow = opts)%dopar% {
                        info<-data.frame()
                        
                        for(n_cal in ncal_seq){
                          
                          # generate data
                          data <- data_generation_regression2(N=T)
                          Value=list(type="<=A",v=quantile(data$y,0.5))
                          his_data <- data_generation_regression2(N=n)
                          
                          p <- ncol(his_data)-1 # dimension of covariates
                          
                          
                          
                          ### some data notations, and index for null data-----
                          datawork=DataSplit(his_data,n,0,n_cal)
                          data_train=datawork$data_train
                          
                          data_cal=datawork$data_cal
                          
                          
                          Null_cal=NullIndex(data_cal$y,Value)
                          
                          
                          X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                          Y_train=as.matrix(data_train$y)
                          X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                          Y_cal=as.matrix(data_cal$y)
                          
                          
                          
                          data_test=data
                          Null_test=NullIndex(data_test$y,Value)
                          Alter_test=setdiff(1:length(data_test$y),Null_test)
                          X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                          Y_test=as.matrix(data_test$y)
                          
                          
                          
                          #theta <- data_test$y
                          theta <- rep(1,T)
                          theta[Null_test]=0
                          
                          # model and estimating locfdr -----
                          
                          model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
                          W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
                          W_test=Pred(algo,model,X_test) #predict classfication score of test data
                          
                          #calculate p-values
                          
                          pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
                          
                          
                          
                          # pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)
                          
                          
                          rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                          res <- CiterionCompute(rej_feedback_LORD$R,theta,"LF")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                          res <- CiterionCompute(rej_feedback_LORD_conse$R,theta,"LFS")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_LORD=LORD(pval,alpha,version = "++")
                          res <- CiterionCompute(rej_LORD$R,theta,"LORD++")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                          res <- CiterionCompute(rej_SAFFRON$R,theta,"SAFFRON")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                          res <- CiterionCompute(rej_feedback_SA$R,theta,"SF")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                          res <- CiterionCompute(rej_feedback_SA_conse$R,theta,"SFS")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          rej_LOND=LOND(pval,alpha)
                          res=CiterionCompute(rej_LOND$R,theta,"LOND")
                          res["size"]=n_cal
                          res["Alg"]="SVM"
                          info=rbind(info,res)
                          
                          
                          
                          
                        }
                        
                        
                        return(info)
                      }
close(pb)
stopCluster(cl)

Result <- Result_SVM
Result$ndraw <- factor(Result$size, levels = c('200', '400', '600'))
Result$Method=factor(Result$Method,levels = c("SF","LF","SFS","LFS","SAFFRON",
                                              "LORD++","LOND"))
Resultdraw <- data.frame(Value = c(Result$FDP, Result$Power), 
                         Method = c(Result$Method, Result$Method), 
                         ndraw = c(Result$ndraw, Result$ndraw), 
                         Type = c(rep('FDP', nrow(Result)), rep('Power', nrow(Result))), 
                         hline = c(rep(alpha, nrow(Result)), rep(NA, nrow(Result))))
head(Result)

P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = TRUE) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1



############### NN ---------------

T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)
ncal_seq <- c(200,400,600)

algo<- new("NN-R") #algorithm used for classification or regression


########## without conformity score selection ----------

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_NN <- foreach(iter = 1:nr, .combine = "rbind",
                     .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                   "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                   "tidyverse","lsa","magrittr","doSNOW","pbmcapply"), 
                     .errorhandling = "remove", 
                     .options.snow = opts)%dopar% {
                       info<-data.frame()
                       
                       for(n_cal in ncal_seq){
                         
                         # generate data
                         data <- data_generation_regression2(N=T)
                         Value=list(type="<=A",v=quantile(data$y,0.5))
                         his_data <- data_generation_regression2(N=n)
                         
                         p <- ncol(his_data)-1 # dimension of covariates
                         
                         
                         
                         ### some data notations, and index for null data-----
                         datawork=DataSplit(his_data,n,0,n_cal)
                         data_train=datawork$data_train
                         
                         data_cal=datawork$data_cal
                         
                         
                         Null_cal=NullIndex(data_cal$y,Value)
                         
                         
                         X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                         Y_train=as.matrix(data_train$y)
                         X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                         Y_cal=as.matrix(data_cal$y)
                         
                         
                         
                         data_test=data
                         Null_test=NullIndex(data_test$y,Value)
                         Alter_test=setdiff(1:length(data_test$y),Null_test)
                         X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                         Y_test=as.matrix(data_test$y)
                         
                         
                         
                         #theta <- data_test$y
                         theta <- rep(1,T)
                         theta[Null_test]=0
                         
                         # model and estimating locfdr -----
                         
                         model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
                         W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
                         W_test=Pred(algo,model,X_test) #predict classfication score of test data
                         
                         #calculate p-values
                         
                         pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
                         
                         
                         
                         # pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)
                         
                         
                         rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                         res <- CiterionCompute(rej_feedback_LORD$R,theta,"LF")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                         res <- CiterionCompute(rej_feedback_LORD_conse$R,theta,"LFS")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_LORD=LORD(pval,alpha,version = "++")
                         res <- CiterionCompute(rej_LORD$R,theta,"LORD++")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                         res <- CiterionCompute(rej_SAFFRON$R,theta,"SAFFRON")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                         res <- CiterionCompute(rej_feedback_SA$R,theta,"SF")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                         res <- CiterionCompute(rej_feedback_SA_conse$R,theta,"SFS")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         rej_LOND=LOND(pval,alpha)
                         res=CiterionCompute(rej_LOND$R,theta,"LOND")
                         res["size"]=n_cal
                         res["Alg"]="NN"
                         info=rbind(info,res)
                         
                         
                         
                         
                       }
                       
                       
                       return(info)
                     }
close(pb)
stopCluster(cl)

Result <- Result_NN
Result$ndraw <- factor(Result$size, levels = c('200', '400', '600'))
Result$Method=factor(Result$Method,levels = c("SF","LF","SFS","LFS","SAFFRON",
                                              "LORD++","LOND"))
Resultdraw <- data.frame(Value = c(Result$FDP, Result$Power), 
                         Method = c(Result$Method, Result$Method), 
                         ndraw = c(Result$ndraw, Result$ndraw), 
                         Type = c(rep('FDP', nrow(Result)), rep('Power', nrow(Result))), 
                         hline = c(rep(alpha, nrow(Result)), rep(NA, nrow(Result))))
head(Result)

P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = TRUE) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1


########### plots ---------------

Result_all <- data.frame(rbind(Result_RF,Result_SVM,Result_NN))
head(Result_all)
write.csv(Result_all,"OCTF_ScenarioB_n_500times.csv")

Result_all$Method <- factor(Result_all$Method, levels = c("SF","LF","SFS","LFS","SAFFRON",
                                                          "LORD++","LOND"))
Result_all$Alg <- factor(Result_all$Alg, levels = c('RF', 'SVM','NN'))
Result_all$ndraw <- factor(Result_all$size, levels = c('200', '400', '600'))

alpha <- 0.2

Result2.new <- Result_all

Result2.tidy <- data.frame(quant = c(Result2.new$FDP, Result2.new$Power), 
                           Method = c(Result2.new$Method, Result2.new$Method), 
                           ndraw = c(Result2.new$ndraw, Result2.new$ndraw), 
                           type = c(rep('FDR', nrow(Result2.new)), 
                                    rep('Power', nrow(Result2.new))), 
                           hline = c(rep(alpha, nrow(Result2.new)), rep(NA, nrow(Result2.new))),
                           Alg = c(Result2.new$Alg,Result2.new$Alg))


Result2.tidy$Method <- factor(Result_all$Method,levels = c("SF","LF","SFS","LFS","SAFFRON",
                                                           "LORD++","LOND"))

head(Result2.tidy)
#Result2.tidy.filtered <- subset(Result2.tidy, Method != "LOND")
Result2.tidy.filtered <- Result2.tidy


pdf(file="OCTF_ScenarioB_n.pdf",
    width=12,height=7)
P2 <- ggplot(Result2.tidy.filtered, aes(x = ndraw, y = quant, color=Method)) +
  geom_boxplot(alpha=0.7) +
  scale_x_discrete(name = "n") +
  ylab("") +
  geom_hline(aes(yintercept = hline), colour = "black", na.rm = TRUE,linetype="dashed") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
  ))+
  facet_grid(type~Alg, scales = "free_y")+
  geom_hline(aes(yintercept = hline), colour = "black", na.rm = TRUE,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") +
  facet_grid(type~Alg,scales = "free_y")
P2
dev.off()




