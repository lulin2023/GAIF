

########## diabetes ----------

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

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("functions_OnSel.R")
source("algoclass_OnSel.R")
source("SAFFRON_feedback functions.R")



# import the data----

data1 <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

head(data1)
colnames(data1)
dim(data1)

table(data1$Diabetes_binary)
data <- data1

data$y <- data1$Diabetes_binary
head(data)
data <- data[,-1]
table(data$y)/nrow(data)

# simulation setting-----
alpha <- 0.2 # significance level


n <- 2000 # number of historical data
N <- 2000 # number of total time points
n_train<- round(n/2) # number of data used for training model
n_cal<- n-n_train #number of data used for estimating locfdr
algo<- new("RFc") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm

alpha_seq <- c(0.15,0.2,0.25,0.3)
colnames(data)
# draw n history sample 
id <- c(1:nrow(data))
his_id <- sample(id,n)
his_data <- data[his_id,]
id1 <- id[-his_id]

id2 <- sample(id1,N)
data_test <- data[id2,]
p <- ncol(data)-1 # dimension of covariates
# Random forest--------

# confirm H0, 6 choices among classification and regression settings-----

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))


### some data notations, and index for null data-----
datawork=DataSplit(his_data,n,0,n_cal)
data_train=datawork$data_train

data_cal=datawork$data_cal
data_rest=datawork$data_rest

Null_cal=NullIndex(data_cal$y,Value)
Null_rest=NullIndex(data_rest$y,Value)

X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
Y_train=as.matrix(data_train$y)
X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
Y_cal=as.matrix(data_cal$y)

X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
Y_rest=as.matrix(data_rest$y)

Null_test=NullIndex(data_test$y,Value)
Alter_test=setdiff(1:length(data_test$y),Null_test)
X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
Y_test=as.matrix(data_test$y)


# model and estimating locfdr -----
colnames(X_train)
model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
W_test=Pred(algo,model,X_test) #predict classfication score of test data


colnames(data)



nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR","randomForest","caret","nnet","glmnet","reshape2","kedd","kernlab"),
                  .errorhandling = "remove", .options.snow = opts)%dopar% {
                    info<-data.frame()
                    
                    
                    
                    # draw n history sample as his_data and draw N sample as data
                    id <- c(1:nrow(data))
                    his_id <- sample(id,n)
                    his_data <- data[his_id,]
                    id1 <- id[-his_id]
                    
                    id2 <- sample(id1,N)
                    data_test <- data[id2,]
                    p <- ncol(data)-1 
                    
                    ### some data notations, and index for null data-----
                    datawork=DataSplit(his_data,n,0,n_cal)
                    data_train=datawork$data_train
                    
                    data_cal=datawork$data_cal
                    data_rest=datawork$data_rest
                    
                    Null_cal=NullIndex(data_cal$y,Value)
                    Null_rest=NullIndex(data_rest$y,Value)
                    
                    X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                    Y_train=as.matrix(data_train$y)
                    X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                    Y_cal=as.matrix(data_cal$y)
                    
                    X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
                    Y_rest=as.matrix(data_rest$y)
                    
                    Null_test=NullIndex(data_test$y,Value)
                    Alter_test=setdiff(1:length(data_test$y),Null_test)
                    X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                    Y_test=as.matrix(data_test$y)
                    
                    # X_test_scale=scale(X_test,center = TRUE,scale = TRUE)
                    # X_test_scale[which(is.na(X_test_scale))]=0
                    # X_test_scale[,5:11]=X_test_scale[,5:11]/7
                    # X_test_scale[,12:13]=X_test_scale[,12:13]/2
                    # X_test_scale[,14:17]=X_test_scale[,14:17]/4
                    # X_test_scale[,18:19]=X_test_scale[,18:19]/2
                    # X_test_scale[,20:24]=X_test_scale[,20:24]/5
                    # X_test_scale[,25:28]=X_test_scale[,25:28]/4
                    # X_test_scale[,]
                    # model and estimating locfdr -----
                    
                    
                    model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
                    W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
                    W_test=Pred(algo,model,X_test) #predict classfication score of test data
                    
                    #calculate p-values
                    
                    T <- nrow(data_test)
                    theta <- rep(1,T)
                    theta[Null_test]=0
                    
                    pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
                    
                    
                    for (alpha in alpha_seq) {
                      rej_SAFFRON=SAFFRON(pval,alpha)
                      res=CiterionCompute(rej_SAFFRON$R,theta,"SAFFRON")
                      res["Alpha"]=alpha
                      info=rbind(info,res)
                      
                      
                      rej_LORD=LORD(pval,alpha)
                      res=CiterionCompute(rej_LORD$R,theta,"LORD++")
                      res["Alpha"]=alpha
                      info=rbind(info,res)
                      
                      # rej_ADDIS=ADDIS(data$p,alpha)
                      # res=CiterionCompute(rej_ADDIS$R,data$theta,"ADDIS")
                      # res["Signal"]=signal
                      # info=rbind(info,res)
                      
                      rej_LOND=LOND(pval,alpha)
                      res=CiterionCompute(rej_LOND$R,theta,"LOND")
                      res["Alpha"]=alpha
                      info=rbind(info,res)
                      
                      
                      rej_feedback=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                      res=CiterionCompute(rej_feedback$R,theta,"OCTF")
                      res["Alpha"]=alpha
                      info=rbind(info,res)
                      
                    }
                    
                    
                    return(info)
                  }
close(pb)
stopCluster(cl)

# sum(rej_feedback$R[1:100])
# sum(rej_feedback$R[7900:8000])
# sum(theta)
# table(theta)


pp <- result%>%
  group_by(Method,Alpha)%>%
  dplyr::summarize(FDR=mean(FDP),POWER=mean(Power),sdFDP=sd(FDP),sdPower=sd(Power))
pp

pp$Method=factor(pp$Method,levels = c("OCTF","SAFFRON","LORD++","LOND"))


P1 <- ggplot(data = pp,aes(x = Alpha, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab(TeX("$alpha$"))+
  ylab("FDR")+
  ylim(0,0.3)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF","#20854EFF"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 
# P1 <- P1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P1

P1 <- P1+ geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", size = 1)


P2 <- ggplot(data = pp,aes(x = Alpha, y = POWER, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab(TeX("$alpha$"))+
  ylab("Power")+
  ylim(0,0.4)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF","#20854EFF"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

P2


Diabetes <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                       font.label = list(size = 20, face = "bold"))

pdf(file = "Diabetes.pdf",width = 10,height = 4) 
Diabetes
dev.off()


################### model optimization --------------
### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))
algo1<- new("SVM") #algorithm used for classification or regression
lambda1<- 1 #specific parameter for the algorithm
algo2<- new("RFc") #algorithm used for classification or regression
lambda2<- 500 #specific parameter for the algorithm
algo3<- new("NN") #algorithm used for classification or regression




##################  compare with random score selection  -------------------

# simulation setting-----
alpha <- 0.3 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 2000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)




nr=200
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                "tidyverse","lsa","magrittr","doSNOW","pbmcapply"), 
                  .errorhandling = "remove", 
                  .options.snow = opts)%dopar% {
                    
                    
                    # draw n history sample as his_data and draw N sample as data
                    id <- c(1:nrow(data))
                    his_id <- sample(id,n)
                    his_data <- data[his_id,]
                    id1 <- id[-his_id]
                    
                    id2 <- sample(id1,N)
                    data_test <- data[id2,]
                    p <- ncol(data)-1 
                    
                    ### some data notations, and index for null data-----
                    datawork=DataSplit(his_data,n,0,n_cal)
                    data_train=datawork$data_train
                    
                    data_cal=datawork$data_cal
                    data_rest=datawork$data_rest
                    
                    Null_cal=NullIndex(data_cal$y,Value)
                    Null_rest=NullIndex(data_rest$y,Value)
                    
                    X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                    Y_train=as.matrix(data_train$y)
                    X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                    Y_cal=as.matrix(data_cal$y)
                    
                    X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
                    Y_rest=as.matrix(data_rest$y)
                    
                    Null_test=NullIndex(data_test$y,Value)
                    Alter_test=setdiff(1:length(data_test$y),Null_test)
                    X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                    Y_test=as.matrix(data_test$y)
                    
                    
                    
                    
                    # training K = 3 models and calculate conformal p-values -----
                    
                    ## Model1: SVM
                    
                    
                    model1=fitting(algo1,X_train,Y_train,lambda1) #estimate model by training data
                    W_cal1=Pred(algo1,model1,X_cal)[,2] #predict classfication score of calibration data
                    W_test1=Pred(algo1,model1,X_test)[,2] #predict classfication score of test data
                    # pval1 <- confomalPvalue_online(W_cal1,W_test1,Null_cal,Value,theta)
                    
                    ## Model2: RF
                    
                    
                    model2=fitting(algo2,X_train,Y_train,lambda2) #estimate model by training data
                    W_cal2=Pred(algo2,model2,X_cal) #predict classfication score of calibration data
                    W_test2=Pred(algo2,model2,X_test) #predict classfication score of test data
                    # pval2 <- confomalPvalue_online(W_cal2,W_test2,Null_cal,Value,theta)
                    
                    ## Model3: NN
                    
                    
                    model3=fitting(algo3,X_train,Y_train,lambda) #estimate model by training data
                    W_cal3=Pred(algo3,model3,X_cal) #predict classfication score of calibration data
                    W_test3=Pred(algo3,model3,X_test) #predict classfication score of test data
                    # pval3 <- confomalPvalue_online(W_cal3,W_test3,Null_cal,Value,theta)
                    
                    
                    
                    
                    W_cal_all <- as.data.frame(cbind(W_cal1,W_cal2,W_cal3))
                    
                    
                    W_test_all <- as.data.frame(cbind(W_test1,W_test2,W_test3))
                    
                    T <- nrow(data_test)
                    theta <- rep(1,T)
                    theta[Null_test]=0
                    
                    # pvals_opt <- confomalPvalue_online_opt(W_cal_all, W_test_all, Null_cal, Value, theta, Y_cal, Y_test)
                    
                    pvals_random <- confomalPvalue_online_random(W_cal_all, W_test_all, Null_cal, Value, theta)
                    
                    pvals_opt <- confomalPvalue_online_opt_nonNull_EWMA(W_cal_all, W_test_all, Null_cal, Value,theta,lambda = 0.95)
                    
                    rej_opt.lordf <- Lord_feedback(pvals_opt,alpha,theta,W0=alpha/2)
                    
                    rej_random.lordf <- Lord_feedback(pvals_random,alpha,theta,W0=alpha/2)
                    
                    rej_opt.lordf.conse <- Lord_feedback_conservative(pvals_opt,alpha,theta,W0=alpha/2)
                    
                    rej_random.lordf.conse <- Lord_feedback_conservative(pvals_random,alpha,theta,W0=alpha/2)
                    
                    
                    rej_opt.saf <- SAFFRON_feedback(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                    
                    rej_random.saf <- SAFFRON_feedback(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                    
                    rej_opt.saf.conse <- SAFFRON_feedback_conservative(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                    
                    rej_random.saf.conse <- SAFFRON_feedback_conservative(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                    
                    
                    rej_LORD_opt <- LORD(pvals_opt,alpha)
                    rej_LORD_random <- LORD(pvals_random,alpha)
                    
                    
                    rej_SA_opt <- SAFFRON(pvals_opt,alpha)
                    rej_SA_random <- SAFFRON(pvals_random,alpha)
                    
                    
                    res_feedback_opt_online.lordf <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.lordf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_random_online.lordf <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.lordf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_opt_online.lordf.conse <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.lordf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_random_online.lordf.conse <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.lordf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    
                    res_feedback_opt_online.saf <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.saf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_random_online.saf <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.saf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_opt_online.saf.conse <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.saf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_feedback_random_online.saf.conse <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.saf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    
                    
                    res_LORD_opt_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_LORD_opt$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_LORD_random_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_LORD_random$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_SA_opt_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_SA_opt$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    res_SA_random_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_SA_random$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    
                    result.opt.lordf <- list(FDP=res_feedback_opt_online.lordf$FDP,
                                             Power=res_feedback_opt_online.lordf$Power,
                                             time=res_feedback_opt_online.lordf$time,
                                             method="Opt-LF")
                    
                    
                    
                    result.random.lordf <- list(FDP=res_feedback_random_online.lordf$FDP,
                                                Power=res_feedback_random_online.lordf$Power,
                                                time=res_feedback_random_online.lordf$time,
                                                method="Ran-LF")
                    
                    
                    result.opt.lordf.conse <- list(FDP=res_feedback_opt_online.lordf.conse$FDP,
                                                   Power=res_feedback_opt_online.lordf.conse$Power,
                                                   time=res_feedback_opt_online.lordf.conse$time,
                                                   method="Opt-LFS")
                    
                    
                    
                    result.random.lordf.conse <- list(FDP=res_feedback_random_online.lordf.conse$FDP,
                                                      Power=res_feedback_random_online.lordf.conse$Power,
                                                      time=res_feedback_random_online.lordf.conse$time,
                                                      method="Ran-LFS")
                    
                    
                    result.opt.saf <- list(FDP=res_feedback_opt_online.saf$FDP,
                                           Power=res_feedback_opt_online.saf$Power,
                                           time=res_feedback_opt_online.saf$time,
                                           method="Opt-SF")
                    
                    
                    
                    result.random.saf <- list(FDP=res_feedback_random_online.saf$FDP,
                                              Power=res_feedback_random_online.saf$Power,
                                              time=res_feedback_random_online.saf$time,
                                              
                                              method="Ran-SF")
                    
                    
                    result.opt.saf.conse <- list(FDP=res_feedback_opt_online.saf.conse$FDP,
                                                 Power=res_feedback_opt_online.saf.conse$Power,
                                                 time=res_feedback_opt_online.saf.conse$time,
                                                 method="Opt-SFS")
                    
                    
                    
                    result.random.saf.conse <- list(FDP=res_feedback_random_online.saf.conse$FDP,
                                                    Power=res_feedback_random_online.saf.conse$Power,
                                                    time=res_feedback_random_online.saf.conse$time,
                                                    
                                                    method="Ran-SFS")
                    
                    
                    
                    result.LORD.opt <- list(FDP=res_LORD_opt_online$FDP,
                                            Power=res_LORD_opt_online$Power,
                                            time=res_LORD_opt_online$time,
                                            
                                            method="Opt-LORD++")
                    
                    result.LORD.random <- list(FDP=res_LORD_random_online$FDP,
                                               Power=res_LORD_random_online$Power,
                                               time=res_LORD_random_online$time,
                                               
                                               method="Ran-LORD++")
                    
                    result.SA.opt <- list(FDP=res_SA_opt_online$FDP,
                                          Power=res_SA_opt_online$Power,
                                          time=res_SA_opt_online$time,
                                          
                                          method="Opt-SAFFRON")
                    
                    result.SA.random <- list(FDP=res_SA_random_online$FDP,
                                             Power=res_SA_random_online$Power,
                                             time=res_SA_random_online$time,
                                             
                                             method="Ran-SAFFRON")
                    
                    
                    rej_LOND_random <- LOND(pvals_random,alpha)
                    res_LOND_random_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_LOND_random$R , .x))%>% unlist  %>% split(.,names(.))
                    result.LOND.random <- list(FDP=res_LOND_random_online$FDP,
                                               Power=res_LOND_random_online$Power,
                                               time=res_LOND_random_online$time,
                                               
                                               method="LOND")
                    
                    
                    
                    info=list(
                      opt.lordf=result.opt.lordf,
                      random.lordf=result.random.lordf,
                      
                      opt.lordf.conse=result.opt.lordf.conse,
                      random.lordf.conse=result.random.lordf.conse,
                      
                      opt.saf=result.opt.saf,
                      random.saf=result.random.saf,
                      
                      opt.saf.conse=result.opt.saf.conse,
                      random.saf.conse=result.random.saf.conse,
                      
                      opt.LORD=result.LORD.opt,
                      random.LORD=result.LORD.random,
                      opt.SAFFRON=result.SA.opt,
                      random.SAFFRON=result.SA.random,
                      LOND= result.LOND.random
                    )
                    
                    
                    
                    return(info)
                  }
close(pb)
stopCluster(cl)

Result

str(Result[[1]])





pp <- Result %>%
  map_dfr(~{
    data <- bind_rows(
      tibble(FDP = .x$FDP, Power = .x$Power, Method = .x$method, Time=.x$time)  # 直接提取FDP, Power和Method
    )
    return(data)
  })

head(pp)
summary(pp)


pp$Method=factor(pp$Method,levels = c("Opt-SF","Ran-SF",
                                      "Opt-SFS","Ran-SFS",
                                      "Opt-LF","Ran-LF",
                                      "Opt-LFS","Ran-LFS",
                                      "Opt-SAFFRON","Ran-SAFFRON",
                                      "Opt-LORD++","Ran-LORD++","LOND"))

pp_saffron <- pp[grepl("SF$|SFS$|SAFFRON", pp$Method), ]

pp_lord <- pp[grepl("LF$|LFS$|LORD\\+\\+", pp$Method), ]



library(ggplot2)
library(dplyr)

#t <- seq(10, 500, 50)
t <- seq(10,1000,50)



library(dplyr)
library(ggplot2)

t <- seq(10, 1000, 50)

pp_subset <- pp %>%
  filter(Time %in% t) %>%
  filter(Method %in% c("Opt-SF", "Opt-LF", "Ran-SAFFRON", "Ran-LORD++","LOND")) %>%
  mutate(Method = case_when(
    Method == "Ran-SAFFRON" ~ "SAFFRON",
    Method == "Ran-LORD++" ~ "LORD++",
    TRUE ~ Method
  )) %>%
  group_by(Time, Method) %>%
  summarise(
    FDP_avg = mean(FDP, na.rm = TRUE),
    Power_avg = mean(Power, na.rm = TRUE),
    FDP_se = sd(FDP, na.rm = TRUE) / sqrt(n()),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

pp_subset$Method <- factor(pp_subset$Method, levels = c("Opt-SF", "Opt-LF", "SAFFRON", "LORD++","LOND"))

p1 <- ggplot(data = pp_subset,aes(x=Time,y=FDP_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  geom_ribbon(aes(ymin = FDP_avg - FDP_se, ymax = FDP_avg + FDP_se), alpha = 0.1, linetype = 1, color = NA) +
  xlab("Time")+
  ylab("FDR")+
  ylim(0,0.4)+
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
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p1 <- p1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p1



p2<- ggplot(data = pp_subset,aes(x=Time,y=Power_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  geom_ribbon(aes(ymin = Power_avg - Power_se, ymax = Power_avg + Power_se), alpha = 0.1, linetype = 1, color = NA) +
  xlab("Time")+
  ylab("Power")+
  ylim(0,0.4)+
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
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


p2
p2_diabetes <- p2
p1_diabetes <- p1

t <- 1000
pp_subset <- pp %>%
  filter(Time %in% t) %>%
  filter(Method %in% c("Opt-SF", "Opt-LF", "Opt-SFS", "Opt-LFS", "Ran-SAFFRON", "Ran-LORD++","LOND")) %>%
  mutate(Method = case_when(
    Method == "Ran-SAFFRON" ~ "SAFFRON",
    Method == "Ran-LORD++" ~ "LORD++",
    TRUE ~ Method
  )) %>%
  group_by(Time, Method) %>%
  summarise(
    FDP_avg = mean(FDP, na.rm = TRUE),
    Power_avg = mean(Power, na.rm = TRUE),
    FDP_se = sd(FDP, na.rm = TRUE) / sqrt(n()),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )
pp_subset 


