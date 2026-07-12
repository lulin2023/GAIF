

############## score optimization ----------------
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("onlineFDR")

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
source("Model-sel-func.R")






confomalPvalue_online_random <- function(W_cal_all, W_test_all, Null_cal, Value, theta) {
  n_cal <- nrow(W_cal_all)
  n_test <- nrow(W_test_all)
  K <- ncol(W_cal_all)  # 预测模型的数量
  
  # 计算所有 K 个模型的 conformity score
  Phi_cal_all <- -apply(W_cal_all, 2, ScoreCompute, Value)
  Phi_test_all <- -apply(W_test_all, 2, ScoreCompute, Value)
  
  # 只取 Null 相关的 conformity score
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  n2 <- nrow(Phi_Null_all)
  
  # 计算每个模型的 p-value
  pvalues <- numeric(n_test)
  
  for (t in 1:n_test) {
    # 随机选择一个模型 k_opt
    k_opt <- sample(1:K, 1)
    
    xi <- runif(1)
    # 计算选定模型的 p-value
    pvalues[t] <- (sum(Phi_Null_all[, k_opt] < Phi_test_all[t, k_opt]) + xi*(1+sum(Phi_Null_all[, k_opt] == Phi_test_all[t, k_opt]))) / (n2 + 1)
    
    # 只有当 theta[t] == 0 时，才加入新的 test 数据到 calibration
    if (theta[t] == 0) {
      new_score <- -ScoreCompute(W_test_all[t, k_opt], Value)
      Phi_Null_all <- rbind(Phi_Null_all, new_score)
      n2 <- nrow(Phi_Null_all)
    }
    print(t)
  }
  return(pvalues)
}








################# with conformity score selection -----------


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
algo1<- new("SVM") #algorithm used for classification or regression
lambda1<- 1 #specific parameter for the algorithm
algo2<- new("RFc") #algorithm used for classification or regression
lambda2<- 500 #specific parameter for the algorithm
algo3<- new("NN") #algorithm used for classification or regression




##################  compare with random score selection  -------------------
T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)




nr=500
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
                    
                    
                    # generate data
                    data <- data_generation_classication1(N=T,mu1= c(2,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=0.3,pi=0.3)
                    
                    # generate history data and estimate K (diversity threshold)---
                    his_data <- data_generation_classication1(N=n,mu1= c(2,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=0.3,pi=0.3)
                    
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
                    
                    theta <- Y_test
                    
                    
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
                      random.SAFFRON=result.SA.random
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
      tibble(FDP = .x$FDP, Power = .x$Power, Method = .x$method, Time=.x$time)  
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
                                      "Opt-LORD++","Ran-LORD++"))




library(ggplot2)
library(dplyr)

my_colors <- c(
  "#BC3C29FF",  # 
  "#0072B5FF",  # 
  "#E18727FF",  # 
  "#20854EFF",  # 
  "#7E6148FF",  # 
  "#56B4E9FF",  # 
  "#D55E00FF",  # 
  "#F0E442FF",  # 
  "#CC79A7FF",  # 
  "#999999FF",  # 
  "#009E73FF",  # 
  "#FF6DB6FF"   # 
)

#t <- seq(10, 500, 50)
t <- seq(10,1000,50)

pp_subset <- pp %>%
  filter(Time %in% t) %>%
  group_by(Time, Method) %>%
  summarize(
    FDP_avg = mean(FDP, na.rm = TRUE),
    Power_avg = mean(Power, na.rm = TRUE),
    FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
    Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power))
  )


pp_subset.filtered  <- pp_subset
summary(pp_subset)

p1 <- ggplot(data = pp_subset.filtered,aes(x=Time,y=FDP_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+geom_ribbon(aes(ymin = FDP_avg-FDP_se,ymax = FDP_avg+FDP_se),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab("FDR")+
  ylim(0,0.2)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c(
    "#BC3C29FF",  # 
    "#0072B5FF",  # 
    "#E18727FF",  # 
    "#20854EFF",  # 
    "#7E6148FF",  # 
    "#56B4E9FF",  # 
    "#D55E00FF",  # 
    "#F0E442FF",  # 
    "#CC79A7FF",  # 
    "#999999FF",  # 
    "#009E73FF",  # 
    "#FF6DB6FF"   # 
  ))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

p1 <- p1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p1



p2<- ggplot(data = pp_subset.filtered,aes(x=Time,y=Power_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab("Power")+
  ylim(0,1)+
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
  ))+theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 


p2

p3<- ggplot(data = pp_subset.filtered,aes(x=Time,y=Power_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab("Power")+
  ylim(0.6,0.85)+
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


p3

p4 <- ggplot(data = pp_subset.filtered,aes(x=Time,y=FDP_avg,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab("FDR")+
  ylim(0,0.2)+
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

p4 <- p4 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p4



plot_cla_Opt <- ggarrange(p4, p3, ncol=2, nrow=1, common.legend = TRUE, legend="right",
                          font.label = list(size = 16, face = "bold"))

pdf(file = "plot_cla_Opt.pdf",width = 10,height = 4) 
plot_cla_Opt
dev.off()
dev.new()


################## 改变不同的non-null proportion ----------------
  
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
lambda1<- 2 #specific parameter for the algorithm
algo2<- new("RFc") #algorithm used for classification or regression
lambda2<- 600 #specific parameter for the algorithm
algo3<- new("NN") #algorithm used for classification or regression



alpha <- 0.05
T <- 500
n <- 1000
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)

pi1_seq=seq(0.1,0.8,0.1)

nr=200
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                                                 "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                                                 "tidyverse","lsa","magrittr","doSNOW","pbmcapply"),
                  .errorhandling = "remove", .options.snow = opts)%dopar% {
                    info<-data.frame()
                    
                    for (pi_1 in pi1_seq) {
                      # # Z0=rnorm(N*(1-prop),0,1)
                      # # Z1=rnorm(N*(prop),signal,1)
                      # 
                      # #data=data.frame(Z=c(Z0,Z1),p=1-pnorm(c(Z0,Z1)),theta=c(rep(0,N*(1-prop)),rep(1,N*(prop) ) ) )
                      # 
                      # data <- generate_data_Gaussian(N, pi_1, mu_c_values)
                      # head(data)
                      # data=data[sample(nrow(data)), ]
                      
                      
                      # generate data
                      data <- data_generation_classication1(N=T,mu1= c(2,0,0,0),mu2=c(0,0,-1,-1),p=4,propotion=pi_1,pi=pi_1)
                      
                      # generate history data and estimate K (diversity threshold)---
                      his_data <- data_generation_classication1(N=n,mu1= c(2,0,0,0),mu2=c(0,0,-1,-1),p=4,propotion=pi_1,pi=pi_1)
                      
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
                      
                      theta <- Y_test
                      
                      
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
                      
                      
                      
                      # pvals_opt <- confomalPvalue_online_opt(W_cal_all, W_test_all, Null_cal, Value, theta, Y_cal, Y_test)
                      
                      pvals_random <- confomalPvalue_online_random(W_cal_all, W_test_all, Null_cal, Value, theta)
                      
                      pvals_opt <- confomalPvalue_online_opt_nonNull_EWMA(W_cal_all, W_test_all, Null_cal, Value,theta,lambda = 0.9, L = 100)
                      
                      
                      rej_opt.lordf <- Lord_feedback(pvals_opt,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_opt.lordf$R,theta,"Opt-LF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_random.lordf <- Lord_feedback(pvals_random,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_random.lordf$R,theta,"Ran-LF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      
                      rej_opt.lordf.conse <- Lord_feedback_conservative(pvals_opt,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_opt.lordf.conse$R,theta,"Opt-LFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_random.lordf.conse <- Lord_feedback_conservative(pvals_random,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_random.lordf.conse$R,theta,"Ran-LFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_opt.saf <- SAFFRON_feedback(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                      res <- CiterionCompute(rej_opt.saf$R,theta,"Opt-SF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_random.saf <- SAFFRON_feedback(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                      res <- CiterionCompute(rej_random.saf$R,theta,"Ran-SF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_opt.saf.conse <- SAFFRON_feedback_conservative(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                      res <- CiterionCompute(rej_opt.saf.conse$R,theta,"Opt-SFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_random.saf.conse <- SAFFRON_feedback_conservative(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                      res <- CiterionCompute(rej_random.saf.conse$R,theta,"Ran-SFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_LORD_opt <- LORD(pvals_opt,alpha)
                      res <- CiterionCompute(rej_LORD_opt$R,theta,"Opt-LORD++")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_LORD_random <- LORD(pvals_random,alpha)
                      res <- CiterionCompute(rej_LORD_random$R,theta,"Ran-LORD++")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_SA_opt <- SAFFRON(pvals_opt,alpha)
                      res <- CiterionCompute(rej_SA_opt$R,theta,"Opt-SAFFRON")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_SA_random <- SAFFRON(pvals_random,alpha)
                      res <- CiterionCompute(rej_SA_random$R,theta,"Ran-SAFFRON")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      
                    }
                    
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)


pp <- result%>%
  group_by(Method,prop)%>%
  dplyr::summarize(FDR=mean(FDP),POWER=mean(Power),
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power)))
pp

# pp$Method=factor(pp$Method,levels = c("Opt-SF","Ran-SF",
#                                       "Opt-SFS","Ran-SFS",
#                                       "Opt-LF","Ran-LF",
#                                       "Opt-LFS","Ran-LFS",
#                                       "Opt-SAFFRON","Ran-SAFFRON",
#                                       "Opt-LORD++","Ran-LORD++"))

# 去除不需要的水平
pp <- pp[!pp$Method %in% c("Opt-SAFFRON", "Opt-LORD++"), ]

# 修改水平名称：去掉 "Ran-" 前缀
pp$Method <- as.character(pp$Method)  # 先转换成字符型
pp$Method[pp$Method == "Ran-SAFFRON"] <- "SAFFRON"
pp$Method[pp$Method == "Ran-LORD++"] <- "LORD++"

# 重新设置因子顺序
pp$Method <- factor(pp$Method, levels = c("Opt-SF", "Ran-SF",
                                          "Opt-SFS", "Ran-SFS",
                                          "Opt-LF", "Ran-LF",
                                          "Opt-LFS", "Ran-LFS",
                                          "SAFFRON", "LORD++"))




pp_saffron <- pp[grepl("SF$|SFS$|SAFFRON", pp$Method), ]

pp_lord <- pp[grepl("LF$|LFS$|LORD\\+\\+", pp$Method), ]


library(ggplot2)
library(latex2exp)

method_levels <- c("Opt-SF", "Ran-SF",
                   "Opt-SFS", "Ran-SFS",
                   "Opt-LF", "Ran-LF",
                   "Opt-LFS", "Ran-LFS",
                   "SAFFRON", "LORD++")

method_colors <- c(
  "#BC3C29", "#F4A582",   # SF
  "#0072B5", "#A6CEE3",   # SFS
  "#E18727", "#FDBF6F",   # LF
  "#20854E", "#B2DF8A",   # LFS
  "#984EA3", "#CAB2D6"    # SAFFRON / LORD++
)
names(method_colors) <- method_levels

shape_values <- c(21, 21, 22, 22, 23, 23, 24, 24, 25, 25)
names(shape_values) <- method_levels

linetype_values <- c("solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "dashed", "dashed")
names(linetype_values) <- method_levels



P1_saffron <- ggplot(data = pp_saffron, aes(x = prop, y = FDR, group = Method,
                                            color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se), alpha = 0.1, color = NA) +
  geom_line(aes(linetype = Method), linewidth = 0.8) +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  scale_shape_manual(values = shape_values) +
  scale_linetype_manual(values = linetype_values) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.3)) +
  theme_bw() +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  ) +
  geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")

P1_saffron



P2_saffron <- ggplot(data = pp_saffron, aes(x = prop, y = POWER, group = Method,
                                            color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se), alpha = 0.1, color = NA) +
  geom_line(aes(linetype = Method), linewidth = 0.8) +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  scale_shape_manual(values = shape_values) +
  scale_linetype_manual(values = linetype_values) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.5)) +
  theme_bw() +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  )

P2_saffron






P1_lord <- ggplot(data = pp_lord, aes(x = prop, y = FDR, group = Method, color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1, 2, 3, 4, 5, 6)) +
  geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se),
              alpha = 0.1,
              linetype = 1,
              color = NA) +
  geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.3)) +
  theme_bw() +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_line(colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.grid.minor = element_blank(),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  ) +
  geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")

P1_lord

# P2_lord <- ggplot(data = pp_lord, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
#   geom_point(size = 2.5) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2,3,4,5,6)) +
#   geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
#               alpha = 0.1,
#               linetype = 1,
#               color = NA) +
#   geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
#   xlab(TeX("$\\pi_1$")) +
#   ylab("Power") +
#   scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
#   scale_y_continuous(limits = c(0, 1)) +
#   theme_bw() +
#   scale_color_manual(values = method_colors) +
#   scale_fill_manual(values = method_colors) +
#   theme(
#     axis.text = element_text(size = 16),
#     axis.title = element_text(size = 20),
#     legend.text = element_text(size = 16),
#     legend.title = element_text(size = 16),
#     panel.grid.major = element_line(colour = NA),
#     panel.background = element_rect(fill = "transparent", colour = NA),
#     plot.background = element_rect(fill = "transparent", colour = NA),
#     panel.grid.minor = element_blank(),
#     text = element_text(size = 16, family = "serif"),
#     legend.position = "bottom"
#   )
# 
# P2_lord

P2_lord <- ggplot(data = pp_lord, aes(x = prop, y = POWER, group = Method,
                                            color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se), alpha = 0.1, color = NA) +
  geom_line(aes(linetype = Method), linewidth = 0.8) +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  scale_shape_manual(values = shape_values) +
  scale_linetype_manual(values = linetype_values) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.6)) +
  theme_bw() +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  )

P2_lord



method_levels <- c("Opt-SF", "Ran-SF",
                   "Opt-SFS", "Ran-SFS",
                   "Opt-LF", "Ran-LF",
                   "Opt-LFS", "Ran-LFS",
                   "SAFFRON", "LORD++")

method_colors <- c(
  "#BC3C29", "#F4A582",   # SF
  "#0072B5", "#A6CEE3",   # SFS
  "#E18727", "#FDBF6F",   # LF
  "#20854E", "#B2DF8A",   # LFS
  "#CAB2D6", "#CAB2D6"    # SAFFRON  LORD++ 
)
names(method_colors) <- method_levels

shape_values <- c(21, 21, 22, 22, 23, 23, 24, 24, 25, 25)
names(shape_values) <- method_levels

linetype_values <- c("solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "dashed", "dashed")
names(linetype_values) <- method_levels


P1_lord <- ggplot(data = pp_lord, aes(x = prop, y = FDR, group = Method,
                                      color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se),
              alpha = 0.1, color = NA) +
  geom_line(aes(linetype = Method), linewidth = 0.8) +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  scale_shape_manual(values = shape_values) +
  scale_linetype_manual(values = linetype_values) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.3)) +
  theme_bw() +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  ) +
  geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")

P1_lord

P2_lord <- ggplot(data = pp_lord, aes(x = prop, y = POWER, group = Method,
                                      color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1, color = NA) +
  geom_line(aes(linetype = Method), linewidth = 0.8) +
  scale_color_manual(values = method_colors) +
  scale_fill_manual(values = method_colors) +
  scale_shape_manual(values = shape_values) +
  scale_linetype_manual(values = linetype_values) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.6)) +
  theme_bw() +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  )

P2_lord



plot_cla_Opt_SAFFRON <- ggarrange(P1_saffron, P2_saffron, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))
pdf(file = "plot_cla_Opt_SAFFRON.pdf",width = 10,height = 4) 
plot_cla_Opt_SAFFRON
dev.off()
dev.new()


plot_cla_Opt_LORD <- ggarrange(P1_lord,P2_lord, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))

pdf(file = "plot_cla_Opt_LORD.pdf",width = 10,height = 4) 
plot_cla_Opt_LORD
dev.off()
dev.new()

plot_cla_Opt <- ggarrange(plot_cla_Opt_LORD,plot_cla_Opt_SAFFRON, ncol=1, nrow=2,
                          common.legend = FALSE, legend="bottom",
                               font.label = list(size = 16, face = "bold"))
pdf(file = "plot_cla_Opt_vary_prop1.pdf",width = 10,height = 8) 
plot_cla_Opt 
dev.off()
dev.new()

write.csv(pp,"plot_cla_opt_data.csv")
