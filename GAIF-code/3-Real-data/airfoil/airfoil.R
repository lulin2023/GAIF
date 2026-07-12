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

dat = read.table("airfoil.txt")
dim(dat)
colnames(dat) = c("Frequency",
                  "Angle",
                  "Chord",
                  "Velocity",
                  "Suction",
                  "Sound")

dat.x = as.matrix(dat[,1:5])
dat.y = as.numeric(dat[,6])
dat.x[,1] = log(dat.x[,1]) # Log transform
dat.x[,5] = log(dat.x[,5]) # Log transform
N = nrow(dat.x); p = ncol(dat.x)

## Exponential tilting functions
w = function(x) {
  exp(x[,c(1,5)] %*% c(-2,1))
}

wsample = function(wts, frac=0.25) {
  n = length(wts)
  i = c()
  while(length(i) <= n*frac) {
    i = c(i, which(runif(n) <= wts/max(wts)))
  }
  return(i)
}





##################  compare with random score selection  -------------------
#T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.3 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
#Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))
algo1<- new("SVM-R") #algorithm used for classification or regression
lambda1<- 1 #specific parameter for the algorithm
algo2<- new("RF") #algorithm used for classification or regression
lambda2<- 500 #specific parameter for the algorithm
algo3<- new("NN") #algorithm used for classification or regression

T <- N



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
                    n = round(N/4)
                    i = sample(N,n)
                    x = dat.x[i,]; y = dat.y[i]
                    x0 = dat.x[-i,]; y0 = dat.y[-i]
                    
                    his_data <- data.frame(x=x,y=y)
                    data_test <- data.frame(x=x0,y=y0)
                    n_cal = round(n/2)
                    
                    Value=list(type="<=A",v=quantile(data_test$y,1-0.6))
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
                    
                    T <- nrow(data_test)
                    theta <- rep(1,T)
                    theta[Null_test]=0
                   
                    
                    # training K = 3 models and calculate conformal p-values -----
                    
                    ## Model1: SVM
                    
                    
                    model1=fitting(algo1,X_train,Y_train,lambda1) #estimate model by training data
                    W_cal1=Pred(algo1,model1,X_cal) #predict classfication score of calibration data
                    W_test1=Pred(algo1,model1,X_test) #predict classfication score of test data
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

# 查看Result中第一个元素的结构
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




library(ggplot2)
library(dplyr)


library(dplyr)
library(ggplot2)

t <- seq(10, 1000, 50)

#t <- 1000

# 筛选并重命名
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
  xlab("Time")+#横坐标名???
  ylab("Power")+#纵坐标名???+
  ylim(0,0.8)+
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
p2_airfoil <- p2
p1_airfoil <- p1


t <- seq(10, 1000, 50)

# 筛选并重命名
pp_subset <- pp %>%
  filter(Time %in% t) %>%
  filter(Method %in% c("Opt-SF", "Opt-LF", "Opt-SFS", "Opt-LFS", "Ran-SAFFRON", "Ran-LORD++")) %>%
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

pp_subset$Method <- factor(pp_subset$Method, levels = c("Opt-SF", "Opt-LF", "Opt-SFS", "Opt-LFS", "SAFFRON", "LORD++"))

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
  xlab("Time")+#横坐标名???
  ylab("Power")+#纵坐标名???+
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

# p_power <- ggarrange(p2_candidate,p2_diabetes,p2_income,p2_airfoil,
#                      ncol=2, nrow=2,
#                      common.legend = TRUE, legend="bottom",
#                      font.label = list(size = 16, face = "bold"))
pdf(file = "plot_power_Opt.pdf",width = 12,height = 4) 
p_power
dev.off()
dev.new()

library(ggplot2)
library(ggpubr)

p2_candidate <- p2_candidate + ggtitle("Task 1") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p2_diabetes <- p2_diabetes + ggtitle("Task 2") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p2_income <- p2_income + ggtitle("Task 3") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p2_airfoil <- p2_airfoil + ggtitle("Task 4") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p_power <- ggarrange(
  p2_candidate, p2_diabetes, p2_income, p2_airfoil,
  ncol = 4, nrow = 1,
  common.legend = TRUE, legend = "bottom",
  font.label = list(size = 16, face = "bold")
)


p1_candidate <- p1_candidate + ggtitle("Task 1") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p1_diabetes <- p1_diabetes + ggtitle("Task 2") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p1_income <- p1_income + ggtitle("Task 3") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p1_airfoil <- p1_airfoil + ggtitle("Task 4") +
  theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))

p_FDR <- ggarrange(
  p1_candidate, p1_diabetes, p1_income, p1_airfoil,
  ncol = 4, nrow = 1,
  common.legend = TRUE, legend = "bottom",
  font.label = list(size = 16, face = "bold")
)

pdf(file = "plot_FDR_Opt.pdf",width = 12,height = 4) 
p_FDR
dev.off()
dev.new()

p_FDR_Power <- ggarrange(
  p1_candidate, p1_diabetes, p1_income, p1_airfoil,
  p2_candidate, p2_diabetes, p2_income, p2_airfoil,
  ncol = 4, nrow = 2,
  common.legend = TRUE, legend = "bottom",
  font.label = list(size = 16, face = "bold")
)

pdf(file = "plot_FDR_Power_Opt.pdf",width = 12,height = 7) 
p_FDR_Power
dev.off()
dev.new()


