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

confomalPvalue_online <- function(W_cal, W_test, Null_cal, Value, theta) {
  Phi_cal <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  n2 <- length(Phi_Null)
  
  pvalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    xi <- runif(1)
    pvalues[i] <- (sum(Phi_Null < t) + xi * sum(Phi_Null == t)) / (n2 + 1)
    
    if (theta[i] == 0) {
      new_score <- -ScoreCompute(W_test[i], Value)
      Phi_Null <- c(Phi_Null, new_score)
      n2 <- length(Phi_Null)
    }
  }
  
  return(pvalues)
}

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

T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)

algo<- new("RF") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm


alpha <- 0.2
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=500
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
                      data <- data_generation_regression(N=T)
                      Value=list(type="<=A",v=quantile(data$y,1-pi_1))
                      his_data <- data_generation_regression(N=n)
                      p <- ncol(his_data)-1 # dimension of covariates
                      
                      head(data)
                      
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
                      
                      
                      
                      
                      rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_feedback_LORD$R,theta,"LF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                      res <- CiterionCompute(rej_feedback_LORD_conse$R,theta,"LFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_LORD=LORD(pval,alpha,version = "++")
                      res <- CiterionCompute(rej_LORD$R,theta,"LORD++")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                      res <- CiterionCompute(rej_SAFFRON$R,theta,"SAFFRON")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                      res <- CiterionCompute(rej_feedback_SA$R,theta,"SF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      res <- CiterionCompute(rej_feedback_SA_conse$R,theta,"SFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_LOND=LOND(pval,alpha)
                      res=CiterionCompute(rej_LOND$R,theta,"LOND")
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

pp$Method=factor(pp$Method,levels =  c("SF","LF","SFS","LFS","SAFFRON",
                                       "LORD++","LOND"))


P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
                                                                       alpha = 0.1,
                                                                       linetype = 1,
                                                                       color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", 
    "#0072B5CC", 
    "#E18727CC", 
    "#20854ECC", 
    "#7876B1CC", 
    "#6F99ADCC", 
    "#FFDC91CC", 
    "#EE4C97CC", 
    "#8F786BCC"  
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
P1 <- P1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P1




P2 <- ggplot(data = pp, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1,
              linetype = 1,
              color = NA) +
  geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +
  scale_color_nejm(palette = c("default"), alpha = 0.9) +
  scale_fill_manual(values = c(
    "#BC3C29CC", 
    "#0072B5CC", 
    "#E18727CC", 
    "#20854ECC", 
    "#7876B1CC", 
    "#6F99ADCC", 
    "#FFDC91CC", 
    "#EE4C97CC", 
    "#8F786BCC"  
  )) +
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
  )

P2


reg_plots_vary_prop <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                 font.label = list(size = 20, face = "bold"))

pdf(file = "reg_plots_vary_prop.pdf",width = 10,height = 4) 
reg_plots_vary_prop
dev.off()

write.csv(pp,"reg_plots_vary_prop_plot_data.csv")






CriterionCompute_VR <- function(rejection, theta, method_name = " ") {
  V <- sum((1 - theta) * rejection)           
  R <- sum(rejection)                         
  FDP <- V / max(R, 1)                        
  Power <- sum(theta * rejection) / sum(theta)  
  
  return(data.frame(FDR = FDP, Power = Power, V = V, R = R, Method = method_name))
}


alpha <- 0.2
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=500
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
                      
                      
                      data <- data_generation_regression2(N=T)
                      Value=list(type="<=A",v=quantile(data$y,1-pi_1))
                      his_data <- data_generation_regression2(N=n)
                      
                      
                      
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
                      
                      
                      
                      
                      rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                      res <- CriterionCompute_VR(rej_feedback_LORD$R,theta,"LF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                      res <- CriterionCompute_VR(rej_feedback_LORD_conse$R,theta,"LFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_LORD=LORD(pval,alpha,version = "++")
                      res <- CriterionCompute_VR(rej_LORD$R,theta,"LORD++")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                      res <- CriterionCompute_VR(rej_SAFFRON$R,theta,"SAFFRON")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                      res <-CriterionCompute_VR(rej_feedback_SA$R,theta,"SF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      res <- CriterionCompute_VR(rej_feedback_SA_conse$R,theta,"SFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_LOND=LOND(pval,alpha)
                      res=CriterionCompute_VR(rej_LOND$R,theta,"LOND")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                    }
                    
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)

pp <- result %>%
  group_by(Method, prop) %>%
  summarize(
    mFDR = mean(V) / pmax(mean(R), 1),
    FDR = mean(FDR),
    POWER = mean(Power),
    FDP_se = sd(FDR, na.rm = TRUE) / sqrt(length(FDR)),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(length(Power)),
    .groups = "drop"
  )

pp

pp$Method=factor(pp$Method,levels =  c("SF","LF","SFS","LFS","SAFFRON",
                                       "LORD++","LOND"))


P1 <- ggplot(data = pp,aes(x = prop, y = mFDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("mFDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", 
    "#0072B5CC", 
    "#E18727CC", 
    "#20854ECC", 
    "#7876B1CC", 
    "#6F99ADCC", 
    "#FFDC91CC", 
    "#EE4C97CC", 
    "#8F786BCC"  
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
P1 <- P1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P1


P2 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", 
    "#0072B5CC", 
    "#E18727CC", 
    "#20854ECC", 
    "#7876B1CC", 
    "#6F99ADCC", 
    "#FFDC91CC", 
    "#EE4C97CC", 
    "#8F786BCC"  
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
P2 <- P2 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P2

reg_plots_vary_prop_mFDR <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                      font.label = list(size = 20, face = "bold"))

pdf(file = "reg_plots_vary_prop_mFDR.pdf",width = 10,height = 4) 
reg_plots_vary_prop_mFDR
dev.off()

write.csv(pp,"reg_plots_vary_prop_mFDR_plot_data.csv")