


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



T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- round(n/2)

algo1<- new("SVM-R") #algorithm used for classification or regression
lambda1<- 1 #specific parameter for the algorithm
algo2<- new("RF") #algorithm used for classification or regression
lambda2<- 500 #specific parameter for the algorithm
algo3<- new("NN-R") #algorithm used for classification or regression


################# 改变不同的non-null proportion ------------
alpha <- 0.05
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=100
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
                      
                      #pi1 <- length(Null_test)/T
                      
                      
                      # model and estimating locfdr -----
                      
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
                      
                      pvals_opt <- confomalPvalue_online_opt_nonNull_EWMA(W_cal_all, W_test_all, Null_cal, Value,theta,lambda = 0.95, L=200)
                      
                      
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

pp <- pp[!pp$Method %in% c("Opt-SAFFRON", "Opt-LORD++"), ]

pp$Method <- as.character(pp$Method)  # 先转换成字符型
pp$Method[pp$Method == "Ran-SAFFRON"] <- "SAFFRON"
pp$Method[pp$Method == "Ran-LORD++"] <- "LORD++"

pp$Method <- factor(pp$Method, levels = c("Opt-SF", "Ran-SF",
                                          "Opt-SFS", "Ran-SFS",
                                          "Opt-LF", "Ran-LF",
                                          "Opt-LFS", "Ran-LFS",
                                          "SAFFRON", "LORD++"))



# P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
#   geom_point(size=2.5)+
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2,3,4,5,6))+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
#                                                                        alpha = 0.1,
#                                                                        linetype = 1,
#                                                                        color=NA)+
#   geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
#   xlab( TeX("$pi_1$"))+
#   ylab("FDR")+
#   scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
#   scale_y_continuous(limits = c(0, 1)) +
#   theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
#   scale_fill_manual(values=c("#BC3C29FF",  # 深红
#                              "#0072B5FF",  # 深蓝
#                              "#E18727FF",  # 橙
#                              "#20854EFF",  # 绿
#                              "#7E6148FF",  # 棕
#                              "#56B4E9FF",  # 浅蓝
#                              "#D55E00FF",  # 橙红
#                              "#F0E442FF",  # 黄
#                              "#CC79A7FF",  # 品红
#                              "#999999FF",  # 灰
#                              "#009E73FF",  # 青绿
#                              "#FF6DB6FF"   ))+
#   theme(axis.text = element_text(size = 16),
#         axis.title = element_text(size = 20),
#         legend.text = element_text(size = 16),
#         legend.title = element_text(size = 16),
#         panel.grid.major=element_line(colour=NA),
#         panel.background = element_rect(fill = "transparent",colour = NA),
#         plot.background = element_rect(fill = "transparent",colour = NA),
#         panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
#   theme(legend.position = "bottom") 
# P1 <- P1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")
# 
# P1


# P2 <- ggplot(data = pp, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
#   geom_point(size = 2.5) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2,3,4,5,6))+
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
#   scale_color_nejm(palette = c("default"), alpha = 0.9) +
#   scale_fill_manual(values = c("#BC3C29FF",  # 深红
#                                "#0072B5FF",  # 深蓝
#                                "#E18727FF",  # 橙
#                                "#20854EFF",  # 绿
#                                "#7E6148FF",  # 棕
#                                "#56B4E9FF",  # 浅蓝
#                                "#D55E00FF",  # 橙红
#                                "#F0E442FF",  # 黄
#                                "#CC79A7FF",  # 品红
#                                "#999999FF",  # 灰
#                                "#009E73FF",  # 青绿
#                                "#FF6DB6FF"   )) +
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
# P2


# P2 <- ggplot(data = pp, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
#   geom_point(size = 2.5) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2,3,4,5,6))+
#   geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
#   xlab(TeX("$\\pi_1$")) +
#   ylab("Power") +
#   scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
#   scale_y_continuous(limits = c(0, 1)) +
#   theme_bw() +
#   scale_color_nejm(palette = c("default"), alpha = 0.9) +
#   scale_fill_manual(values = c("#BC3C29FF",  # 深红
#                                "#0072B5FF",  # 深蓝
#                                "#E18727FF",  # 橙
#                                "#20854EFF",  # 绿
#                                "#7E6148FF",  # 棕
#                                "#56B4E9FF",  # 浅蓝
#                                "#D55E00FF",  # 橙红
#                                "#F0E442FF",  # 黄
#                                "#CC79A7FF",  # 品红
#                                "#999999FF",  # 灰
#                                "#009E73FF",  # 青绿
#                                "#FF6DB6FF"   )) +
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
# P2


pp_saffron <- pp[grepl("SF$|SFS$|SAFFRON", pp$Method), ]

pp_lord <- pp[grepl("LF$|LFS$|LORD\\+\\+", pp$Method), ]

# P1_saffron <- ggplot(data = pp_saffron, aes(x = prop, y = FDR, group = Method, color = Method, shape = Method, fill = Method)) +
#   geom_point(size = 2.5) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1, 2, 3, 4, 5, 6)) +
#   geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se), alpha = 0.1, linetype = 1, color = NA) +
#   geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
#   xlab(TeX("$\\pi_1$")) +
#   ylab("FDR") +
#   scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
#   scale_y_continuous(limits = c(0, 1)) +
#   theme_bw() +
#   scale_color_manual(values = c(
#     "#BC3C29CC", # red
#     "#0072B5CC", # blue
#     "#E18727CC", # orange
#     "#20854ECC", # green
#     "#7876B1CC", # purple
#     "#6F99ADCC", # cyan
#     "#FFDC91CC", # light orange
#     "#EE4C97CC", # pink
#     "#8F786BCC"  # soft brown 
#   )) +
#   scale_fill_manual(values = c(
#     "#BC3C29CC", # red
#     "#0072B5CC", # blue
#     "#E18727CC", # orange
#     "#20854ECC", # green
#     "#7876B1CC", # purple
#     "#6F99ADCC", # cyan
#     "#FFDC91CC", # light orange
#     "#EE4C97CC", # pink
#     "#8F786BCC"  # soft brown 
#   )) +
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
#   ) +
#   geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")
# 
# P1_saffron
# 
# 
# P2_saffron <- ggplot(data = pp_saffron, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
#   geom_point(size = 2.5) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1, 2, 3, 4, 5, 6)) +
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
#   scale_color_manual(values = c(
#     "#BC3C29CC", # red
#     "#0072B5CC", # blue
#     "#E18727CC", # orange
#     "#20854ECC", # green
#     "#7876B1CC", # purple
#     "#6F99ADCC", # cyan
#     "#FFDC91CC", # light orange
#     "#EE4C97CC", # pink
#     "#8F786BCC"  # soft brown 
#   )) +
#   scale_fill_manual(values = c(
#     "#BC3C29CC", # red
#     "#0072B5CC", # blue
#     "#E18727CC", # orange
#     "#20854ECC", # green
#     "#7876B1CC", # purple
#     "#6F99ADCC", # cyan
#     "#FFDC91CC", # light orange
#     "#EE4C97CC", # pink
#     "#8F786BCC"  # soft brown 
#   )) +
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
# P2_saffron
# 
# my_colors <- c(
#   "#BC3C29CC", # red
#   "#0072B5CC", # blue
#   "#E18727CC", # orange
#   "#20854ECC", # green
#   "#7876B1CC", # purple
#   "#6F99ADCC", # cyan
#   "#FFDC91CC", # light orange
#   "#EE4C97CC", # pink
#   "#8F786BCC"  # soft brown 
# )
# 
# 
# 
# 



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
  scale_y_continuous(limits = c(0, 1)) +
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
  scale_y_continuous(limits = c(0, 1)) +
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
  scale_y_continuous(limits = c(0, 1)) +
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
  scale_y_continuous(limits = c(0, 1)) +
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



plot_reg_Opt_SAFFRON <- ggarrange(P1_saffron, P2_saffron, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                  font.label = list(size = 16, face = "bold"))


plot_reg_Opt_LORD <- ggarrange(P1_lord,P2_lord, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                               font.label = list(size = 16, face = "bold"))


plot_reg_Opt <- ggarrange(plot_reg_Opt_LORD,plot_reg_Opt_SAFFRON, ncol=1, nrow=2,
                          common.legend = FALSE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))
pdf(file = "plot_reg_Opt_vary_prop.pdf",width = 10,height = 8) 
plot_reg_Opt 
dev.off()
dev.new()

write.csv(pp,"plot_reg_opt_data.csv")
