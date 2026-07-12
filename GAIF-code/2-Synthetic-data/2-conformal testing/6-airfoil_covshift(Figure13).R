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

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
## Load the airfoil data
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

summary(dat.y)
## Exponential tilting functions
w = function(x) {
  exp(x[,c(1,5)] %*% c(3,-1))
}

wsample = function(wts, frac=0.25) {
  n = length(wts)
  i = c()
  while(length(i) <= n*frac) {
    i = c(i, which(runif(n) <= wts/max(wts)))
  }
  return(i)
}

library(randomForest)


conformalPvalue_online_weighted_given <- function(
    W_cal, W_test,
    Null_cal, Value, theta,
    w_cal, w_test
) {
  Phi_cal  <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  w_null   <- w_cal[Null_cal]
  
  pvalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    xi <- runif(1)
    
    less <- sum(w_null * (Phi_Null < t))
    ties <- sum(Phi_Null == t)
    num  <- less + xi * w_test[i] * ties
    den  <- sum(w_null) + w_test[i]
    pvalues[i] <- num / den
    
    if (theta[i] == 0) {
      Phi_Null <- c(Phi_Null, t)
      w_null   <- c(w_null, w_test[i])
    }
  }
  
  return(pvalues)
}


conformalPvalue_fixed_weighted_given <- function(
    W_cal, W_test,
    Null_cal, Value, theta,
    w_cal, w_test
) {
  Phi_cal  <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  w_null   <- w_cal[Null_cal]
  
  pvalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    xi <- runif(1)
    
    less <- sum(w_null * (Phi_Null < t))
    ties <- sum(Phi_Null == t)
    num  <- less + xi * w_test[i] * ties
    den  <- sum(w_null) + w_test[i]
    pvalues[i] <- num / den
    
    # if (theta[i] == 0) {
    #   Phi_Null <- c(Phi_Null, t)
    #   w_null   <- c(w_null, w_test[i])
    # }
  }
  
  return(pvalues)
}


alpha <- 0.3
T <- 1000
p <- 5
n <- 1000
n_cal <- n/2

algo<- new("NN-R") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm
pi1_seq=seq(0.2,0.8,0.1)


nr=500
cl = makeCluster(5)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c(
  "MASS","onlineFDR","randomForest","caret","nnet","glmnet","reshape2",
  "kedd","kernlab","e1071","ks","tidyverse","lsa","magrittr","doSNOW",
  "pbmcapply"
), .options.snow = opts, .errorhandling="remove") %dopar% {
  
  info <- data.frame()
  for (pi_1 in pi1_seq) {
    n = round(N/2)
    i = sample(N,n)
    x = dat.x[i,]; y = dat.y[i]
    x0 = dat.x[-i,]; y0 = dat.y[-i]
    
    # Tilting
    i0 = wsample(w(x0))
    x00 = x0[i0,]; y00 = y0[i0]
    
    # after you compute the raw tilt:
    raw_w <- w(rbind(x, x00))
    
    # split into historical vs test
    n_hist <- nrow(x)
    n_test <- nrow(x00)
    raw_w_cal  <- raw_w[       1:n_hist      ]
    raw_w_test <- raw_w[(n_hist+1):(n_hist+n_test)]
    
    # now rescale each vector to [0,1]
    w_cal  <- (raw_w_cal  - min(raw_w_cal )) / (max(raw_w_cal ) - min(raw_w_cal ))
    w_test <- (raw_w_test - min(raw_w_test)) / (max(raw_w_test) - min(raw_w_test))
    
    # (optional) if you want strictly in (0,1), you can bump by eps:
    eps <- 1e-6
    w_cal  <- pmin(1-eps, pmax(eps, w_cal))
    w_test <- pmin(1-eps, pmax(eps, w_test))
    
    
    
    his_data <- data.frame(X=x,Y=y)
    data_test <- data.frame(X=x00,Y=y00)
    n_cal = round(n/2)
    
    
    
    
    Value=list(type="<=A",v=quantile(data_test$Y,1-pi_1))
    
    # 2. Split history into train & calibration
    datawork   <- DataSplit(his_data, n, 0, n_cal)
    data_train <- datawork$data_train
    data_cal   <- datawork$data_cal
    
    X_train <- as.matrix(data_train[, -ncol(data_train)])
    Y_train <- data_train$Y
    X_cal   <- as.matrix(data_cal[,   -ncol(data_cal)])
    Y_cal   <- data_cal$Y
    
    T <- nrow(data_test)
    # 3. Null indices and theta
    Null_cal  <- NullIndex(Y_cal,    Value)
    Null_test <- NullIndex(data_test$Y, Value)
    theta     <- rep(1, T)
    theta[Null_test] <- 0
    
    # 4. Fit model & compute raw scores
    model   <- fitting(algo, X_train, Y_train, lambda)
    W_cal   <- Pred(algo, model, X_cal)  # these are V_i = -ScoreCompute
    W_test  <- Pred(algo, model, as.matrix(data_test[, -ncol(data_test)]))
    
    
    # 5. Compute online *weighted* conformal p-values
    # pval_w <- conformalPvalue_online_weighted_given(
    #   W_cal, W_test,
    #   Null_cal, Value, theta,
    #   w_cal, w_test
    # )
    
    
    pval_w <- conformalPvalue_online_weighted_given(
      W_cal, W_test,
      Null_cal, Value, theta,
      w_cal, w_test
    )
    
    # pval <- confomalPvalue_online(W_cal, W_test, Null_cal, Value, theta)
    
    plot(pval_w,theta)
    
    summary(pval_w)
    length(pval_w)
    length(theta)
    
    # 6. Apply Online-FDR procedures
    out1 <- Lord_feedback(pval_w, alpha, theta, W0=alpha/2)
    r1   <- CiterionCompute(out1$R, theta, "LF"); r1["prop"]=pi_1;  info <- rbind(info, r1)
    
    out2 <- Lord_feedback_conservative(pval_w, alpha, theta, W0=alpha/2)
    r2   <- CiterionCompute(out2$R, theta, "LFS"); r2["prop"]=pi_1; info <- rbind(info, r2)
    
    out3 <- LORD(pval_w, alpha, version="++")
    summary(pval_w)
    r3   <- CiterionCompute(out3$R, theta, "LORD++"); r3["prop"]=pi_1; info <- rbind(info, r3)
    
    out4 <- SAFFRON(pval_w, alpha, lambda=0.5)
    r4   <- CiterionCompute(out4$R, theta, "SAFFRON"); r4["prop"]=pi_1; info <- rbind(info, r4)
    
    out5 <- SAFFRON_feedback(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r5   <- CiterionCompute(out5$R, theta, "SF");   r5["prop"]=pi_1;   info <- rbind(info, r5)
    
    out6 <- SAFFRON_feedback_conservative(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r6   <- CiterionCompute(out6$R, theta, "SFS");   r6["prop"]=pi_1;  info <- rbind(info, r6)
    
    out7 <- LOND(pval_w, alpha)
    r7   <- CiterionCompute(out7$R, theta, "LOND"); r7["prop"]=pi_1;   info <- rbind(info, r7)
    
  }
  return(info)
}  # end foreach


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
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.1), limits = c(0.2, 0.8)) +
  scale_y_continuous(limits = c(0, 0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
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
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.1), limits = c(0.2, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +
  scale_color_nejm(palette = c("default"), alpha = 0.9) +
  scale_fill_manual(values = c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
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


reg_plots_vary_prop_airfoil <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                         font.label = list(size = 20, face = "bold"))
reg_plots_vary_prop_airfoil

pdf(file = "reg_plots_vary_prop_airfoil.pdf",width = 10,height = 4) 
reg_plots_vary_prop_airfoil
dev.off()


#################### use unweighted p-values ----------

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c(
  "MASS","onlineFDR","randomForest","caret","nnet","glmnet","reshape2",
  "kedd","kernlab","e1071","ks","tidyverse","lsa","magrittr","doSNOW",
  "pbmcapply"
), .options.snow = opts, .errorhandling="remove") %dopar% {
  
  info <- data.frame()
  for (pi_1 in pi1_seq) {
    n = round(N/2)
    i = sample(N,n)
    x = dat.x[i,]; y = dat.y[i]
    x0 = dat.x[-i,]; y0 = dat.y[-i]
    
    # Tilting
    i0 = wsample(w(x0))
    x00 = x0[i0,]; y00 = y0[i0]
    
    # after you compute the raw tilt:
    raw_w <- w(rbind(x, x00))
    
    # split into historical vs test
    n_hist <- nrow(x)
    n_test <- nrow(x00)
    raw_w_cal  <- raw_w[       1:n_hist      ]
    raw_w_test <- raw_w[(n_hist+1):(n_hist+n_test)]
    
    # now rescale each vector to [0,1]
    w_cal  <- (raw_w_cal  - min(raw_w_cal )) / (max(raw_w_cal ) - min(raw_w_cal ))
    w_test <- (raw_w_test - min(raw_w_test)) / (max(raw_w_test) - min(raw_w_test))
    
    # (optional) if you want strictly in (0,1), you can bump by eps:
    eps <- 1e-6
    w_cal  <- pmin(1-eps, pmax(eps, w_cal))
    w_test <- pmin(1-eps, pmax(eps, w_test))
    
    
    
    his_data <- data.frame(X=x,Y=y)
    data_test <- data.frame(X=x00,Y=y00)
    n_cal = round(n/2)
    
    
    
    
    Value=list(type="<=A",v=quantile(data_test$Y,1-pi_1))
    
    # 2. Split history into train & calibration
    datawork   <- DataSplit(his_data, n, 0, n_cal)
    data_train <- datawork$data_train
    data_cal   <- datawork$data_cal
    
    X_train <- as.matrix(data_train[, -ncol(data_train)])
    Y_train <- data_train$Y
    X_cal   <- as.matrix(data_cal[,   -ncol(data_cal)])
    Y_cal   <- data_cal$Y
    
    T <- nrow(data_test)
    # 3. Null indices and theta
    Null_cal  <- NullIndex(Y_cal,    Value)
    Null_test <- NullIndex(data_test$Y, Value)
    theta     <- rep(1, T)
    theta[Null_test] <- 0
    
    # 4. Fit model & compute raw scores
    model   <- fitting(algo, X_train, Y_train, lambda)
    W_cal   <- Pred(algo, model, X_cal)  # these are V_i = -ScoreCompute
    W_test  <- Pred(algo, model, as.matrix(data_test[, -ncol(data_test)]))
    
    
    # 5. Compute online *weighted* conformal p-values
    # pval_w <- conformalPvalue_online_weighted_given(
    #   W_cal, W_test,
    #   Null_cal, Value, theta,
    #   w_cal, w_test
    # )
    
    pval_w <- confomalPvalue_online(W_cal, W_test, Null_cal, Value, theta)
    
    plot(pval_w,theta)
    
    summary(pval_w)
    length(pval_w)
    length(theta)
    
    # 6. Apply Online-FDR procedures
    out1 <- Lord_feedback(pval_w, alpha, theta, W0=alpha/2)
    r1   <- CiterionCompute(out1$R, theta, "LF"); r1["prop"]=pi_1;  info <- rbind(info, r1)
    
    out2 <- Lord_feedback_conservative(pval_w, alpha, theta, W0=alpha/2)
    r2   <- CiterionCompute(out2$R, theta, "LFS"); r2["prop"]=pi_1; info <- rbind(info, r2)
    
    out3 <- LORD(pval_w, alpha, version="++")
    r3   <- CiterionCompute(out3$R, theta, "LORD++"); r3["prop"]=pi_1; info <- rbind(info, r3)
    
    out4 <- SAFFRON(pval_w, alpha, lambda=0.5)
    r4   <- CiterionCompute(out4$R, theta, "SAFFRON"); r4["prop"]=pi_1; info <- rbind(info, r4)
    
    out5 <- SAFFRON_feedback(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r5   <- CiterionCompute(out5$R, theta, "SF");   r5["prop"]=pi_1;   info <- rbind(info, r5)
    
    out6 <- SAFFRON_feedback_conservative(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r6   <- CiterionCompute(out6$R, theta, "SFS");   r6["prop"]=pi_1;  info <- rbind(info, r6)
    
    out7 <- LOND(pval_w, alpha)
    r7   <- CiterionCompute(out7$R, theta, "LOND"); r7["prop"]=pi_1;   info <- rbind(info, r7)
    
  }
  return(info)
}  # end foreach


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
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
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


reg_plots_vary_prop_airfoil <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                         font.label = list(size = 20, face = "bold"))
reg_plots_vary_prop_airfoil

pdf(file = "reg_plots_vary_prop_airfoil.pdf",width = 10,height = 4) 
reg_plots_vary_prop_airfoil
dev.off()



#################### compare weighted vs unweighted methods ---------------
w = function(x) {
  exp(x[,c(1,5)] %*% c(2,-3))
}

wsample = function(wts, frac=0.25) {
  n = length(wts)
  i = c()
  while(length(i) <= n*frac) {
    i = c(i, which(runif(n) <= wts/max(wts)))
  }
  return(i)
}

algo<- new("RF") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm
pi1_seq=seq(0.2,0.8,0.1)


nr=500
cl = makeCluster(5)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c(
  "MASS","onlineFDR","randomForest","caret","nnet","glmnet","reshape2",
  "kedd","kernlab","e1071","ks","tidyverse","lsa","magrittr","doSNOW",
  "pbmcapply"
), .options.snow = opts, .errorhandling="remove") %dopar% {
  
  info <- data.frame()
  for (pi_1 in pi1_seq) {
    n = round(N/2)
    i = sample(N,n)
    x = dat.x[i,]; y = dat.y[i]
    x0 = dat.x[-i,]; y0 = dat.y[-i]
    
    # Tilting
    i0 = wsample(w(x0))
    x00 = x0[i0,]; y00 = y0[i0]
    
    # after you compute the raw tilt:
    raw_w <- w(rbind(x, x00))
    
    # split into historical vs test
    n_hist <- nrow(x)
    n_test <- nrow(x00)
    raw_w_cal  <- raw_w[       1:n_hist      ]
    raw_w_test <- raw_w[(n_hist+1):(n_hist+n_test)]
    
    # now rescale each vector to [0,1]
    w_cal  <- (raw_w_cal  - min(raw_w_cal )) / (max(raw_w_cal ) - min(raw_w_cal ))
    w_test <- (raw_w_test - min(raw_w_test)) / (max(raw_w_test) - min(raw_w_test))
    
    # (optional) if you want strictly in (0,1), you can bump by eps:
    eps <- 1e-6
    w_cal  <- pmin(1-eps, pmax(eps, w_cal))
    w_test <- pmin(1-eps, pmax(eps, w_test))
    
    
    
    his_data <- data.frame(X=x,Y=y)
    data_test <- data.frame(X=x00,Y=y00)
    n_cal = round(n/2)
    
    
    
    
    Value=list(type="<=A",v=quantile(data_test$Y,1-pi_1))
    
    # 2. Split history into train & calibration
    datawork   <- DataSplit(his_data, n, 0, n_cal)
    data_train <- datawork$data_train
    data_cal   <- datawork$data_cal
    
    X_train <- as.matrix(data_train[, -ncol(data_train)])
    Y_train <- data_train$Y
    X_cal   <- as.matrix(data_cal[,   -ncol(data_cal)])
    Y_cal   <- data_cal$Y
    
    T <- nrow(data_test)
    # 3. Null indices and theta
    Null_cal  <- NullIndex(Y_cal,    Value)
    Null_test <- NullIndex(data_test$Y, Value)
    theta     <- rep(1, T)
    theta[Null_test] <- 0
    
    # 4. Fit model & compute raw scores
    model   <- fitting(algo, X_train, Y_train, lambda)
    W_cal   <- Pred(algo, model, X_cal)  # these are V_i = -ScoreCompute
    W_test  <- Pred(algo, model, as.matrix(data_test[, -ncol(data_test)]))
    
    
    # 5. Compute online *weighted* conformal p-values
    pval_w <- conformalPvalue_online_weighted_given(
      W_cal, W_test,
      Null_cal, Value, theta,
      w_cal, w_test
    )
    
    pval <- confomalPvalue_online(W_cal, W_test, Null_cal, Value, theta)
    
    plot(pval_w,theta)
    
    summary(pval_w)
    length(pval_w)
    length(theta)
    
    # 6. Apply Online-FDR procedures
    out1 <- Lord_feedback(pval_w, alpha, theta, W0=alpha/2)
    r1   <- CiterionCompute(out1$R, theta, "wLF"); r1["prop"]=pi_1;  info <- rbind(info, r1)
    
    out2 <- Lord_feedback_conservative(pval_w, alpha, theta, W0=alpha/2)
    r2   <- CiterionCompute(out2$R, theta, "wLFS"); r2["prop"]=pi_1; info <- rbind(info, r2)
    
    out3 <- Lord_feedback(pval, alpha, theta, W0=alpha/2)
    r3   <- CiterionCompute(out3$R, theta, "LF"); r3["prop"]=pi_1;  info <- rbind(info, r3)
    
    out4 <- Lord_feedback_conservative(pval, alpha, theta, W0=alpha/2)
    r4   <- CiterionCompute(out4$R, theta, "LFS"); r4["prop"]=pi_1; info <- rbind(info, r4)
    
    
    out5 <- SAFFRON_feedback(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r5   <- CiterionCompute(out5$R, theta, "wSF");   r5["prop"]=pi_1;   info <- rbind(info, r5)
    
    out6 <- SAFFRON_feedback_conservative(pval_w, alpha, theta, w0=alpha/2, lambda=0.5)
    r6   <- CiterionCompute(out6$R, theta, "wSFS");   r6["prop"]=pi_1;  info <- rbind(info, r6)
    
    out7 <- SAFFRON_feedback(pval, alpha, theta, w0=alpha/2, lambda=0.5)
    r7   <- CiterionCompute(out7$R, theta, "SF");   r7["prop"]=pi_1;   info <- rbind(info, r7)
    
    out8 <- SAFFRON_feedback_conservative(pval, alpha, theta, w0=alpha/2, lambda=0.5)
    r8   <- CiterionCompute(out8$R, theta, "SFS");   r8["prop"]=pi_1;  info <- rbind(info, r8)
    
    
  }
  return(info)
}  # end foreach


close(pb)
stopCluster(cl)

pp <- result%>%
  group_by(Method,prop)%>%
  dplyr::summarize(FDR=mean(FDP),POWER=mean(Power),
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power)))
pp

pp$Method=factor(pp$Method,levels =  c("wSFS","wLFS","wSF","wLF",
                                       "SFS","LFS","SF","LF"))


P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2))+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
                                                                         alpha = 0.1,
                                                                         linetype = 1,
                                                                         color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.1), limits = c(0.2, 0.8)) +
  scale_y_continuous(limits = c(0, 0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
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
  theme(legend.position = "bottom") 
P1 <- P1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P1


P2 <- ggplot(data = pp, aes(x = prop, y = POWER, group = Method, color = Method, shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1,2))+
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1,
              linetype = 1,
              color = NA) +
  geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.1), limits = c(0.2, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +
  scale_color_nejm(palette = c("default"), alpha = 0.9) +
  scale_fill_manual(values = c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
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


reg_plots_vary_prop_airfoil <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                         font.label = list(size = 20, face = "bold"))
reg_plots_vary_prop_airfoil

pdf(file = "reg_plots_vary_prop_airfoil.pdf",width = 10,height = 4) 
reg_plots_vary_prop_airfoil
dev.off()