

####### OCTF-delay ########

theta_wealth <- function(theta, alpha_t, gamma_t, i){
  
  idx <- which(theta[1:i] == 1)
  
  if(length(idx) == 0) return(0)
  
  lag <- i - idx
  
  valid <- lag > 0
  
  sum(alpha_t[idx[valid]] * gamma_t[lag[valid]])
}


Lord_feedback<-function(pval,alpha,theta,W0,gamma_t){
  N=length(pval)
  decision=rep(0,N)
  gamma_t=sapply(1:N,function(j){1/j^(1.6)})*0.4374901658
  alpha_t=c(W0*gamma_t,0)
  
  
  for (i in 1:N) {
    if(pval[i]<=alpha_t[i]){
      decision[i]=1
    }
    ## updating alpha
    
    
    if(sum(decision)==0){
      alpha_t[i+1]=W0*gamma_t[i+1]+theta_wealth(theta,alpha_t,gamma_t,i)
    }  
    if(sum(decision)==1){
      tau1=which(decision==1)
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0)*gamma_t[i+1-tau1]+theta_wealth(theta,alpha_t,gamma_t,i)
    }
    if(sum(decision)>1){
      tau_seq=which(decision==1)
      tau_seq=tau_seq[2:(length(tau_seq))]
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0+alpha_t[tau1]*theta[tau1])*gamma_t[i+1-tau1]+
        sum(sapply(tau_seq, function(tau_j){
          (alpha)*gamma_t[i+1-tau_j]
        }))+theta_wealth(theta,alpha_t,gamma_t,i)
    }
  }
  
  
  return(data.frame(pval=pval,alphai=alpha_t[-(N+1)],R=decision))
}


# conservative LORD-F adaptation
Lord_feedback_conservative <- function(pval, alpha, theta, W0) {
  N <- length(pval)
  gamma_t <- sapply(1:N, function(j) 1 / (j^1.6)) * 0.4374901658
  decision <- integer(N)
  alpha_t <- numeric(N + 1)
  alpha_t[1] <- W0 * gamma_t[1]
  
  for (i in seq_len(N)) {
    if (pval[i] <= alpha_t[i]) decision[i] <- 1
    null_times <- which(decision[1:i] == 1 & theta[1:i] == 0)
    term1 <- W0 * gamma_t[i + 1]
    term2 <- term3 <- 0
    if (length(null_times) >= 1) {
      tau1_tilde <- null_times[1]
      term2 <- (alpha - W0) * gamma_t[(i + 1) - tau1_tilde]
      if (length(null_times) > 1) term3 <- alpha * sum(gamma_t[(i + 1) - null_times[-1]])
    }
    term4 <- theta_wealth(theta, alpha_t, gamma_t, i)
    alpha_t[i + 1] <- term1 + term2 + term3 + term4
  }
  
  data.frame(pval = pval, alphai = alpha_t[-(N + 1)], R = decision)
}


#####  2025-12-07 update ###

theta_wealth_bandit <- function(theta, decision,alpha_t, gamma_t, i){
  
  idx <- which(theta[1:i]*decision[1:i] == 1)
  
  if(length(idx) == 0) return(0)
  
  lag <- i - idx
  
  valid <- lag > 0
  
  sum(alpha_t[idx[valid]] * gamma_t[lag[valid]])
}

Lord_feedback_bandit<-function(pval,alpha,theta,W0){
  N=length(pval)
  decision=rep(0,N)
  gamma_t=sapply(1:N,function(j){1/j^(1.6)})*0.4374901658
  alpha_t=c(W0*gamma_t,0)
  
  
  for (i in 1:N) {
    if(pval[i]<=alpha_t[i]){
      decision[i]=1
    }
    ## updating alpha
    
    
    if(sum(decision)==0){
      alpha_t[i+1]=W0*gamma_t[i+1]+theta_wealth(theta,alpha_t,gamma_t,i)
    }  
    if(sum(decision)==1){
      tau1=which(decision==1)
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0)*gamma_t[i+1-tau1]+theta_wealth_bandit(theta,decision,alpha_t,gamma_t,i)
    }
    if(sum(decision)>1){
      tau_seq=which(decision==1)
      tau_seq=tau_seq[2:(length(tau_seq))]
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0+alpha_t[tau1]*theta[tau1])*gamma_t[i+1-tau1]+
        sum(sapply(tau_seq, function(tau_j){
          (alpha)*gamma_t[i+1-tau_j]
        }))+theta_wealth_bandit(theta,decision,alpha_t,gamma_t,i)
    }
  }
  
  
  return(data.frame(pval=pval,alphai=alpha_t[-(N+1)],R=decision))
}


################################################################
# OCTF-delay via (d+1) independent sub-streams, matching Algorithm
# "Online conformal testing with delayed feedback (OCTF-delay)".
################################################################

run_substream_GAIF <- function(pval, theta, alpha, d, s0, base_fun, ...) {
  
  N <- length(pval)
  sub_id <- ((seq_len(N) - 1) %% (d + 1)) + 1
  alpha_sub <- alpha / (d + 1)
  
  if (s0 > alpha_sub) {
    stop(sprintf(
      "s0 (%.4f) must be <= alpha/(d+1) (%.4f) for each sub-stream's GAIF procedure.",
      s0, alpha_sub))
  }
  
  R       <- integer(N)
  alphai  <- numeric(N)
  
  for (j in seq_len(d + 1)) {
    idx_j <- which(sub_id == j)
    if (length(idx_j) == 0) next
    
    out_j <- base_fun(pval[idx_j], alpha_sub, theta[idx_j], s0, ...)
    
    R[idx_j]      <- out_j$R
    alphai[idx_j] <- out_j$alphai
  }
  
  data.frame(pval = pval, alphai = alphai, R = R)
}

LF_delayed <- function(pval, alpha, theta, W0, d) {
  alpha_sub <- alpha / (d + 1)
  s0 <- if (missing(W0)) alpha_sub / 2 else W0
  run_substream_GAIF(pval, theta, alpha, d, s0, Lord_feedback)
}

LFS_delayed <- function(pval, alpha, theta, W0, d) {
  alpha_sub <- alpha / (d + 1)
  s0 <- if (missing(W0)) alpha_sub / 2 else W0
  run_substream_GAIF(pval, theta, alpha, d, s0, Lord_feedback_conservative)
}

theta_wealth_SF <- function(theta, alphai, gammai, cand, i) {
  idx <- which(theta[1:i] == 1 & cand[1:i] == 0)
  
  if(length(idx) == 0) return(0)
  
  cand_cumsum <- cumsum(cand[1:i])
  
  lag <- (i + 1 - idx) - (cand_cumsum[i] - cand_cumsum[idx])
  
  valid <- lag > 0
  sum(alphai[idx[valid]] * gammai[lag[valid]])
}

theta_wealth_SF_bandit <- function(theta, decision,alphai, gammai, cand, i) {
  idx <- which(theta[1:i] == 1 & cand[1:i] == 0 & decision[1:i]==1)
  
  if(length(idx) == 0) return(0)
  
  cand_cumsum <- cumsum(cand[1:i])
  
  lag <- (i + 1 - idx) - (cand_cumsum[i] - cand_cumsum[idx])
  
  valid <- lag > 0
  sum(alphai[idx[valid]] * gammai[lag[valid]])
}

SAFFRON_feedback <- function(pval, alpha = 0.05, theta, gammai, w0, lambda = 0.5, 
                             random = TRUE, display_progress = FALSE) {
  
  N <- length(pval)
  
  
  if (missing(gammai)) {
    gammai <- 0.4374901658 / (seq_len(N)^(1.6))
  } 
  
  if (missing(w0)) {
    w0 <- alpha / 2
  } 
  
  saffron_faster_feedback_R <- function(pval, gammai, lambda, alpha, w0, theta) {
    N <- length(pval)
    alphai <- numeric(N)
    Rvec <- logical(N)
    
    alphai[1] <- min((1 - lambda) * w0 * gammai[1], lambda)
    Rvec[1] <- (pval[1] <= alphai[1])
    
    candsum <- 0
    cand <- integer(N)
    Cjplus <- integer(N)
    tau <- integer(0)
    
    K <- if (Rvec[1]) 1 else 0
    
    for (i in 2:N) {
      cand[i - 1] <- as.integer(pval[i - 1] <= lambda)
      candsum <- candsum + cand[i - 1]
      
      if (K > 1) {
        if (Rvec[i - 1]) tau <- c(tau, i - 1)
        
        Cjplussum <- 0
        if ((K - 1) >= 1) {
          for (j in 1:(K - 1)) {
            Cjplus[j] <- Cjplus[j] + cand[i - 1]
            idx <- i - tau[j] - Cjplus[j]
            if (idx < 1) idx <- 1
            Cjplussum <- Cjplussum + gammai[idx]
          }
        }
        Cjplus[K] <- 0
        low <- tau[K] + 1
        high <- max(i - 1, tau[K] + 1)
        if (low <= high) Cjplus[K] <- sum(cand[low:high])
        
        idx1 <- i - tau[K] - Cjplus[K]
        idx2 <- i - tau[1] - Cjplus[1]
        if (idx1 < 1) idx1 <- 1
        if (idx2 < 1) idx2 <- 1
        Cjplussum <- Cjplussum + gammai[idx1] - gammai[idx2]
        
        alphaitilde <- (1 - lambda) * (w0 * gammai[max(i - candsum, 1)] +
                                         (alpha - w0) * gammai[max(i - tau[1] - Cjplus[1], 1)] +
                                         alpha * Cjplussum)
        
      } else if (K == 1) {
        if (Rvec[i - 1]) tau[1] <- i - 1
        
        low <- tau[1] + 1
        high <- max(i - 1, tau[1] + 1)
        if (low <= high) {
          Cjplus[1] <- sum(cand[low:high])
        } else {
          Cjplus[1] <- 0
        }
        
        alphaitilde <- (1 - lambda) * (w0 * gammai[max(i - candsum, 1)] +
                                         (alpha - w0) * gammai[max(i - tau[1] - Cjplus[1], 1)])
      } else {
        alphaitilde <- (1 - lambda) * w0 * gammai[max(i - candsum, 1)]
      }
      
      feedback_bonus <- theta_wealth_SF(theta, alphai, gammai, cand, i - 1)
      
      alphaitilde <- alphaitilde + feedback_bonus
      alphai[i] <- min(lambda, alphaitilde)
      
      if (pval[i] <= alphai[i]) {
        Rvec[i] <- TRUE
        K <- K + 1
      }
    }
    
    data.frame(pval = pval, alphai = alphai, R = as.numeric(Rvec))
  }
  
  out <- saffron_faster_feedback_R(pval, gammai, lambda, alpha, w0, theta)
  out
}



SAFFRON_feedback_conservative <- function(
    pval,
    alpha = 0.05,
    theta,
    gammai,
    w0,
    lambda = 0.5,
    random = TRUE,
    display_progress = FALSE) {
  
  N <- length(pval)
  
  
  
  if (missing(gammai)) {
    gammai <- 0.4374901658 / (seq_len(N)^1.6)
  } 
  
  if (missing(w0)) {
    w0 <- alpha / 2
  } 
  
  saffron_feedback_conservative_R <- function(
    pval, gammai, lambda, alpha, w0, theta) {
    
    N <- length(pval)
    
    alphai <- numeric(N)
    Rvec   <- logical(N)
    
    alphai[1] <- min((1 - lambda) * w0 * gammai[1], lambda)
    Rvec[1]   <- (pval[1] <= alphai[1])
    
    candsum <- 0
    cand <- integer(N)
    Cjplus <- integer(N)
    
    for (i in 2:N) {
      
      cand[i - 1] <- as.integer(pval[i - 1] <= lambda)
      candsum <- candsum + cand[i - 1]
      
      tautilde <- which(Rvec[1:(i - 1)] &
                          (theta[1:(i - 1)] == 0))
      
      K <- length(tautilde)
      
      if (K > 1) {
        
        Cjplussum <- 0
        
        for (j in 1:(K - 1)) {
          
          low_j  <- tautilde[j] + 1
          high_j <- i - 1
          
          if (low_j <= high_j) {
            Cjplus[j] <- sum(cand[low_j:high_j])
          } else {
            Cjplus[j] <- 0
          }
          
          idx <- i - tautilde[j] - Cjplus[j]
          idx <- max(idx, 1)
          
          Cjplussum <- Cjplussum + gammai[idx]
        }
        
        lowK  <- tautilde[K] + 1
        highK <- i - 1
        
        if (lowK <= highK) {
          Cjplus[K] <- sum(cand[lowK:highK])
        } else {
          Cjplus[K] <- 0
        }
        
        idx1 <- max(i - tautilde[K] - Cjplus[K], 1)
        idx2 <- max(i - tautilde[1] - Cjplus[1], 1)
        
        Cjplussum <- Cjplussum +
          gammai[idx1] -
          gammai[idx2]
        
        alphaitilde <-
          (1 - lambda) *
          (
            w0 * gammai[max(i - candsum, 1)] +
              (alpha - w0) *
              gammai[max(i - tautilde[1] - Cjplus[1], 1)] +
              alpha * Cjplussum
          )
        
      } else if (K == 1) {
        
        low <- tautilde[1] + 1
        high <- i - 1
        
        if (low <= high) {
          Cjplus[1] <- sum(cand[low:high])
        } else {
          Cjplus[1] <- 0
        }
        
        alphaitilde <-
          (1 - lambda) *
          (
            w0 * gammai[max(i - candsum, 1)] +
              (alpha - w0) *
              gammai[max(i - tautilde[1] - Cjplus[1], 1)]
          )
        
      } else {
        
        alphaitilde <-
          (1 - lambda) *
          w0 * gammai[max(i - candsum, 1)]
      }
      
      feedback_bonus <-
        theta_wealth_SF(
          theta,
          alphai,
          gammai,
          cand,
          i - 1
        )
      
      alphaitilde <- alphaitilde + feedback_bonus
      
      alphai[i] <- min(lambda, alphaitilde)
      
      if (pval[i] <= alphai[i]) {
        Rvec[i] <- TRUE
      }
    }
    
    data.frame(
      pval = pval,
      alphai = alphai,
      R = as.numeric(Rvec)
    )
  }
  
  out <- saffron_feedback_conservative_R(
    pval,
    gammai,
    lambda,
    alpha,
    w0,
    theta
  )
  
  out
}


################################################################
# SF-delay / SFS-delay via (d+1) independent sub-streams.
################################################################

SF_delayed <- function(pval, alpha = 0.05, theta, w0, lambda = 0.5, d) {
  alpha_sub <- alpha / (d + 1)
  s0 <- if (missing(w0)) alpha_sub / 2 else w0
  
  base_fun <- function(pval_j, alpha_j, theta_j, s0_j, lambda) {
    SAFFRON_feedback(pval_j, alpha = alpha_j, theta = theta_j, w0 = s0_j, lambda = lambda)
  }
  
  run_substream_GAIF(pval, theta, alpha, d, s0, base_fun, lambda = lambda)
}

SFS_delayed <- function(pval, alpha = 0.05, theta, w0, lambda = 0.5, d) {
  alpha_sub <- alpha / (d + 1)
  s0 <- if (missing(w0)) alpha_sub / 2 else w0
  
  base_fun <- function(pval_j, alpha_j, theta_j, s0_j, lambda) {
    SAFFRON_feedback_conservative(pval_j, alpha = alpha_j, theta = theta_j, w0 = s0_j, lambda = lambda)
  }
  
  run_substream_GAIF(pval, theta, alpha, d, s0, base_fun, lambda = lambda)
}


########## simulation 

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
source("LORD_feedback_functions.R")     # defines run_substream_GAIF(), Lord_feedback*, Lord_feedback_*_delayed
source("SAFFRON_feedback functions.R")  # defines SAFFRON_feedback*, SAFFRON_feedback_*_delayed


# Computes conformal p-values for OCTF-delay: at each global time t,
# p_t is computed using the CALIBRATION SET OF t'S OWN SUB-STREAM
# (C_{0t}^j in the algorithm), which grows only via feedback releases
# that have already occurred strictly before t.
conformalPvalue_online_delay <- function(
    W_cal, W_test, Null_cal, Value,
    theta, d
) {
  Phi_cal  <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Tn <- length(Phi_test)
  
  sub_id <- ((seq_len(Tn) - 1) %% (d + 1)) + 1
  
  Phi_Null <- vector("list", d + 1)
  for (j in 1:(d + 1)) {
    Phi_Null[[j]] <- Phi_cal[Null_cal]
  }
  
  pvalues <- numeric(Tn)
  
  pending <- vector("list", d + 1)
  
  for (t in seq_len(Tn)) {
    j <- sub_id[t]
    Phi_Null_j <- Phi_Null[[j]]
    n_j <- length(Phi_Null_j)
    
    xi <- runif(1)
    pvalues[t] <- (
      sum(Phi_Null_j < Phi_test[t]) +
        xi * (1+sum(Phi_Null_j == Phi_test[t]))
    ) / (n_j + 1)
    
    pending[[j]] <- c(pending[[j]], t)
    
    for (i in pending[[j]]) {
      if (i + d + 1 == t && theta[i] == 0) {
        Phi_Null[[j]] <- c(Phi_Null[[j]], Phi_test[i])
      }
    }
    
    pending[[j]] <- pending[[j]][pending[[j]] + d + 1 > t]
  }
  
  return(pvalues)
}




Tn <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 4000 # number of historical data
n_train<- round(n/2) # number of data used for training model
n_cal <- round(n/2)

algo<- new("NN") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm
Value=list(type="==A,R",v=0)


################## different non-null proportions ------------
alpha <- 0.2
Tn <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=500
cl = makeCluster(5)
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
                      
                      data <- data_generation_classication1(N=Tn,mu1= c(1,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=pi_1,pi=pi_1)
                      
                      his_data <- data_generation_classication1(N=n,mu1= c(1,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=pi_1,pi=pi_1)
                      
                      p <- ncol(his_data)-1
                      
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
                      
                      theta <- rep(1,Tn)
                      theta[Null_test]=0
                      
                      model=fitting(algo,X_train,Y_train,lambda)
                      W_cal=Pred(algo,model,X_cal)
                      W_test=Pred(algo,model,X_test)
                      
                      pval <- conformalPvalue_online_delay(W_cal,W_test,Null_cal,Value,theta,d=3)
                      
                      # NOTE: W0/w0 below is now interpreted as the PER-SUB-STREAM
                      # initial wealth s0 directly (must satisfy s0 <= alpha/(d+1)).
                      d_val <- 5
                      
                      ################################################
                      # Delay-respecting benchmarks: LF-delay, LFS-delay,
                      # SF-delay, SFS-delay (independent sub-streams,
                      # each at level alpha/(d+1); has finite-sample
                      # mFDR guarantee under the delay structure).
                      ################################################
                      
                      rej_feedback_LORD=LF_delayed(pval,alpha,theta,W0=alpha/(2*(d_val+1)),d=d_val)
                      res <- CiterionCompute(rej_feedback_LORD$R,theta,"LF-sub")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_LORD_conse=LFS_delayed(pval,alpha,theta,W0=alpha/(2*(d_val+1)),d=d_val)
                      res <- CiterionCompute(rej_feedback_LORD_conse$R,theta,"LFS-sub")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_SA=SF_delayed(pval,alpha=alpha,theta=theta,w0=alpha/(2*(d_val+1)),lambda=0.5,d=d_val)
                      res <- CiterionCompute(rej_feedback_SA$R,theta,"SF-sub")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      d_val_SFS <- 10
                      rej_feedback_SA_conse=SFS_delayed(pval,alpha,theta,w0=alpha/(2*(d_val_SFS+1)),lambda=0.5,d=d_val_SFS)
                      res <- CiterionCompute(rej_feedback_SA_conse$R,theta,"SFS-sub")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      
                      rej_feedback_LORD_nodelay=Lord_feedback_delayed(pval,alpha,theta,W0=alpha/2,d=5)
                      res <- CiterionCompute(rej_feedback_LORD_nodelay$R,theta,"LF-FD")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_LORD_conse_nodelay=Lord_feedback_conservative_delayed(pval,alpha,theta,W0=alpha/2,d=5)
                      res <- CiterionCompute(rej_feedback_LORD_conse_nodelay$R,theta,"LFS-FD")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_SA_nodelay=SAFFRON_feedback_delayed(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5,d=5)
                      res <- CiterionCompute(rej_feedback_SA_nodelay$R,theta,"SF-FD")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_SA_conse_nodelay=SAFFRON_feedback_conservative_delayed(pval,alpha,theta,w0=alpha/2,lambda=0.5,d=5)
                      res <- CiterionCompute(rej_feedback_SA_conse_nodelay$R,theta,"SFS-FD")
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

pp$Method=factor(pp$Method,levels =  c("SF-sub","LF-sub","SFS-sub","LFS-sub",
                                       "SF-FD","LF-FD","SFS-FD","LFS-FD"))


P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1, 2))+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
                                                                          alpha = 0.1,
                                                                          linetype = 1,
                                                                          color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC", "#7876B1CC",
    "#6F99ADCC", "#FFDC91CC", "#EE4C97CC", "#8F786BCC"
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
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1, 2))+
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
    "#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC", "#7876B1CC",
    "#6F99ADCC", "#FFDC91CC", "#EE4C97CC", "#8F786BCC"
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

cla_plots_vary_prop_delayed <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                         font.label = list(size = 20, face = "bold"))

pdf(file = "cla_plots_vary_prop_delayed.pdf",width = 10,height = 4) 
cla_plots_vary_prop_delayed
dev.off()
