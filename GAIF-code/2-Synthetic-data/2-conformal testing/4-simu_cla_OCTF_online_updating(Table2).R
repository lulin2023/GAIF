

########## simulation ##########

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

################################################
############ strategy 1: fixed model ###########
################################################

pvalue_fixed_model <- function(X_train,Y_train,
                               X_cal,X_test,
                               Null_cal,Value,theta,
                               algo,lambda){
  
  model=fitting(algo,X_train,Y_train,lambda)
  
  W_cal=Pred(algo,model,X_cal)
  W_test=Pred(algo,model,X_test)
  
  pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
  
  return(pval)
}

################################################
######## strategy 2: model selection (EWMA) ####
################################################
# confomalPvalue_online_opt_nonNull_EWMA: user's original EWMA-based model
# selection function, kept in its original structure/naming. Three bugs
# fixed inline (see FIX 1/2/3 comments below); no restructuring otherwise.

confomalPvalue_online_opt_nonNull_EWMA <- function(W_cal_all, W_test_all, Null_cal, Value, theta, lambda = 0.9, L = 200) {
  # W_cal_all: n_cal x K matrix of raw predictions on calibration data
  # W_test_all: n_test x K matrix of raw predictions on test data
  # Null_cal: indices of calibration data considered as null
  # Value: parameters for ScoreCompute
  # theta: a vector of length n_test with feedback (theta[t] == 0 means null at time t)
  # lambda: decay parameter for EWMA (0 < lambda < 1)
  
  n_cal <- nrow(W_cal_all)
  n_test <- nrow(W_test_all)
  K <- ncol(W_cal_all)  # number of candidate models
  
  # Compute conformity scores for all models
  Phi_cal_all <- -apply(W_cal_all, 2, ScoreCompute, Value)
  Phi_test_all <- -apply(W_test_all, 2, ScoreCompute, Value)
  
  # DIAGNOSTIC CHECK: apply() silently returns a list (not a matrix) if
  # ScoreCompute doesn't return the same length for every column (e.g. an
  # algo whose Pred() output has a different shape). Surface this clearly
  # instead of failing later with a cryptic dimension-mismatch error.
  if (!is.matrix(Phi_cal_all) || nrow(Phi_cal_all) != n_cal) {
    stop("ScoreCompute did not return a length-n_cal vector for every column of W_cal_all. Check that all K algorithms' Pred() outputs have a consistent shape (e.g. a model returning a 2-column softmax matrix instead of a single score column).")
  }
  if (!is.matrix(Phi_test_all) || nrow(Phi_test_all) != n_test) {
    stop("ScoreCompute did not return a length-n_test vector for every column of W_test_all. Check Pred() output shape consistency across algorithms.")
  }
  
  # Only consider calibration scores for null samples.
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  n2 <- nrow(Phi_Null_all)
  
  # p-values to be returned
  pvalues <- numeric(n_test)
  
  # EWMA storage for each model's past non-null p-values, initialized as NA
  ewma <- rep(NA, K)
  pval_models <- rep(NA, K)
  
  for (t in 1:n_test) {
    
    # FIX 1: handle t = 1 explicitly so start:(t-1) never becomes 1:0
    if (t == 1) {
      C1t <- integer(0)
    } else {
      start <- max(1, t - L)
      C1t   <- which(theta[start:(t-1)] == 1) + start - 1
    }
    
    if (length(C1t) == 0) {
      k_opt <- sample.int(K, 1)
    } else {
      
      # Select optimal model using EWMA of past non-null samples (if available)
      if (t > 1) {
        # Replace NA with Inf so that models without non-null history are not selected
        candidate_ewma <- ifelse(is.na(ewma), Inf, ewma)
        k_opt <- which.min(candidate_ewma)
      } else {
        k_opt <- sample(1:K, 1)  # For the first sample, select a model at random
      }
    }
    
    # Compute p-value using the selected model
    xi = runif(1)
    pvalues[t] <- (sum(Phi_Null_all[, k_opt] < Phi_test_all[t, k_opt]) + xi*(1+sum(Phi_Null_all[, k_opt] == Phi_test_all[t, k_opt]))) / (n2 + 1)
    
    # FIX 2: pval_models[k] must be a single scalar — the auxiliary p-value
    # of the CURRENT point t under model k against the current null pool —
    # not a vector recomputed over the whole window C1t (that vector-into-
    # scalar assignment is exactly what throws "number of items to replace
    # is not a multiple of replacement length").
    if (theta[t] != 0) {
      for (k in 1:K) {
        pval_models[k] <- sum(Phi_Null_all[, k] <= Phi_test_all[t, k]) / (n2 + 1)
        
        if (is.na(ewma[k])) {
          ewma[k] <- pval_models[k]
        } else {
          ewma[k] <- lambda * ewma[k] + (1 - lambda) * pval_models[k]
        }
      }
    }
    
    # FIX 3: when sample t is null, grow ALL K models' null pools using the
    # already-computed Phi_test_all[t, ] row (length K), instead of
    # recomputing a single scalar score for k_opt only and rbind-ing a
    # dimension-mismatched value into a K-column matrix.
    if (theta[t] == 0) {
      Phi_Null_all <- rbind(Phi_Null_all, Phi_test_all[t, ])
      n2 <- nrow(Phi_Null_all)
    }
    
    cat("t =", t, "optimal model k =", k_opt, "\n")
  }
  
  return(pvalues)
}

# Helper: build an n x K score/probability matrix from K pre-trained models,
# one column per algorithm. Needed because confomalPvalue_online_opt_nonNull_EWMA
# expects W_cal_all / W_test_all as matrices, but different algo classes
# (e.g. NN vs RFc vs SVM) may return Pred() outputs with different shapes.
build_W_matrix <- function(algos, models, Xdata){
  cols <- lapply(seq_along(algos), function(k){
    algo_k <- algos[[k]]$algo
    w <- Pred(algo_k, models[[k]], Xdata)
    # If an algo's Pred() returns multiple columns (e.g. NN returning a
    # 2-column softmax), this needs to be resolved to a single column here;
    # otherwise the algorithms cannot be combined into one K-column matrix.
    if (!is.null(dim(w)) && ncol(w) > 1) {
      stop(sprintf("Pred() for algo k=%d ('%s') returned %d columns; need a single score/probability column here. Please specify which column to use.", k, class(algo_k), ncol(w)))
    }
    as.numeric(w)
  })
  do.call(cbind, cols)
}

# Wrapper matching the same call signature used elsewhere in the script
# (X_train, Y_train, X_cal, X_test, Null_cal, Value, theta, algos): pre-trains
# the K candidate models, builds the score matrices, then calls the
# (bug-fixed) EWMA function above.
pvalue_model_selection <- function(X_train,Y_train,
                                   X_cal,X_test,
                                   Null_cal,Value,theta,
                                   algos, lambda=0.9, L=100){
  
  K <- length(algos)
  models <- vector("list", K)
  
  for(k in 1:K){
    algo_k   <- algos[[k]]$algo
    lambda_k <- algos[[k]]$lambda
    models[[k]] <- fitting(algo_k, X_train, Y_train, lambda_k)
  }
  
  W_cal_all  <- build_W_matrix(algos, models, X_cal)
  W_test_all <- build_W_matrix(algos, models, X_test)
  
  confomalPvalue_online_opt_nonNull_EWMA(
    W_cal_all, W_test_all, Null_cal, Value, theta,
    lambda = lambda, L = L)
}

################################################
######## strategy 3: online retraining #########
################################################

pvalue_online_retraining <- function(X_train,Y_train,
                                     X_cal,X_test,Y_test,
                                     Null_cal,Value,theta,
                                     algo=new("RFc"),lambda=500){
  
  Tn=nrow(X_test)
  
  X_train_online=X_train
  Y_train_online=Y_train
  
  model=fitting(algo,X_train_online,Y_train_online,lambda)
  
  W_cal=Pred(algo,model,X_cal)
  Phi_cal=-ScoreCompute(W_cal,Value)
  Phi_Null=Phi_cal[Null_cal]
  
  pval=numeric(Tn)
  
  for(t in 1:Tn){
    
    model=fitting(algo,X_train_online,Y_train_online,lambda)
    
    W_test=Pred(algo,model,matrix(X_test[t,],nrow=1))
    
    Phi_test=-ScoreCompute(W_test,Value)[1]
    
    xi=runif(1)
    
    pval[t]=(sum(Phi_Null < Phi_test)+
               xi*(1+sum(Phi_Null==Phi_test)))/(length(Phi_Null)+1)
    
    if(theta[t]==0){
      Phi_Null=c(Phi_Null,Phi_test)
    }
    
    X_train_online=rbind(X_train_online,X_test[t,])
    Y_train_online=c(Y_train_online,Y_test[t])
  }
  
  return(pval)
}

################################################
############ simulation setting ################
################################################

Tn <- 300
alpha <- 0.2
n <- 2000

n_train<- round(n/2)
n_cal <- round(n/2)

algo<- new("RFc")
lambda<- 500

algos <- list(
  list(algo=new("NN"), lambda=500),
  list(algo=new("RFc"), lambda=500),
  list(algo=new("NN"), lambda=400)
)

K <- length(algos)

Value=list(type="==A,R",v=0)

pi1_seq=0.5

nr=500

# EWMA model-selection tuning parameters
ewma_lambda <- 0.9
ewma_L      <- 100

################################################
############ parallel setting ##################
################################################

cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

################################################
############ simulation loop ###################
################################################

result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS","onlineFDR","randomForest",
                                "caret","nnet","glmnet","reshape2",
                                "kedd","kernlab","e1071","ks",
                                "tidyverse","lsa","magrittr",
                                "doSNOW","pbmcapply"),
                  .options.snow = opts)%dopar% {
                    
                    info<-data.frame()
                    
                    for (pi_1 in pi1_seq) {
                      
                    
                      data <- data_gen_cla_shift_smooth(m=Tn,pattern = "sine",d=4,mu1=c(3,0,0,0),mu2 = c(0,0,-1,-1))
                      # generate history data and estimate K (diversity threshold)---
                      his_data <- data_gen_cla_shift_smooth(m=n,pattern = "constant",d=4,mu1=c(3,0,0,0),mu2 = c(0,0,-1,-1))

                      p <- ncol(his_data)-1
                      
                      datawork=DataSplit(his_data,n,0,n_cal)
                      
                      data_train=datawork$data_train
                      data_cal=datawork$data_cal
                      
                      Null_cal=NullIndex(data_cal$y,Value)
                      
                      X_train=as.matrix(data_train[,-(p+1)])
                      Y_train=as.matrix(data_train$y)
                      
                      X_cal=as.matrix(data_cal[,-(p+1)])
                      Y_cal=as.matrix(data_cal$y)
                      
                      data_test=data
                      
                      Null_test=NullIndex(data_test$y,Value)
                      
                      X_test=as.matrix(data_test[,-(p+1)])
                      Y_test=as.matrix(data_test$y)
                      
                      theta <- rep(1,Tn)
                      theta[Null_test]=0
                      
                      ################################################
                      ############ strategy 1 ########################
                      ################################################
                      
                      start=Sys.time()
                      
                      pval <- pvalue_fixed_model(
                        X_train,Y_train,
                        X_cal,X_test,
                        Null_cal,Value,theta,
                        algo,lambda)
                      
                      runtime=as.numeric(difftime(Sys.time(),start,units="secs"))
                      
                      rej=SAFFRON_feedback(
                        pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      
                      res=CiterionCompute(rej$R,theta,"Fixed")
                      res["prop"]=pi_1
                      res["time"]=runtime
                      
                      info=rbind(info,res)
                      
                      ################################################
                      ############ strategy 2 (EWMA model selection) #
                      ################################################
                      
                      start=Sys.time()
                      
                      pval <- pvalue_model_selection(
                        X_train,Y_train,
                        X_cal,X_test,
                        Null_cal,Value,theta,
                        algos,lambda=ewma_lambda,L=ewma_L)
                      
                      runtime=as.numeric(difftime(Sys.time(),start,units="secs"))
                      
                      rej=SAFFRON_feedback(
                        pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      
                      res=CiterionCompute(rej$R,theta,"ModelSelection")
                      res["prop"]=pi_1
                      res["time"]=runtime
                      
                      info=rbind(info,res)
                      
                      ################################################
                      ############ strategy 3 ########################
                      ################################################
                      
                      start=Sys.time()
                      
                      pval <- pvalue_online_retraining(
                        X_train,Y_train,
                        X_cal,X_test,Y_test,
                        Null_cal,Value,theta,
                        algo,lambda)
                      
                      runtime=as.numeric(difftime(Sys.time(),start,units="secs"))
                      
                      rej=SAFFRON_feedback(
                        pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      
                      res=CiterionCompute(rej$R,theta,"OnlineRetrain")
                      res["prop"]=pi_1
                      res["time"]=runtime
                      
                      info=rbind(info,res)
                      
                    }
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)

################################################
############ summary table #####################
################################################

summary_table <- result %>%
  group_by(Method) %>%
  summarise(
    n         = sum(!is.na(FDP)),
    FDR       = mean(FDP, na.rm = TRUE),
    FDR_se    = sd(FDP, na.rm = TRUE)/sqrt(n),
    power     = mean(Power, na.rm = TRUE),
    power_se  = sd(Power, na.rm = TRUE)/sqrt(n),
    Runtime      = mean(time, na.rm = TRUE),
    Runtime_se   = sd(time, na.rm = TRUE)/sqrt(n)
  )

print(summary_table)


