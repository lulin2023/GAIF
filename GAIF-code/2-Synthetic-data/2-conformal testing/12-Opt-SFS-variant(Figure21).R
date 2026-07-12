
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





setwd("C:/Users/LuLin/OneDrive/Online multiple testing with feedback/GAIF codes final/useful functions")

source("functions_OnSel.R")
source("algoclass_OnSel.R")
source("SAFFRON_feedback functions.R")

confomalPvalue_online_random <- function(W_cal_all, W_test_all, Null_cal, Value, theta) {
  n_cal <- nrow(W_cal_all)
  n_test <- nrow(W_test_all)
  K <- ncol(W_cal_all)  
  
  Phi_cal_all <- -apply(W_cal_all, 2, ScoreCompute, Value)
  Phi_test_all <- -apply(W_test_all, 2, ScoreCompute, Value)
  
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  n2 <- nrow(Phi_Null_all)
  
  pvalues <- numeric(n_test)
  
  for (t in 1:n_test) {
    k_opt <- sample(1:K, 1)
    
    pvalues[t] <- sum(Phi_Null_all[, k_opt] <= Phi_test_all[t, k_opt]) / (n2 + 1)
    
    if (theta[t] == 0) {
      new_score <- -ScoreCompute(W_test_all[t, k_opt], Value)
      Phi_Null_all <- rbind(Phi_Null_all, new_score)
      n2 <- nrow(Phi_Null_all)
    }
    print(t)
  }
  return(pvalues)
}





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
  
  # Only consider calibration scores for null samples.
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  n2 <- nrow(Phi_Null_all)
  
  # p-values to be returned
  pvalues <- numeric(n_test)
  
  # EWMA storage for each model's past non-null p-values, initialized as NA
  ewma <- rep(NA, K)
  pval_models <- rep(NA, K)
  
  for (t in 1:n_test) {
    # Compute conformal p-values for all models at time t
    # pval_models <- sapply(1:K, function(k) {
    #   sum(Phi_Null_all[, k] <= Phi_test_all[t, k]) / (n2 + 1)
    # })
    
    start <- max(1, t - L)
    C1t   <- which(theta[start:(t-1)] == 1) + start - 1
    
    if (length(C1t) == 0) {
      k_opt <- sample.int(K, 1)
    } else {
      for (k in seq_len(K)) {
        pval_models[k] <- sapply(C1t, function(j) {
          # Use current Phi_null to compute auxiliary p-value of sample j:
          sum(Phi_Null_all[, k] <= Phi_test_all[j, k]) / (n2 + 1)
        })
        # weights <- lambda ^ (t - 1 - C1t)
        # M[k] <- sum(weights * tilde_p) / sum(weights)
      }
      
      
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
    pvalues[t] <- sum(Phi_Null_all[, k_opt] <= Phi_test_all[t, k_opt]) / (n2 + 1)
    
    # Update EWMA for each model if sample t is non-null
    if (theta[t] != 0) {
      for (k in 1:K) {
        if (is.na(ewma[k])) {
          ewma[k] <- pval_models[k]
        } else {
          ewma[k] <- lambda * ewma[k] + (1 - lambda) * pval_models[k]
        }
      }
    }
    
    # If sample t is null, add its score to the calibration set for future p-value computations
    if (theta[t] == 0) {
      new_score <- -ScoreCompute(W_test_all[t, k_opt], Value)
      Phi_Null_all <- rbind(Phi_Null_all, new_score)
      n2 <- nrow(Phi_Null_all)
    }
    
    cat("t =", t, "optimal model k =", k_opt, "\n")
  }
  
  return(pvalues)
}


#  Exclude t

confomalPvalue_online_opt_nonNull_EWMA_exclude <- function(
    W_cal_all, W_test_all, Null_cal, Value, theta, lambda = 0.9, L = 200
){
  
  n_test <- nrow(W_test_all)
  K <- ncol(W_cal_all)
  
  Phi_cal_all <- -apply(W_cal_all, 2, ScoreCompute, Value)
  Phi_test_all <- -apply(W_test_all, 2, ScoreCompute, Value)
  
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  n2 <- nrow(Phi_Null_all)
  
  pvalues <- numeric(n_test)
  
  ewma <- rep(NA, K)
  pval_models <- rep(NA, K)
  
  for (t in 1:n_test) {
    
    start <- max(1, t - L)
    C1t <- which(theta[start:(t-1)] == 1) + start - 1
    
    if (length(C1t) == 0) {
      k_opt <- sample.int(K, 1)
    } else {
      
      for (k in seq_len(K)) {
        pval_models[k] <- mean(
          sapply(C1t, function(j) {
            sum(Phi_Null_all[, k] <= Phi_test_all[j, k]) / (n2 + 1)
          })
        )
      }
      
      candidate_ewma <- ifelse(is.na(ewma), Inf, ewma)
      k_opt <- which.min(candidate_ewma)
    }
    
    pvalues[t] <- sum(Phi_Null_all[, k_opt] <= Phi_test_all[t, k_opt]) / (n2 + 1)
    
    if (theta[t] != 0) {
      for (k in 1:K) {
        if (is.na(ewma[k])) {
          ewma[k] <- pval_models[k]
        } else {
          ewma[k] <- lambda * ewma[k] + (1 - lambda) * pval_models[k]
        }
      }
    }
    
    if (theta[t] == 0) {
      new_score <- -ScoreCompute(W_test_all[t, k_opt], Value)
      Phi_Null_all <- rbind(Phi_Null_all, new_score)
      n2 <- nrow(Phi_Null_all)
    }
  }
  
  return(pvalues)
}


### Include t + truncation

confomalPvalue_online_opt_nonNull_EWMA_trunc <- function(
    W_cal_all, W_test_all, Null_cal, Value, theta, lambda = 0.9, L = 200
){
  
  n_test <- nrow(W_test_all)
  K <- ncol(W_cal_all)
  
  Phi_cal_all <- -apply(W_cal_all, 2, ScoreCompute, Value)
  Phi_test_all <- -apply(W_test_all, 2, ScoreCompute, Value)
  
  Phi_Null_all <- Phi_cal_all[Null_cal, , drop = FALSE]
  
  pvalues <- numeric(n_test)
  
  ewma <- rep(NA, K)
  pval_models <- rep(NA, K)
  
  for (t in 1:n_test) {
    
    start <- max(1, t - L)
    C1t <- which(theta[start:(t-1)] == 1) + start - 1
    
    if (length(C1t) == 0) {
      k_opt <- sample.int(K, 1)
    } else {
      
      for (k in seq_len(K)) {
        
        scores <- c(Phi_Null_all[,k], Phi_test_all[t,k])
        
        scores_trunc <- scores[-which.min(scores)]
        
        pval_models[k] <- mean(
          sapply(C1t, function(j){
            sum(scores_trunc <= Phi_test_all[j,k])/(length(scores_trunc)+1)
          })
        )
      }
      
      candidate_ewma <- ifelse(is.na(ewma), Inf, ewma)
      k_opt <- which.min(candidate_ewma)
    }
    
    scores <- c(Phi_Null_all[,k_opt], Phi_test_all[t,k_opt])
    scores_trunc <- scores[-which.min(scores)]
    
    pvalues[t] <- sum(scores_trunc <= Phi_test_all[t,k_opt])/(length(scores_trunc)+1)
    
    if (theta[t] != 0) {
      for (k in 1:K) {
        if (is.na(ewma[k])) {
          ewma[k] <- pval_models[k]
        } else {
          ewma[k] <- lambda * ewma[k] + (1 - lambda) * pval_models[k]
        }
      }
    }
    
    if (theta[t] == 0) {
      new_score <- -ScoreCompute(W_test_all[t, k_opt], Value)
      Phi_Null_all <- rbind(Phi_Null_all, new_score)
    }
  }
  
  return(pvalues)
}


library(MASS)
library(randomForest)
library(caret)
library(nnet)
library(glmnet)
library(tidyverse)
library(doParallel)
library(doSNOW)
library(onlineFDR)

############################################################
# data generation
############################################################

mvrnorm_new <- function(theta,mu1,mu2,d){
  data <- mvrnorm(n=2,(1-theta)*mu1+theta*mu2,Sigma=diag(d))[1,]
  return(data)
}

data_gen_cla_shift_smooth <- function(m = 1000,
                                      pattern = "sine",
                                      d = 4,
                                      mu1 = c(2,0,0,0),
                                      mu2 = c(0,0,-2,-2)) {
  
  p <- (sin(100*pi*(1:m)/m)+1)/4
  
  Y <- map_dbl(p, ~sample(c(0,1),1,prob=c(1-.x,.x)))
  
  theta <- Y
  
  X <- matrix(
    unlist(lapply(theta,function(y)
      mvrnorm_new(y,mu1,mu2,d))),
    length(theta),d,byrow=TRUE
  )
  
  data <- as.data.frame(cbind(X,Y))
  names(data)[d+1] <- "y"
  
  return(data)
}


############################################################
# simulation setup
############################################################

alpha <- 0.1
T <- 1000
n <- 1000
nr <- 200

Value=list(type="==A,R",v=0)

algo1 <- new("SVM")
algo2 <- new("RFc")
algo3 <- new("NN")

lambda1 <- 2
lambda2 <- 600

############################################################
# Monte Carlo
############################################################

alpha <- 0.1
T <- 1000
n <- 1000
n_train <- round(n/2)
n_cal <- round(n/2)

nr = 100

cl = makeCluster(10)
registerDoSNOW(cl)

pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS","onlineFDR","randomForest","caret",
                                "nnet","glmnet","reshape2","kedd","kernlab",
                                "e1071","ks","tidyverse","lsa","magrittr",
                                "doSNOW","pbmcapply"),
                  .errorhandling = "remove",
                  .options.snow = opts)%dopar% {
                    
                    info <- data.frame()
                    
                    ####################################################
                    # Generate data
                    ####################################################
                    
                    data <- data_gen_cla_shift_smooth(
                      m=T, pattern="sine", d=4,
                      mu1=c(2,0,0,0),
                      mu2=c(0,0,-2,-2)
                    )
                    
                    his_data <- data_gen_cla_shift_smooth(
                      m=n, pattern="constant", d=4,
                      mu1=c(2,0,0,0),
                      mu2=c(0,0,-2,-2)
                    )
                    
                    p <- ncol(his_data)-1
                    
                    ####################################################
                    # Data split
                    ####################################################
                    
                    datawork = DataSplit(his_data,n,0,n_cal)
                    
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
                    
                    ####################################################
                    # Train models
                    ####################################################
                    
                    ## SVM
                    model1=fitting(algo1,X_train,Y_train,lambda1)
                    W_cal1=Pred(algo1,model1,X_cal)[,2]
                    W_test1=Pred(algo1,model1,X_test)[,2]
                    
                    ## RF
                    model2=fitting(algo2,X_train,Y_train,lambda2)
                    W_cal2=Pred(algo2,model2,X_cal)
                    W_test2=Pred(algo2,model2,X_test)
                    
                    ## NN
                    model3=fitting(algo3,X_train,Y_train,lambda)
                    W_cal3=Pred(algo3,model3,X_cal)
                    W_test3=Pred(algo3,model3,X_test)
                    
                    ####################################################
                    # Combine prediction scores
                    ####################################################
                    
                    W_cal_all <- cbind(W_cal1,W_cal2,W_cal3)
                    W_test_all <- cbind(W_test1,W_test2,W_test3)
                    
                    ####################################################
                    # Conformal p-values
                    ####################################################
                    
                    pvals_random <- confomalPvalue_online_random(
                      W_cal_all, W_test_all, Null_cal, Value, theta
                    )
                    
                    pvals_opt <- confomalPvalue_online_opt_nonNull_EWMA(
                      W_cal_all, W_test_all, Null_cal, Value, theta,
                      lambda=0.9, L=100
                    )
                    
                    pvals_opt_exclude <- confomalPvalue_online_opt_nonNull_EWMA_exclude(
                      W_cal_all, W_test_all, Null_cal, Value, theta,
                      lambda=0.9, L=100
                    )
                    
                    pvals_opt_trunc <- confomalPvalue_online_opt_nonNull_EWMA_trunc(
                      W_cal_all, W_test_all, Null_cal, Value, theta,
                      lambda=0.9, L=100
                    )
                    
                    ####################################################
                    # SAFFRON-feedback
                    ####################################################
                    
                    rej_opt.saf <- SAFFRON_feedback_conservative(
                      pvals_opt, alpha=alpha, theta=theta, w0=alpha/2
                    )
                    
                    rej_opt_exclude.saf <- SAFFRON_feedback_conservative(
                      pvals_opt_exclude, alpha=alpha, theta=theta, w0=alpha/2
                    )
                    
                    rej_opt_trunc.saf <- SAFFRON_feedback_conservative(
                      pvals_opt_trunc, alpha=alpha, theta=theta, w0=alpha/2
                    )
                    
                    rej_random.saf <- SAFFRON_feedback_conservative(
                      pvals_random, alpha=alpha, theta=theta, w0=alpha/2
                    )
                    
                    ####################################################
                    # Online FDP / Power
                    ####################################################
                    
                    res_Opt_SF_online <- 1:T %>%
                      map(~CiterionCompute_each(Alter_test,
                                                decisions=rej_opt.saf$R,
                                                .x)) %>%
                      unlist %>% split(.,names(.))
                    
                    res_OptEx_SF_online <- 1:T %>%
                      map(~CiterionCompute_each(Alter_test,
                                                decisions=rej_opt_exclude.saf$R,
                                                .x)) %>%
                      unlist %>% split(.,names(.))
                    
                    res_OptTr_SF_online <- 1:T %>%
                      map(~CiterionCompute_each(Alter_test,
                                                decisions=rej_opt_trunc.saf$R,
                                                .x)) %>%
                      unlist %>% split(.,names(.))
                    
                    res_Ran_SF_online <- 1:T %>%
                      map(~CiterionCompute_each(Alter_test,
                                                decisions=rej_random.saf$R,
                                                .x)) %>%
                      unlist %>% split(.,names(.))
                    
                    ####################################################
                    # Result lists
                    ####################################################
                    
                    result.SF.opt <- list(
                      FDP=res_Opt_SF_online$FDP,
                      Power=res_Opt_SF_online$Power,
                      time=res_Opt_SF_online$time,
                      method="Opt-SFS"
                    )
                    
                    result.SF.opt_exclude <- list(
                      FDP=res_OptEx_SF_online$FDP,
                      Power=res_OptEx_SF_online$Power,
                      time=res_OptEx_SF_online$time,
                      method="OptEx-SFS"
                    )
                    
                    result.SF.opt_trunc <- list(
                      FDP=res_OptTr_SF_online$FDP,
                      Power=res_OptTr_SF_online$Power,
                      time=res_OptTr_SF_online$time,
                      method="OptTr-SFS"
                    )
                    
                    result.SF.ran <- list(
                      FDP=res_Ran_SF_online$FDP,
                      Power=res_Ran_SF_online$Power,
                      time=res_Ran_SF_online$time,
                      method="Ran-SFS"
                    )
                    
                    info=list(
                      result.SF.opt,
                      result.SF.opt_exclude,
                      result.SF.opt_trunc,
                      result.SF.ran
                    )
                    
                    return(info)
                    
                  }

close(pb)
stopCluster(cl)

result


####################################################
# Reshape results
####################################################

pp <- result %>%
  map_dfr(~{
    tibble(
      FDP = .x$FDP,
      Power = .x$Power,
      Method = .x$method,
      Time = .x$time
    )
  })

pp$Method = factor(
  pp$Method,
  levels = c(
    "Opt-SFS",
    "OptEx-SFS",
    "OptTr-SFS",
    "Ran-SFS"
  )
)

head(pp)
summary(pp)

library(ggplot2)
library(dplyr)
library(patchwork)
library(ggpubr)  
t <- c(seq(10, 500, 50), 500)

method_levels <- c("Opt-SFS", "OptEx-SFS", "OptTr-SFS", "Ran-SFS")

method_colors <- c(
  "#BC3C29", "#0072B5", "#E18727", "#20854E"
)
names(method_colors) <- method_levels

shape_values <- c(21, 22, 23, 24)
names(shape_values) <- method_levels

linetype_values <- c("solid", "solid", "solid", "dashed")
names(linetype_values) <- method_levels

pp$Method <- factor(pp$Method, levels = method_levels)

pp_saffron <- pp %>% filter(Method %in% method_levels)

summary_by_group <- function(data) {
  data %>%
    filter(Time %in% t) %>%
    group_by(Time, Method) %>%
    summarize(
      FDP_avg = mean(FDP, na.rm = TRUE),
      Power_avg = mean(Power, na.rm = TRUE),
      FDP_se = sd(FDP, na.rm = TRUE)/sqrt(n()),
      Power_se = sd(Power, na.rm = TRUE)/sqrt(n()),
      .groups = "drop"
    )
}

pp_saffron_summary <- summary_by_group(pp_saffron)

plot_metric <- function(df, y_avg, y_se, y_label, y_lim, add_alpha_line = FALSE) {
  p <- ggplot(df, aes(x = Time, y = .data[[y_avg]], group = Method)) +
    geom_point(aes(color = Method, fill = Method, shape = Method), size = 2.5) +
    geom_ribbon(aes(ymin = .data[[y_avg]] - .data[[y_se]],
                    ymax = .data[[y_avg]] + .data[[y_se]],
                    fill = Method), alpha = 0.1, color = NA) +
    geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
    scale_color_manual(values = method_colors) +
    scale_fill_manual(values = method_colors) +
    scale_shape_manual(values = shape_values) +
    scale_linetype_manual(values = linetype_values) +
    ylim(y_lim[1], y_lim[2]) +
    xlab("Time") + ylab(y_label) +
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
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      text = element_text(size = 16, family = "serif")
    )
  
  if (add_alpha_line) {
    p <- p + geom_hline(yintercept = alpha, colour = "black", linetype = "dashed")
  }
  
  return(p)
}

p_saf_fdp   <- plot_metric(pp_saffron_summary, "FDP_avg", "FDP_se", "FDR", c(0, 1), add_alpha_line = TRUE)
p_saf_power <- plot_metric(pp_saffron_summary, "Power_avg", "Power_se", "Power", c(0, 1), add_alpha_line = FALSE)

plot_cla_SAFFRON <- ggarrange(p_saf_fdp, p_saf_power, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                              font.label = list(size = 16, face = "bold"))

pdf(file = "plot_cla_Opt_SAFFRON-response.pdf", width = 10, height = 4)
plot_cla_SAFFRON
dev.off()
dev.new()

