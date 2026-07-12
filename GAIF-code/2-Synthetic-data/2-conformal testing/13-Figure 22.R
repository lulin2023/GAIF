########## Response to Comment 3: LFS/SFS vs. alternative finite-sample-valid baselines
##
## compare four benchmarks：
##   (1) LOND      + classic (offline, fixed calibration set) conformal p-values
##   (2) e-LOND     + online conformal e-values (online updated calibration set theta)
##   (3) LFS  (Lord_feedback_conservative)  + online conformal p-values
##   (4) SFS  (SAFFRON_feedback_conservative) + online conformal p-values
##

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


################################################################
## 1. online conformal p-value
##    work with LFS / SFS 
################################################################

confomalPvalue_online <- function(W_cal, W_test, Null_cal, Value, theta) {
  Phi_cal <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  n2 <- length(Phi_Null)
  
  pvalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    xi <- runif(1)
    pvalues[i] <- (sum(Phi_Null < t) + xi *(1+ sum(Phi_Null == t))) / (n2 + 1)
    
    if (theta[i] == 0) {
      new_score <- -ScoreCompute(W_test[i], Value)
      Phi_Null <- c(Phi_Null, new_score)
      n2 <- length(Phi_Null)
    }
  }
  
  return(pvalues)
}


################################################################
## 2. conformal p-value with fixed calibration set
##    work with "LOND + classic conformal p-values" 
################################################################

conformalPvalue_classic <- function(W_cal, W_test, Null_cal, Value) {
  Phi_cal  <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  n2 <- length(Phi_Null)
  
  pvalues <- numeric(length(Phi_test))
  for (i in seq_along(Phi_test)) {
    t  <- Phi_test[i]
    xi <- runif(1)
    pvalues[i] <- (sum(Phi_Null < t) + xi *(1+ sum(Phi_Null == t))) / (n2 + 1)
  }
  return(pvalues)
}


################################################################
## 3. online conformal e-value
##    work with "e-LOND + conformal e-values" 
##
##      E_t = (n_t + 1)/k_t * 1{ rank_t <= k_t }
##
##  alpha_t^{e-LOND} := alpha * gamma_t * (|R_{t-1}^{e-LOND}| + 1)
################################################################

conformalEvalue_online <- function(W_cal, W_test, Null_cal, Value, theta,
                                   topk_frac = 0.05) {
  Phi_cal  <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  Phi_Null <- Phi_cal[Null_cal]
  n2 <- length(Phi_Null)
  
  evalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    
    k_t <- max(1, floor(topk_frac * (n2 + 1)))
    
    n_greater <- sum(Phi_Null > t)
    n_equal   <- sum(Phi_Null == t)
    
    if (n_equal > 0) {
      rank_t <- n_greater + sample.int(n_equal + 1, 1)
    } else {
      rank_t <- n_greater + 1
    }
    
    evalues[i] <- (n2 + 1) / k_t * as.numeric(rank_t <= k_t)
    
    if (theta[i] == 0) {
      new_score <- -ScoreCompute(W_test[i], Value)
      Phi_Null <- c(Phi_Null, new_score)
      n2 <- length(Phi_Null)
    }
  }
  
  return(evalues)
}

# conformalEvalue_online <- function(
    #     W_cal,
#     W_test,
#     Null_cal,
#     Value,
#     theta,
#     tau = 0.01
# ) {
#   
#   Phi_cal  <- -ScoreCompute(W_cal, Value)
#   Phi_test <- -ScoreCompute(W_test, Value)
#   
#   Phi_Null <- Phi_cal[Null_cal]
#   n2 <- length(Phi_Null)
#   
#   evalues <- numeric(length(Phi_test))
#   
#   for (i in seq_along(Phi_test)) {
#     
#     t <- Phi_test[i]
#     
#     ngreater <- sum(Phi_Null > t)
#     nequal   <- sum(Phi_Null == t)
#     
#     U <- runif(1)
#     
#     pval <- (ngreater + U * nequal + 1) /
#       (n2 + 1)
#     
#     evalues[i] <- as.numeric(pval <= tau) / tau
#     
#     if (theta[i] == 0) {
#       
#       Phi_Null <- c(
#         Phi_Null,
#         -ScoreCompute(W_test[i], Value)
#       )
#       
#       n2 <- length(Phi_Null)
#       
#     }
#   }
#   
#   return(evalues)
# }

eLOND <- function(evalue, alpha, gammai) {
  N <- length(evalue)
  
  if (missing(gammai)) {
    idx <- seq_len(N)
    # the same discounting sequence gamma_t = betai/alpha as onlineFDR::LOND()  
    gammai <- 0.07720838 * log(pmax(idx, 2)) / (idx * exp(sqrt(log(idx))))
  }
  
  alphai <- numeric(N)
  R <- numeric(N)
  D <- 0
  
  for (i in seq_len(N)) {
    alphai[i] <- alpha * gammai[i] * (D + 1)
    if (alphai[i] > 0 && evalue[i] >= 1 / alphai[i]) {
      R[i] <- 1
      D <- D + 1
    }
  }
  
  return(list(alphai = alphai, R = as.numeric(R)))
}


T <- 1000  # number of total time points

# simulation setting-----
alpha  <- 0.1  # significance level
pi     <- 0.5  # Bernoulli(pi)
n      <- 1000 # number of historical data
n_train <- round(n / 2)
n_cal   <- 50

algo<- new("SVM") #algorithm used for classification or regression
lambda<- 2 #specific parameter for the algorithm
Value=list(type="==A,R",v=0)



# tok-k proportion of e-value
topk_frac <- 0.3


################## 改变不同的 non-null proportion ------------
pi1_seq <- seq(0.1, 0.8, 0.1)

nr <- 500
cl <- makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS", "onlineFDR", "randomForest", "caret", "nnet",
                                "glmnet", "reshape2", "kedd", "kernlab", "e1071", "ks",
                                "tidyverse", "lsa", "magrittr", "doSNOW", "pbmcapply"),
                  .errorhandling = "remove", .options.snow = opts) %dopar% {
                    
                    info <- data.frame()
                    
                    for (pi_1 in pi1_seq) {
                      
                      data <- data_generation_classication1(
                        N = T,
                        mu1 = c(5, 0, 0, 0),
                        mu2 = c(0, 0, -4, -4),
                        p = 4,
                        propotion = pi_1,
                        pi = pi_1
                      )
                      
                      
                      # generate training data: mixed distribution, depends on pi_1
                      data_train <- data_generation_classication1(
                        N = n - n_cal,
                        mu1 = c(5, 0, 0, 0),
                        mu2 = c(0, 0, -4, -4),
                        p = 4,
                        propotion = pi_1,
                        pi = pi_1
                      )
                      
                      # generate calibration data: pure null, independent of pi_1
                      data_cal <- data_generation_classication1(
                        N = n_cal,
                        mu1 = c(5, 0, 0, 0),
                        mu2 = c(0, 0, -4, -4),
                        p = 4,
                        propotion = 0,
                        pi = 0
                      )
                      
                      p <- ncol(data_train) - 1
                      
                      # since data_cal is pure null, all calibration indices are null
                      Null_cal <- seq_len(nrow(data_cal))
                      
                      X_train <- as.matrix(data_train[colnames(data_train)[-p - 1]])
                      Y_train <- as.matrix(data_train$y)
                      X_cal   <- as.matrix(data_cal[colnames(data_cal)[-p - 1]])
                      Y_cal   <- as.matrix(data_cal$y)
                      
                      data_test  <- data
                      Null_test  <- NullIndex(data_test$y, Value)
                      Alter_test <- setdiff(1:length(data_test$y), Null_test)
                      X_test     <- as.matrix(data_test[colnames(data_test)[-p - 1]])
                      Y_test     <- as.matrix(data_test$y)
                      
                      theta <- rep(1, length(Y_test))
                      theta[Null_test] <- 0
                      
                      # model and prediction -----
                      model  <- fitting(algo, X_train, Y_train, lambda)
                      W_cal  <- Pred(algo, model, X_cal)[,2]
                      W_test <- Pred(algo, model, X_test)[,2]
                      
                      ## ---------- 在线 conformal p-value（供 LFS / SFS 使用） ----------
                      pval_online <- confomalPvalue_online(W_cal, W_test, Null_cal, Value, theta)
                      
                      ## ---------- (1) LOND + classic conformal p-values ----------
                      pval_classic <- conformalPvalue_classic(W_cal, W_test, Null_cal, Value)
                      rej_LOND_classic <- LOND(pval_classic, alpha)
                      res <- CiterionCompute(rej_LOND_classic$R, theta, "LOND")
                      res["prop"] <- pi_1
                      info <- rbind(info, res)
                      
                      ## ---------- (2) e-LOND + online conformal e-values ----------
                      eval_online <- conformalEvalue_online(W_cal, W_test, Null_cal, Value, theta)
                      rej_eLOND <- eLOND(eval_online, alpha)
                      res <- CiterionCompute(rej_eLOND$R, theta, "e-LOND")
                      res["prop"] <- pi_1
                      info <- rbind(info, res)
                      
                      
                      
                      ## ---------- (3) LFS ----------
                      rej_feedback_LORD_conse <- Lord_feedback_conservative(pval_online, alpha, theta, W0 = alpha / 2)
                      res <- CiterionCompute(rej_feedback_LORD_conse$R, theta, "LFS")
                      res["prop"] <- pi_1
                      info <- rbind(info, res)
                      
                      ## ---------- (4) SFS ----------
                      rej_feedback_SA_conse <- SAFFRON_feedback_conservative(pval_online, alpha, theta,
                                                                             w0 = alpha / 2, lambda = 0.5)
                      res <- CiterionCompute(rej_feedback_SA_conse$R, theta, "SFS")
                      res["prop"] <- pi_1
                      info <- rbind(info, res)
                    }
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)


################################################################
## summary+plotting
################################################################

pp <- result %>%
  group_by(Method, prop) %>%
  dplyr::summarize(FDR = mean(FDP), POWER = mean(Power),
                   FDP_se   = sd(FDP, na.rm = TRUE) / sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE) / sqrt(length(Power)))
print(pp,n=32)

pp$Method <- factor(pp$Method, levels = c("SFS", "LFS", "LOND", "e-LOND"))

P1 <- ggplot(data = pp, aes(x = prop, y = FDR, group = Method, color = Method,
                            shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  scale_shape_manual(values = c(21, 22, 23, 24)) +
  geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se),
              alpha = 0.1, linetype = 1, color = NA) +
  geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.2)) +
  theme_bw() +
  scale_color_nejm(palette = c("default"), alpha = 0.9) +
  scale_fill_manual(values = c("#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC")) +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major = element_line(colour = NA),
        panel.background = element_rect(fill = "transparent", colour = NA),
        plot.background = element_rect(fill = "transparent", colour = NA),
        panel.grid.minor = element_blank()) +
  theme(text = element_text(size = 16, family = "serif")) +
  theme(legend.position = "bottom")
P1 <- P1 + geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")
P1

P2 <- ggplot(data = pp, aes(x = prop, y = POWER, group = Method, color = Method,
                            shape = Method, fill = Method)) +
  geom_point(size = 2.5) +
  scale_shape_manual(values = c(21, 22, 23, 24)) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1, linetype = 1, color = NA) +
  geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.02)) +
  theme_bw() +
  scale_color_nejm(palette = c("default"), alpha = 0.9) +
  scale_fill_manual(values = c("#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC")) +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major = element_line(colour = NA),
        panel.background = element_rect(fill = "transparent", colour = NA),
        plot.background = element_rect(fill = "transparent", colour = NA),
        panel.grid.minor = element_blank(),
        text = element_text(size = 16, family = "serif"),
        legend.position = "bottom")
P2

cla_plots_vary_prop_safe_baselines <- ggarrange(P1, P2, ncol = 2, nrow = 1,
                                                common.legend = TRUE, legend = "bottom",
                                                font.label = list(size = 20, face = "bold"))

pdf(file = "cla_plots_vary_prop_safe_baselines.pdf", width = 10, height = 4)
cla_plots_vary_prop_safe_baselines
dev.off()
