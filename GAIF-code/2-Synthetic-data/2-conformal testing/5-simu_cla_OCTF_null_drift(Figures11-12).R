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
source("SAFFRON_feedback functions.R")
source("Model-sel-func.R")

conformalPvalue_online_sliding <- function(
    W_cal, W_test, Null_cal, Value, theta,
    window = c("growing", "sliding"), W = 200
) {
  window <- match.arg(window)
  
  # initial calibration score (null part)
  Phi_cal <- -ScoreCompute(W_cal, Value)
  Phi_Null <- Phi_cal[Null_cal]
  
  # calibration scores for test points (used for the window)
  Phi_test_all <- -ScoreCompute(W_test, Value)
  
  T_test <- length(Phi_test_all)
  pvalues <- numeric(T_test)
  
  for (i in seq_len(T_test)) {
    # --------- (1) build the current window's calibration null set ----------
    if (window == "growing") {
      # add all past theta = 0 test samples to calibration
      idx <- which(theta[seq_len(i - 1)] == 0)
      Phi_Null_i <- c(Phi_Null, Phi_test_all[idx])
    } else if (window == "sliding") {
      # sliding window: from max(1, i-W) to (i-1)
      start_i <- max(1, i - W)
      idx <- which(theta[start_i:(i - 1)] == 0)
      if (length(idx) > 0) {
        # idx are relative indices, convert to absolute test indices
        test_idx <- seq(start_i, i - 1)[idx]
        Phi_Null_i <- c(Phi_Null, Phi_test_all[test_idx])
      } else {
        Phi_Null_i <- Phi_Null  # no test-null in the current window
      }
    }
    
    n2 <- length(Phi_Null_i)
    
    # --------- (2) compute conformal p-value ----------
    t_val <- Phi_test_all[i]
    xi <- runif(1)
    
    pvalues[i] <- (sum(Phi_Null_i < t_val) + xi * sum(Phi_Null_i == t_val)) / (n2 + 1)
  }
  
  return(pvalues)
}



generate_data_nulldrift <- function(T = 1000, pi1 = 0.2, Delta = 0) {
  # T: total number of observations
  # pi1: proportion of non-null (Y=1)
  # Delta: drift severity parameter
  
  p <- 4  # dimension
  
  # Generate labels
  Y <- rbinom(T, size = 1, prob = pi1)
  
  # Initialize matrix for X
  X <- matrix(0, nrow = T, ncol = p)
  
  for (t in 1:T) {
    if (Y[t] == 0) {
      # Null: drifting mean
      mu_t <- c(2 - Delta * (t / T), 0, 0, 0)
      X[t, ] <- MASS::mvrnorm(1, mu = mu_t, Sigma = diag(p))
    } else {
      # Alternative: fixed mean
      mu_1 <- rep(0, p)
      X[t, ] <- MASS::mvrnorm(1, mu = mu_1, Sigma = diag(p))
    }
  }
  
  # Construct data frame with only X and Y
  data <- data.frame(
    X1 = X[, 1],
    X2 = X[, 2],
    X3 = X[, 3],
    X4 = X[, 4],
    y = Y
  )
  
  return(data)
}

set.seed(123)
data <- generate_data_nulldrift(T = 1000, pi1 = 0.2, Delta = 3)
head(data)



T <- 1000 # number of total time points

# simulation setting-----
alpha <- 0.2 # significance level
pi <- 0.5 # Bernoulli(pi)
n <- 500 # number of historical data
n_train<- round(n/2) # number of data used for training model
n_cal <- round(n/2)

algo<- new("NN") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm
Value=list(type="==A,R",v=0)


##################  ------------
alpha <- 0.2
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)
pi1 <- 0.5


nr <- 500  # number of repeats
cl <- makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                "tidyverse","lsa","magrittr","doSNOW","pbmcapply"),
                  .errorhandling = "remove", .options.snow = opts) %dopar% {
                    # --- local environment for this repeat ---
                    set.seed(1000 + iter) # optional: for reproducibility
                    
                    # generate test data with null drift
                    T <- 1000
                    data <- generate_data_nulldrift(T = T, pi1 = 0.3, Delta = -2)
                    
                    # generate historical data (no drift) for model training/calibration
                    n <- 1000
                    n_cal <- 200
                    his_data <- generate_data_nulldrift(T = n, pi1 = 0.3, Delta = 0)
                    
                    p <- ncol(his_data) - 1
                    
                    # split historical data (DataSplit is assumed to return list(data_train, data_cal))
                    datawork <- DataSplit(his_data, n, 0, n_cal)
                    data_train <- datawork$data_train
                    data_cal <- datawork$data_cal
                    
                    # calibration null indices (NullIndex is assumed to return logical or integer indices)
                    Null_cal <- NullIndex(data_cal$y, Value)
                    
                    X_train <- as.matrix(data_train[colnames(data_train)[- (p + 1)]])
                    Y_train <- as.matrix(data_train$y)
                    X_cal   <- as.matrix(data_cal[colnames(data_cal)[- (p + 1)]])
                    Y_cal   <- as.matrix(data_cal$y)
                    
                    data_test <- data
                    Null_test <- NullIndex(data_test$y, Value)
                    Alter_test <- setdiff(seq_len(nrow(data_test)), Null_test)
                    X_test <- as.matrix(data_test[colnames(data_test)[- (p + 1)]])
                    Y_test <- as.numeric(data_test$y)
                    theta <- Y_test
                    
                    # fit model (fitting, Pred etc. are defined elsewhere)
                    model <- fitting(algo, X_train, Y_train, lambda)
                    W_cal  <- Pred(algo, model, X_cal)
                    W_test <- Pred(algo, model, X_test)
                    
                    # compute two types of p-values: growing (online) vs sliding window (W = 200)
                    pvals_online <- conformalPvalue_online(W_cal, W_test, Null_cal, Value, theta) # growing
                    pvals_window <- conformalPvalue_online_sliding(W_cal, W_test, Null_cal, Value, theta,
                                                                   window = "sliding", W = 100)    # sliding
                    
                    # ---- Apply multiple testing procedures ----
                    # NOTE: each procedure is applied to both types of p-values, named *_grow / *_slide
                    methods_results <- list()
                    
                    # helper to compute metrics for a rejection vector R (length T)
                    compute_metrics <- function(R) {
                      res <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions = R, .x)) %>% unlist() %>% split(., names(.))
                      list(FDP = res$FDP, Power = res$Power, time = res$time)
                    }
                    
                    # LF (Lord_feedback) — growing
                    rej_LF_grow <- Lord_feedback(pvals_online, alpha, theta, W0 = alpha/2)
                    metrics <- compute_metrics(rej_LF_grow$R)
                    methods_results[["LF_grow"]] <- c(metrics, method = "LF_grow")
                    
                    # LF (sliding)
                    rej_LF_slide <- Lord_feedback(pvals_window, alpha, theta, W0 = alpha/2)
                    metrics <- compute_metrics(rej_LF_slide$R)
                    methods_results[["LF_slide"]] <- c(metrics, method = "LF_slide")
                    
                    # LF conservative
                    rej_LFS_grow <- Lord_feedback_conservative(pvals_online, alpha, theta, W0 = alpha/2)
                    metrics <- compute_metrics(rej_LFS_grow$R)
                    methods_results[["LFS_grow"]] <- c(metrics, method = "LFS_grow")
                    
                    rej_LFS_slide <- Lord_feedback_conservative(pvals_window, alpha, theta, W0 = alpha/2)
                    metrics <- compute_metrics(rej_LFS_slide$R)
                    methods_results[["LFS_slide"]] <- c(metrics, method = "LFS_slide")
                    
                    # SF (SAFFRON_feedback)
                    rej_SF_grow <- SAFFRON_feedback(pvals_online, alpha = alpha, theta = theta, w0 = alpha/2)
                    metrics <- compute_metrics(rej_SF_grow$R)
                    methods_results[["SF_grow"]] <- c(metrics, method = "SF_grow")
                    
                    rej_SF_slide <- SAFFRON_feedback(pvals_window, alpha = alpha, theta = theta, w0 = alpha/2)
                    metrics <- compute_metrics(rej_SF_slide$R)
                    methods_results[["SF_slide"]] <- c(metrics, method = "SF_slide")
                    
                    # SF conservative
                    rej_SFS_grow <- SAFFRON_feedback_conservative(pvals_online, alpha = alpha, theta = theta, w0 = alpha/2)
                    metrics <- compute_metrics(rej_SFS_grow$R)
                    methods_results[["SFS_grow"]] <- c(metrics, method = "SFS_grow")
                    
                    rej_SFS_slide <- SAFFRON_feedback_conservative(pvals_window, alpha = alpha, theta = theta, w0 = alpha/2)
                    metrics <- compute_metrics(rej_SFS_slide$R)
                    methods_results[["SFS_slide"]] <- c(metrics, method = "SFS_slide")
                    
                    # LORD++ and SAFFRON (batch online baselines) — applied to both p-value types
                    rej_LORD_grow <- LORD(pvals_online, alpha)
                    metrics <- compute_metrics(rej_LORD_grow$R)
                    methods_results[["LORD_grow"]] <- c(metrics, method = "LORD_grow")
                    
                    rej_LORD_slide <- LORD(pvals_window, alpha)
                    metrics <- compute_metrics(rej_LORD_slide$R)
                    methods_results[["LORD_slide"]] <- c(metrics, method = "LORD_slide")
                    
                    rej_SAFFRON_grow <- SAFFRON(pvals_online, alpha)
                    metrics <- compute_metrics(rej_SAFFRON_grow$R)
                    methods_results[["SAFFRON_grow"]] <- c(metrics, method = "SAFFRON_grow")
                    
                    rej_SAFFRON_slide <- SAFFRON(pvals_window, alpha)
                    metrics <- compute_metrics(rej_SAFFRON_slide$R)
                    methods_results[["SAFFRON_slide"]] <- c(metrics, method = "SAFFRON_slide")
                    
                    # --- pack info for this iteration ---
                    # convert to list-of-lists, each element has named vectors for FDP/Power/time and method tag
                    info <- list()
                    for (name in names(methods_results)) {
                      m <- methods_results[[name]]
                      info[[length(info) + 1]] <- list(FDP = m$FDP, Power = m$Power, time = m$time, method = m$method)
                    }
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)

# result is an nr x 1 list, each element is a list (multi-method results for one iteration)
# standardize result into a data.frame/tibble for downstream summarizing/plotting

library(purrr)
library(tibble)

all_runs <- map_dfr(result, function(m) {
  Tlen <- length(m$FDP)
  tibble(
    Time = seq_len(Tlen),
    FDP = m$FDP,
    Power = m$Power,
    TimeIndex = m$time,
    Method = m$method
  )
}, .id = "iter")


# Note: all_runs above is a very long table (nr * #methods * T rows);
# it can be summarized by TimeIndex (or Time) and Method to get means and SEs for plotting.

pp <- all_runs %>%
  group_by(Method, TimeIndex) %>%
  summarize(
    FDP = mean(FDP, na.rm = TRUE),
    Power = mean(Power, na.rm = TRUE),
    FDP_se = sd(FDP, na.rm = TRUE)/sqrt(n()),
    Power_se = sd(Power, na.rm = TRUE)/sqrt(n()),
    .groups = "drop"
  )

# reset Method factor order (adjust as needed)
pp$Method <- factor(pp$Method, levels = c("SF_grow","SF_slide","SFS_grow","SFS_slide",
                                          "LF_grow","LF_slide","LFS_grow","LFS_slide",
                                          "SAFFRON_grow","SAFFRON_slide","LORD_grow","LORD_slide"))

# save the result object for later inspection
save(result, file = "conformal_compare_result.RData")


########### plots ------------

library(dplyr)
library(purrr)
library(ggplot2)
library(ggpubr)

# method names (including grow/slide)
method_levels <- c(
  "SF_grow","SF_slide",
  "SFS_grow","SFS_slide",
  "LF_grow","LF_slide",
  "LFS_grow","LFS_slide",
  "SAFFRON_grow","SAFFRON_slide",
  "LORD_grow","LORD_slide"
)

# corresponding colors (adjust as needed)
method_colors <- c(
  "#BC3C29", "#F4A582",   # SF
  "#0072B5", "#A6CEE3",   # SFS
  "#E18727", "#FDBF6F",   # LF
  "#20854E", "#B2DF8A",   # LFS
  "#984EA3", "#CAB2D6",   # SA
  "#6A3D9A", "#CAB2D6"    # LORD (two purple shades)
)
names(method_colors) <- method_levels

# shapes
shape_values <- c(21, 22, 23, 24, 25, 21, 22, 23, 24, 25, 21, 22)
names(shape_values) <- method_levels

# linetypes: solid for grow, dashed for slide
linetype_values <- rep(c("solid", "dashed"), length(method_levels)/2)
names(linetype_values) <- method_levels

# convert result into long-format data frame
all_runs <- map_dfr(result, function(m) {
  tibble(
    Time = seq_along(m$FDP),
    FDP = m$FDP,
    Power = m$Power,
    TimeIndex = m$time,
    Method = m$method
  )
}, .id = "iter") %>%
  mutate(Method = factor(Method, levels = method_levels)) %>%
  filter(!is.na(Method))

# subsample time points to avoid overly dense points
t_points <- c(seq(10, 1000, 100), 1000)

# summary function
summary_by_group <- function(df, t_pts) {
  df %>%
    filter(Time %in% t_pts) %>%
    group_by(Time, Method) %>%
    summarize(
      FDP_avg = mean(FDP, na.rm = TRUE),
      Power_avg = mean(Power, na.rm = TRUE),
      FDP_se = sd(FDP, na.rm = TRUE)/sqrt(n()),
      Power_se = sd(Power, na.rm = TRUE)/sqrt(n()),
      .groups = "drop"
    )
}

# SF/SA group (SF, SFS, SA)
pp_saffron <- all_runs %>%
  filter(Method %in% c("SF_grow","SF_slide","SFS_grow","SFS_slide","SAFFRON_grow","SAFFRON_slide"))
pp_saffron_summary <- summary_by_group(pp_saffron, t_points)

# LF/LFS/LORD group
pp_lord <- all_runs %>%
  filter(Method %in% c("LF_grow","LF_slide","LFS_grow","LFS_slide","LORD_grow","LORD_slide"))
pp_lord_summary <- summary_by_group(pp_lord, t_points)

# plotting function with confidence band and custom colors/shapes/linetypes
plot_metric <- function(df, y_avg, y_se, y_label, y_lim = c(0,1), add_alpha_line = FALSE, alpha = 0.05) {
  p <- ggplot(df, aes(x = Time, y = .data[[y_avg]], group = Method)) +
    geom_point(aes(color = Method, fill = Method, shape = Method), size = 2.5) +
    geom_ribbon(aes(ymin = .data[[y_avg]] - .data[[y_se]], ymax = .data[[y_avg]] + .data[[y_se]],
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
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      text = element_text(size = 16, family = "serif")
    )
  
  if (add_alpha_line) {
    p <- p + geom_hline(yintercept = 0.2, colour = "black", linetype = "dashed")
  }
  
  return(p)
}
alpha <- 0.2
# plot FDP and Power for the SF/SA group
p_saf_fdp <- plot_metric(pp_saffron_summary, "FDP_avg", "FDP_se", "FDR", c(0,1), add_alpha_line = TRUE)
p_saf_power <- plot_metric(pp_saffron_summary, "Power_avg", "Power_se", "Power", c(0,1), add_alpha_line = FALSE)

# plot FDP and Power for the LF/LFS/LORD group
p_lord_fdp <- plot_metric(pp_lord_summary, "FDP_avg", "FDP_se", "FDR", c(0,1), add_alpha_line = TRUE)
p_lord_power <- plot_metric(pp_lord_summary, "Power_avg", "Power_se", "Power", c(0,1), add_alpha_line = FALSE)

# combine plots
plot_cla_Opt_SAFFRON <- ggarrange(p_saf_fdp, p_saf_power, ncol=2, nrow=1,
                                  common.legend = TRUE, legend="bottom",
                                  font.label = list(size = 16, face = "bold"))

plot_cla_Opt_LORD <- ggarrange(p_lord_fdp, p_lord_power, ncol=2, nrow=1,
                               common.legend = TRUE, legend="bottom",
                               font.label = list(size = 16, face = "bold"))

plot_cla_Opt <- ggarrange(plot_cla_Opt_LORD, plot_cla_Opt_SAFFRON, ncol=1, nrow=2,
                          common.legend = TRUE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))

plot_cla_Opt

# save pdf (adjust path as needed)
pdf("plot_cla_Opt_SAFFRON.pdf", width = 10, height = 4)
print(plot_cla_Opt_SAFFRON)
dev.off()

pdf("plot_cla_Opt_LORD.pdf", width = 10, height = 4)
print(plot_cla_Opt_LORD)
dev.off()

pdf("plot_cla_Opt_vary_null_shift.pdf", width = 10, height = 8)
print(plot_cla_Opt)
dev.off()