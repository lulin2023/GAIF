


confomalPvalue_online <- function(W_cal, W_test, Null_cal, Value, theta) {
  Phi_cal <- -ScoreCompute(W_cal, Value)
  Phi_test <- -ScoreCompute(W_test, Value)
  
  
  Phi_Null <- Phi_cal[Null_cal]
  n2 <- length(Phi_Null)
  
  pvalues <- numeric(length(Phi_test))
  
  for (i in seq_along(Phi_test)) {
    t <- Phi_test[i]
    xi <- runif(1)
    
    pvalues[i] <- (sum(Phi_Null < t) + xi * (1+sum(Phi_Null == t))) / (n2 + 1)
    
    
    if (theta[i] == 0) {
      new_score <- -ScoreCompute(W_test[i], Value)
      Phi_Null <- c(Phi_Null, new_score)
      n2 <- length(Phi_Null)
    }
  }
  
  return(pvalues)
}



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
    
    xi <- runif(1)
    
    pvalues[t] <- (sum(Phi_Null_all[, k_opt] < Phi_test_all[t, k_opt]) + xi*(1+sum(Phi_Null_all[, k_opt] == Phi_test_all[t, k_opt]))) / (n2 + 1)
    
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
    xi = runif(1)
    pvalues[t] <- (sum(Phi_Null_all[, k_opt] < Phi_test_all[t, k_opt]) + xi*(1+sum(Phi_Null_all[, k_opt] == Phi_test_all[t, k_opt]))) / (n2 + 1)
    
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




