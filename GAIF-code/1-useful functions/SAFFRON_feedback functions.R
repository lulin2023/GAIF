

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
  if (alpha <= 0 || alpha > 1) stop("alpha 必须在 (0,1] 内。")
  if (lambda <= 0 || lambda > 1) stop("lambda 必须在 (0,1] 内。")
  
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

theta_wealth_SF_delayed <- function(theta, alphai, gammai, cand, i, d) {
 
  upper <- i - d
  if (upper < 1) return(0)

  upper <- min(upper, length(theta), length(alphai), length(cand))
  idx <- which(theta[1:upper] == 1 & cand[1:upper] == 0)
  
  if (length(idx) == 0) return(0)
  
  cand_cumsum <- cumsum(cand[1:i])
  lag <- (i + 1 - idx) - (cand_cumsum[i] - cand_cumsum[idx])
  
  valid <- lag > 0 & lag <= length(gammai)
  
  if (sum(valid) == 0) return(0)
  wealth <- sum(alphai[idx[valid]] * gammai[lag[valid]])
  return(wealth)
}

SAFFRON_feedback_bandit <- function(pval, alpha = 0.05, theta, gammai, w0, lambda = 0.5, 
                             random = TRUE, display_progress = FALSE) {
  
  N <- length(pval)
  
  if (missing(gammai)) {
    gammai <- 0.4374901658 / (seq_len(N)^(1.6))
  } 
  
  if (missing(w0)) {
    w0 <- alpha / 2
  } 
  
 
  saffron_faster_feedback_R_bandit <- function(pval, gammai, lambda, alpha, w0, theta) {
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
      
    
      feedback_bonus <- theta_wealth_SF_bandit(theta, decision=Rvec,alphai, gammai, cand, i - 1)
      
      alphaitilde <- alphaitilde + feedback_bonus
      alphai[i] <- min(lambda, alphaitilde)
      
      if (pval[i] <= alphai[i]) {
        Rvec[i] <- TRUE
        K <- K + 1
      }
    }
    
    data.frame(pval = pval, alphai = alphai, R = as.numeric(Rvec))
  }
  
  out <- saffron_faster_feedback_R_bandit(pval,gammai, lambda, alpha, w0, theta)
  out
}


SAFFRON_feedback_delayed <- function(pval, alpha = 0.05, theta, gammai, w0, lambda = 0.5, 
                             random = TRUE, display_progress = FALSE,d) {
  
  N <- length(pval)
  
  
  if (missing(gammai)) {
    gammai <- 0.4374901658 / (seq_len(N)^(1.6))
  }
  
  if (missing(w0)) {
    w0 <- alpha / 2
  } 
  
  
  saffron_faster_feedback_R_delayed <- function(pval, gammai, lambda, alpha, w0, theta,d) {
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
      
      
      feedback_bonus <- theta_wealth_SF_delayed(theta, alphai, gammai, cand, i - 1,d)
      
      alphaitilde <- alphaitilde + feedback_bonus
      alphai[i] <- min(lambda, alphaitilde)
      
      if (pval[i] <= alphai[i]) {
        Rvec[i] <- TRUE
        K <- K + 1
      }
    }
    
    data.frame(pval = pval, alphai = alphai, R = as.numeric(Rvec))
  }
  
  out <- saffron_faster_feedback_R_delayed(pval, gammai, lambda, alpha, w0, theta,d)
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
      
      ## false discoveries only
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

