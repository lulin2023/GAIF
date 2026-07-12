
############ 

theta_wealth <- function(theta, alpha_t, gamma_t, i){

  idx <- which(theta[1:i] == 1)

  if(length(idx) == 0) return(0)

  lag <- i - idx

  valid <- lag > 0

  sum(alpha_t[idx[valid]] * gamma_t[lag[valid]])
}



Lord_feedback_dep <- function(pval, L, theta,
                              gammai = sapply(seq_along(pval), function(j) 1/j^(1.6)) * 0.4374901658,
                              W0 = 0.025, alpha = 0.05) {
  N <- length(pval)
  stopifnot(length(L) == N, length(theta) == N, length(gammai) == N)
  
  alpha_t <- numeric(N)
  decision <- integer(N)
  
  alpha_t[1] <- min(alpha, W0 * gammai[1])
  decision[1] <- as.integer(pval[1] <= alpha_t[1])
  
  for (i in 2:N) {
    cutoff <- i - L[i]
    
    taus <- if (cutoff >= 1) which(decision[1:cutoff] == 1) else integer(0)
    
    if (length(taus) == 0) {
      alpha_base <- W0 * gammai[i]
    } else {
      first_term <- (alpha - W0) * gammai[i - taus[1]]
      other_term <- if (length(taus) > 1) alpha * sum(gammai[i - taus[-1]], na.rm = TRUE) else 0
      alpha_base <- W0 * gammai[i] + first_term + other_term
    }
    
    wealth <- theta_wealth_LORD_dep(theta, alpha_t, gammai, i, cutoff)
    
    alpha_t[i] <- alpha_base + wealth
    decision[i] <- as.integer(pval[i] <= alpha_t[i])
  }
  
  data.frame(pval = pval, lag = L, alphai = alpha_t, R = decision)
}


theta_wealth_SF_dep <- function(theta, alphai, gammai, cand, i, bound) {
  if (bound <= 0) return(0)
  
  
  idx <- which(theta[1:bound] == 1 & cand[1:bound] == 0)
  if (length(idx) == 0) return(0)
  
  # C_{j+}^* = sum_{i=j+1}^{t-L_t} C_i
  cand_cumsum <- cumsum(cand[1:bound])
  
  # lag = (t - j) - C_{j+}^*
  # C_{j+}^*(i) = cand_cumsum[bound] - cand_cumsum[idx]
  Cjplus_star <- cand_cumsum[bound] - cand_cumsum[idx]
  lag <- (i - idx) - Cjplus_star
  
  valid <- lag > 0
  sum(alphai[idx[valid]] * gammai[lag[valid]])
}

SAFFRON_feedback_dep <- function(pval, L, 
                                 gammai = sapply(seq_along(pval), function(j) 1/j^(1.6)) * 0.4374901658,
                                 w0 = 0.025, lambda = 0.5, alpha = 0.05, theta, 
                                 display_progress = TRUE) {
  N <- length(pval)
  alphai <- numeric(N)
  Rvec <- logical(N)
  cand <- integer(N)
  
  alphai[1] <- min((1 - lambda) * gammai[1] * w0, lambda)
  Rvec[1] <- (pval[1] <= alphai[1])
  cand[1] <- as.integer(pval[1] <= lambda)
  
  if (display_progress) pb <- txtProgressBar(min = 0, max = N, style = 3)
  
  for (i in 2:N) {
    cand[i - 1] <- as.integer(pval[i - 1] <= lambda)
    
    # bound = t - L_t，对应公式中求和上限
    bound <- i - L[i]
    
    # r_k：在 bound 内的 rejection 时间点
    taus <- if (bound >= 1) which(Rvec[1:bound]) else integer(0)
    K <- length(taus)
    
    # C_{0+} = sum_{i=1}^{t-L_t} C_i
    candsum <- if (bound > 0) sum(cand[1:bound]) else 0
    
    # --- 1. 基底 SAFFRON 部分 (乘 1-lambda，对应公式前三项) ---
    if (K == 0) {
      alphaitilde_base <- (1 - lambda) * w0 * gammai[max(i - candsum, 1)]
      
    } else {
      # C_{j+} = sum_{i=r_j+1}^{t-L_t} C_i，上限为 bound
      cand_cumsum_bound <- if (bound > 0) cumsum(cand[1:bound]) else integer(0)
      
      Cjplus <- integer(K)
      for (j in seq_len(K)) {
        rj <- taus[j]
        if (rj < bound) {
          Cjplus[j] <- cand_cumsum_bound[bound] - cand_cumsum_bound[rj]
        } else {
          Cjplus[j] <- 0
        }
      }
      
      idx_gamma_first <- max(i - taus[1] - Cjplus[1], 1)
      
      if (K > 1) {
        other_sum <- sum(sapply(2:K, function(j) gammai[max(i - taus[j] - Cjplus[j], 1)]))
        alphaitilde_base <- (1 - lambda) * (
          w0 * gammai[max(i - candsum, 1)] +
            (alpha - w0) * gammai[idx_gamma_first] +
            alpha * other_sum
        )
      } else {
        alphaitilde_base <- (1 - lambda) * (
          w0 * gammai[max(i - candsum, 1)] +
            (alpha - w0) * gammai[idx_gamma_first]
        )
      }
    }
    
    # --- 2. Feedback bonus (对应公式红色项，上限为 bound) ---
    feedback_bonus <- theta_wealth_SF_dep(theta, alphai, gammai, cand, i, bound)
    
    # --- 3. 合并并截断 ---
    alphai[i] <- min(lambda, alphaitilde_base + feedback_bonus)
    Rvec[i] <- (pval[i] <= alphai[i])
    
    if (display_progress) setTxtProgressBar(pb, i)
  }
  
  if (display_progress) close(pb)
  
  data.frame(pval = pval, lag = L, alphai = alphai, R = Rvec)
}
