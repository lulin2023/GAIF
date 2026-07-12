


data_generation_classication1 <- function(N=5000,mu1= c(2,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=0.2,pi=0.2){
  Y <- rbinom(n=2*N, size=1, prob=pi)
  ident <- diag(p)
  X0 <- mvrnorm(n=2*N,mu1,Sigma=ident)
  X1 <- mvrnorm(n=2*N,mu2,Sigma=ident)
  id <- c(1:(2*N))
  id_0 <- id[which(Y==0)]
  id_1 <- id[which(Y==1)]
  train_0=sample(id_0,N*(1-propotion))
  train_1=sample(id_1,N*propotion)
  data <- matrix(NA, nrow = 2*N, ncol = p+1)
  data[,p+1] <- Y
  data[train_0,-(p+1)] <- X0[train_0,] 
  data[train_1,-(p+1)] <- X1[train_1,] 
  data <- as.data.frame(data)
  names(data)[5] <- "y"
  data1 <- data[complete.cases(data),]
  return(data=data1)
}



data_generation_regression<- function(N=5000){
  X1 <- rnorm(n=N,0,1)
  X2 <- rnorm(n=N,0,1)
  X3 <- rnorm(n=N,0,1)
  X4 <- rnorm(n=N,0,1)
  epsilon <- rnorm(n=N,0,1)
  Y <- -2*X1^2+3*exp(X2)+5*(X3+X4)^2+epsilon
  data <- cbind(X1,X2,X3,X4,Y)
  data <- as.data.frame(data)
  names(data)[5] <- "y"
  return(data=data)
}


data_generation_regression2 <- function(N = 5000) {
  X1 <- rnorm(n = N, 0, 1)
  X2 <- rnorm(n = N, 0, 1)
  X3 <- rnorm(n = N, 0, 1)
  X4 <- rnorm(n = N, 0, 1)
  epsilon <- rnorm(n = N, 0, 2)
  
  # 降低信号强度
  Y <- -0.5 * X1^2 + 1 * exp(X2) + (X3 + X4)^2 + epsilon
  
  data <- data.frame(X1, X2, X3, X4, y = Y)
  return(data)
}



block_cov.data <- function(t_max=500,mean_value=4,proportion=0.2,rho=0.5){
  mu <- rep(0, t_max)  # For simplicity, assuming a mean of 0 for all variables
  
  subset_h1 <- sample(c(1:t_max),size=t_max*proportion,replace = FALSE)
  
  mu[subset_h1] <- mean_value 
  
  theta <- rep(0, t_max)
  theta[subset_h1] <- 1 
  # Covariance matrix
  Sigma <- matrix(c(rep(c(1,rep(rho,t_max)),t_max-1),1), nrow = t_max, ncol = t_max)  # Example: a diagonal matrix with 0.5 on the diagonal
  
  # Simulate multivariate normally distributed variables
  data <- mvrnorm(n = 1, mu = mu, Sigma = Sigma)
  
  pvalue <- 1-pnorm(data)
  return(list(data=data,pvalue=pvalue,theta=theta))
  
}


generate_block_covariance <- function(block_sizes, block_correlation) {
  
  num_variables <- sum(block_sizes)
  covariance_matrix <- matrix(0, nrow = num_variables, ncol = num_variables)
  
  # Generate block-structured covariance matrix
  start_idx <- 1
  for (block_size in block_sizes) {
    end_idx <- start_idx + block_size - 1
    covariance_matrix[start_idx:end_idx, start_idx:end_idx] <- block_correlation
    start_idx <- end_idx + 1
  }
  
  diag(covariance_matrix) <- 1
  return(covariance_matrix)
}



block_cov.data_new <- function(t_max=500,mean_value=4,proportion=0.2,rho=0.5,nbatch=5){
  mu <- rep(0, t_max)  # For simplicity, assuming a mean of 0 for all variables
  
  subset_h1 <- sample(c(1:t_max),size=t_max*proportion,replace = FALSE)
  
  mu[subset_h1] <- mean_value 
  
  theta <- rep(0, t_max)
  theta[subset_h1] <- 1 
  # Covariance matrix
  block_sizes <- rep(t_max/nbatch,nbatch)
  
  Sigma <- generate_block_covariance(block_sizes, block_correlation=rho)
  
  # Simulate multivariate normally distributed variables
  Z <- mvrnorm(n = 1, mu = mu, Sigma = Sigma)
  
  pvalue <- 1-pnorm(Z)
  return(list(Z=Z,pvalue=pvalue,theta=theta))
}



# data <- data_generation_regression()
# hist(data$y)
# sum(data$y>quantile(data$y,0.8))
     

DataSplit<-function(data,n,n_test,n_cal,n_rest)
{
  if(n_test>0)
  {  index_test=sample(1:n,n_test,replace=FALSE)
  data_test=data[index_test,]
  data_rest2=data[-index_test,]
  data_rest=data_rest2[sample(1:dim(data_rest2)[1],n_rest),]
  index_cal=sample(1:dim(data_rest)[1],n_cal)
  data_train=data_rest[-index_cal,]
  data_cal=data_rest[index_cal,]
  return(list(data_train=data_train,data_cal=data_cal,data_test=data_test,data_rest=data_rest))}else
  {
    data_rest=data[sample(1:dim(data)[1],n_rest,replace=FALSE),]
    index_cal=sample(1:dim(data_rest)[1],n_cal)
    data_train=data_rest[-index_cal,]
    data_cal=data_rest[index_cal,]
    return(list(data_train=data_train,data_cal=data_cal,data_test=0,data_rest=data_rest))
  }
  
}

confomalPvalue<-function(W_cal,W_test,Null_cal,Value)
{
  Phi_cal=-ScoreCompute(W_cal,Value)
  Phi_test=-ScoreCompute(W_test,Value)
  Phi_Null=Phi_cal[Null_cal]
  n2=length(Phi_Null)
  pvalue=sapply(Phi_test,function(t){
    xi=runif(1)
    (sum(Phi_Null<t)+xi*sum(Phi_Null==t))/(n2+1)
  })
  return(pvalue)
}

ScoreCompute<-function(pred,Value)
{
  if(Value$type=="==A,S")
  {Phi=-pred}else if(Value$type=="==A,R")
  {Phi=pred}else if(Value$type=="<=A")
  {Phi=pred}else if(Value$type==">=B")
  {Phi=-pred}else if(Value$type=="<=A|>=B")
  {Phi=pmin(pred-Value$v[1],Value$v[2]-pred)}else if(Value$type==">=A&<=B")
  {Phi=pmax(Value$v[1]-pred,pred-Value$v[2])}
  return(Phi)
}



NullIndex<-function(y,Value)
{
  if(Value$type=="==A,S"|Value$type=="==A,R")
  {index=which(y==Value$v)}else if(Value$type=="<=A")
  {index=which(y<=Value$v)}else if(Value$type==">=B")
  {index=which(y>=Value$v)}else if(Value$type=="<=A|>=B")
  {index=which(y<=Value$v[1]|y>=Value$v[2])}else if(Value$type==">=A&<=B")
  {index=which(y>=Value$v[1]&y<=Value$v[2])}
  return(index)
}






CiterionCompute_each<-function(Alter_test,decisions,t)
{TrueSignal=intersect(which(decisions[1:t]==1),Alter_test)
  if (sum(decisions[1:t])!=0){
    FDP=1-length(TrueSignal)/sum(decisions[1:t])
  } else {
    FDP <- 0
  }

if (sum(decisions[1:t])!=0){
  Power=length(TrueSignal)/length(intersect(1:t,Alter_test))
} else {
  Power=0
}
select.num=sum(decisions[1:t])
select.num.true=length(TrueSignal)
  return(data.frame(FDP=FDP,Power=Power,time=t,select.num=select.num,
                    select.num.true=select.num.true))
}



TimeCovert<-function(FDP,decisions)
{
  FDP_time=rep(0,length(decisions))
  current_FDP=0
  select.num=0
  for (i in 1:length(decisions)) {
    if(decisions[i]==1){
      select.num=select.num+1
      current_FDP=FDP[select.num]
    }
    FDP_time[i]=current_FDP
  }
  return(FDP_time)
}


removeColsAllNa  <- function(x){x[, apply(x, 2, function(y) any(!is.na(y)))]}







CiterionCompute<-function(rejection,theta,method_name=" ")
{  FDP=(sum((1-theta)*rejection))/max(sum(rejection),1)
Power=sum(theta*rejection)/sum(theta)
return(data.frame(FDP=FDP,Power=Power,Method=method_name))
}




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


theta_wealth_bandit <- function(theta, decision,alpha_t, gamma_t, i){
  
  idx <- which(theta[1:i]*decision[1:i] == 1)
  
  if(length(idx) == 0) return(0)
  
  lag <- i - idx
  
  valid <- lag > 0
  
  sum(alpha_t[idx[valid]] * gamma_t[lag[valid]])
}



theta_wealth_delayed <- function(theta, alpha_t, gamma_t, i, d=10) {

  upper <- i - d - 1
  
  if (upper < 1) {
    return(0)
  }
  
  idx <- which(theta[1:upper] == 1)
  
  if (length(idx) == 0) {
    return(0)
  }
  
  lag <- i - idx
  
  valid <- lag > 0 & lag <= length(gamma_t)
  
  if (sum(valid) == 0) {
    return(0)
  }
  
  wealth <- sum(alpha_t[idx[valid]] * gamma_t[lag[valid]])
  
  return(wealth)
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


Lord_feedback_delayed<-function(pval,alpha,theta,W0,d){
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
      alpha_t[i+1]=W0*gamma_t[i+1]+theta_wealth_delayed(theta,alpha_t,gamma_t,i,d)
    }  
    if(sum(decision)==1){
      tau1=which(decision==1)
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0)*gamma_t[i+1-tau1]+theta_wealth_delayed(theta,alpha_t,gamma_t,i,d)
    }
    if(sum(decision)>1){
      tau_seq=which(decision==1)
      tau_seq=tau_seq[2:(length(tau_seq))]
      alpha_t[i+1]=W0*gamma_t[i+1]+(alpha-W0+alpha_t[tau1]*theta[tau1])*gamma_t[i+1-tau1]+
        sum(sapply(tau_seq, function(tau_j){
          (alpha)*gamma_t[i+1-tau_j]
        }))+theta_wealth_delayed(theta,alpha_t,gamma_t,i,d)
    }
  }
  
  
  return(data.frame(pval=pval,alphai=alpha_t[-(N+1)],R=decision))
}


Lord_feedback_conservative_delayed <- function(pval, alpha, theta, W0,d) {
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
    term4 <- theta_wealth_delayed(theta, alpha_t, gamma_t, i,d)
    alpha_t[i + 1] <- term1 + term2 + term3 + term4
  }
  
  data.frame(pval = pval, alphai = alpha_t[-(N + 1)], R = decision)
}






