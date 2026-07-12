


####### Simulation for GAIF-----------

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

#base <- "/Users/GAIF code"

source(file.path(base, "useful functions", "functions_OnSel.R"))
source(file.path(base, "useful functions", "SAFFRON_feedback functions.R"))
source(file.path(base, "useful functions", "local dependence func.R"))


# Load necessary libraries
library(MASS)  # for mvrnorm function

# Function to generate the data
generate_data_Gaussian <- function(T, pi_1, mu_c_values) {
  # Initialize vectors to store mu_t and p_t
  mu_t <- numeric(T)
  p_t <- numeric(T)
  Z_t <- numeric(T)
  theta <- numeric(T)
  
  # Loop over each time point t
  for (t in 1:T) {
    # Generate mu_t from the mixture model
    if (runif(1) <= pi_1) {
      # With probability pi_1, mu_t = F1 ~ N(mu_c, 1)
      mu_t[t] <- sample(mu_c_values, 1)
      theta[t] <- 1
    } else {
      # With probability 1 - pi_1, mu_t = 0
      mu_t[t] <- 0
      theta[t] <- 0
    }
    
    
    # Generate the observation Z_t ~ N(mu_t, 1)
    Z_t[t] <- rnorm(1, mu_t[t], 1)
    
    # Calculate the one-sided p-value p_t = Phi(-Z_t)
    p_t[t] <- pnorm(-Z_t[t])
  }
  
  # Return the generated mu_t and p_t values
  return(data.frame(Z = Z_t, p = p_t,theta=theta))
}



mu_c_values <- 2.5

N=1000
#prop=0.1
alpha=0.1
lambda=0.5
pi1_seq=seq(0.1,0.8,0.1)

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  info<-data.frame()
  
  for (pi_1 in pi1_seq) {
    
    data <- generate_data_Gaussian(T=N, pi_1, mu_c_values)
    head(data)
    data=data[sample(nrow(data)), ]
    
    
    rej_SAFFRON=SAFFRON(data$p,alpha,w0=alpha/2)
    res=CiterionCompute(rej_SAFFRON$R,data$theta,"SAFFRON")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_LORD=LORD(data$p,alpha,w0=alpha/2)
    res=CiterionCompute(rej_LORD$R,data$theta,"LORD++")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_LOND=LOND(data$p,alpha)
    res=CiterionCompute(rej_LOND$R,data$theta,"LOND")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_feedback_SA=SAFFRON_feedback(data$p,alpha=alpha,w0=alpha,lambda=0.5,theta=data$theta)
    res=CiterionCompute(rej_feedback_SA$R,data$theta,"SF")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_feedback_LORD=Lord_feedback(data$p,alpha,data$theta,W0=alpha)
    res=CiterionCompute(rej_feedback_LORD$R,data$theta,"LF")
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
                   n = sum(!is.na(FDP)),
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(n),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(n))
pp


pp <- result %>%
  group_by(Method, prop) %>%
  dplyr::summarize(
    FDR = mean(FDP, na.rm = TRUE),
    POWER = mean(Power, na.rm = TRUE),
    
    n_fdp = sum(!is.na(FDP)),
    n_power = sum(!is.na(Power)),
    
    FDP_se = sd(FDP, na.rm = TRUE) / sqrt(n_fdp),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(n_power)
  )

pp$Method=factor(pp$Method,levels = c("SF","LF","SAFFRON","LORD++","LOND"))


head(pp)
P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  ylim(0,0.5)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC", "#7876B1CC", "#6F99ADCC"))+
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


P2 <- ggplot(data = pp,aes(x = prop, y = POWER, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+geom_ribbon(aes(ymin = POWER-Power_se,ymax = POWER+Power_se),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab(TeX("$pi_1$"))+
  ylab("Power")+
  ylim(0,0.8)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29CC", "#0072B5CC", "#E18727CC", "#20854ECC", "#7876B1CC", "#6F99ADCC"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

P2



GAIF_Gaussian <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                           font.label = list(size = 20, face = "bold"))

pdf(file = "GAIF_Gaussian.pdf",width = 10,height = 4) 
GAIF_Gaussian
dev.off()




