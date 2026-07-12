


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

source("functions_OnSel.R")
source("algoclass_OnSel.R")
source("SAFFRON_feedback functions.R")


generate_p_values_Beta <- function(T = 1000, pi1 = 0.1, m = 0.5, n = 5) {
  # Generate indicator for alternative hypothesis
  is_alternative <- rbinom(T, size = 1, prob = pi1)
  
  # Generate p-values based on the model
  p_values <- ifelse(is_alternative == 1, rbeta(T, shape1 = m, shape2 = n), runif(T))
  
  data=data.frame(p=p_values,theta=is_alternative)
  
  return(data)
}

# Example usage
set.seed(123)  # For reproducibility
data <- generate_p_values_Beta()
head(data)

table(data$theta)




N=1000
prop=0.2
alpha=0.1
lambda=0.5
prop_seq <- seq(0.1,0.8,0.1)
#gammai <-0.4374901658 / (seq_len(N)^(1.72))

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  info<-data.frame()
  
  for (prop in prop_seq) {
    
    data <- generate_p_values_Beta(T=N,pi1=prop,m=0.5,n=4)
    
    rej_SAFFRON=SAFFRON(data$p,alpha,w0=alpha/5)
    res=CiterionCompute(rej_SAFFRON$R,data$theta,"SAFFRON")
    res["prop"]=prop
    info=rbind(info,res)
    
    
    rej_LORD=LORD(data$p,alpha,w0=alpha/2)
    res=CiterionCompute(rej_LORD$R,data$theta,"LORD++")
    res["prop"]=prop
    info=rbind(info,res)
    
    # rej_ADDIS=ADDIS(data$p,alpha)
    # res=CiterionCompute(rej_ADDIS$R,data$theta,"ADDIS")
    # res["Signal"]=signal
    # info=rbind(info,res)
    
    rej_LOND=LOND(data$p,alpha)
    res=CiterionCompute(rej_LOND$R,data$theta,"LOND")
    res["prop"]=prop
    info=rbind(info,res)
    
    
    rej_feedback_SA=SAFFRON_feedback(data$p,alpha=alpha,w0=alpha,lambda=0.8,theta=data$theta)
    res=CiterionCompute(rej_feedback_SA$R,data$theta,"SF")
    res["prop"]=prop
    info=rbind(info,res)
    
    rej_feedback_LORD=Lord_feedback(data$p,alpha,data$theta,W0=alpha)
    res=CiterionCompute(rej_feedback_LORD$R,data$theta,"LF")
    res["prop"]=prop
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

pp$Method=factor(pp$Method,levels = c("SF","LF","SAFFRON","LORD++","LOND"))

write.csv(pp,"GAIF_Beta_plot_data.csv")

P1 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+geom_ribbon(aes(ymin = FDR-FDP_se,ymax = FDR+FDP_se),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  ylim(0,0.3)+
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
  ylim(0,1)+
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

GAIF_Beta <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                           font.label = list(size = 20, face = "bold"))

pdf(file = "GAIF_Beta.pdf",width = 10,height = 4) 
GAIF_Beta
dev.off()



