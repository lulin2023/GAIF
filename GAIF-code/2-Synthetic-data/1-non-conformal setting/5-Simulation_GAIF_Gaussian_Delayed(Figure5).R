
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

base <- ":/Users/GAIF-code"

source(file.path(base, "useful functions", "functions_OnSel.R"))
source(file.path(base, "useful functions", "SAFFRON_feedback functions.R"))


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
    
    rej_SAFFRON=SAFFRON(data$p,alpha)
    res=CiterionCompute(rej_SAFFRON$R,data$theta,"SAFFRON")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_LORD=LORD(data$p,alpha)
    res=CiterionCompute(rej_LORD$R,data$theta,"LORD++")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_LOND=LOND(data$p,alpha)
    res=CiterionCompute(rej_LOND$R,data$theta,"LOND")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    
    rej_feedback_SA=SAFFRON_feedback_delayed(data$p,alpha=alpha,w0=alpha/2,theta=data$theta,d=10)
    res=CiterionCompute(rej_feedback_SA$R,data$theta,"SF-FD")
    res["prop"]=pi_1
    info=rbind(info,res)
    
    rej_feedback_LORD=Lord_feedback_delayed(data$p,alpha,data$theta,W0=alpha/2,d=10)
    res=CiterionCompute(rej_feedback_LORD$R,data$theta,"LF-FD")
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
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power)))
pp



pp$Method=factor(pp$Method,levels = c("SF-FD","LF-FD","SAFFRON","LORD++","LOND"))

write.csv(pp,"GAIF_Gaussian_Delayed_plot_data.csv")


#pp <- read.csv("GAIF_Gaussian_Bandit_plot_data.csv")[,-1]
head(pp)
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



GAIF_Gaussian_Delayed <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                  font.label = list(size = 20, face = "bold"))

pdf(file = "GAIF_Gaussian_Delayed.pdf",width = 10,height = 4) 
GAIF_Gaussian_Delayed
dev.off()
dev.new()

############ results across different d----------


###############################################
### 1. Parameter settings                   ###
###############################################

nr <- 500              # number of repetitions
N  <- 1000                # total hypotheses per run
alpha <- 0.1              # FDR level
pi1_seq <- seq(0.1, 0.8, by = 0.1)  # signal proportion sequence
d_seq <- c(0, 10, 100)   # different delay values
mu_c_values <- 2.5          # signal mean (example)


############################################################
### 2. Start parallel simulation (d = 0,10,100 compared) ###
############################################################

cl <- makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

result <- foreach(iter = 1:nr, .combine = "rbind",
                  .packages = c("MASS", "onlineFDR"),
                  .errorhandling = "remove",
                  .options.snow = opts) %dopar% {
                    
                    info <- data.frame()
                    
                    for (pi_1 in pi1_seq) {
                      
                      ### Generate Gaussian data ###
                      data <- generate_data_Gaussian(T = N, pi_1 = pi_1, mu_c_values)
                      data <- data[sample(nrow(data)), ]  # shuffle
                      
                      ###############################
                      ### 2.1 No-feedback methods ###
                      ###############################
                      
                      ## SAFFRON
                      rej_SAFFRON = SAFFRON(data$p, alpha)
                      res = CiterionCompute(rej_SAFFRON$R, data$theta, "SAFFRON")
                      res["prop"] = pi_1
                      info = rbind(info, res)
                      
                      ## LORD++
                      rej_LORD = LORD(data$p, alpha)
                      res = CiterionCompute(rej_LORD$R, data$theta, "LORD++")
                      res["prop"] = pi_1
                      info = rbind(info, res)
                      
                      ## LOND
                      rej_LOND = LOND(data$p, alpha)
                      res = CiterionCompute(rej_LOND$R, data$theta, "LOND")
                      res["prop"] = pi_1
                      info = rbind(info, res)
                      
                      
                      ##############################################
                      ### 2.2 Feedback methods with delay (d_seq) ###
                      ##############################################
                      
                      for (d in d_seq) {
                        
                        ### SF-FD ###
                        rej_feedback_SA = SAFFRON_feedback_delayed(
                          data$p, alpha = alpha, w0 = alpha/2, theta = data$theta, d = d
                        )
                        res = CiterionCompute(rej_feedback_SA$R, data$theta,
                                              paste0("SF-FD-d", d))
                        res["prop"] = pi_1
                        info = rbind(info, res)
                        
                        
                        ### LF-FD ###
                        rej_feedback_LORD = Lord_feedback_delayed(
                          data$p, alpha, data$theta, W0 = alpha/2, d = d
                        )
                        res = CiterionCompute(rej_feedback_LORD$R, data$theta,
                                              paste0("LF-FD-d", d))
                        res["prop"] = pi_1
                        info = rbind(info, res)
                        
                      }
                    }
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)



###############################################
### 3. Summarize results                     ###
###############################################

pp <- result %>%
  group_by(Method, prop) %>%
  summarize(
    FDR = mean(FDP),
    POWER = mean(Power),
    FDP_se = sd(FDP) / sqrt(n()),
    Power_se = sd(Power) / sqrt(n()),
    .groups = "drop"
  )

pp
# Set factor order (optional)
pp$Method <- factor(pp$Method, levels = c(
  "SF-FD-d0","SF-FD-d10","SF-FD-d100",
  "LF-FD-d0","LF-FD-d10","LF-FD-d100",
  "SAFFRON","LORD++","LOND"
))


table(pp$Method)
###############################################
### 4. Plot FDR curves                       ###
###############################################

P1 <- ggplot(pp, aes(x = prop, y = FDR,
                     color = Method, linetype = Method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = FDR - FDP_se,
                  ymax = FDR + FDP_se,
                  fill = Method),
              alpha = 0.1, color = NA) +
  geom_hline(aes(yintercept = alpha), linetype = "dashed") +
  xlab(TeX("$\\pi_1$")) + ylab("FDR") +
  ylim(0, 0.3) +
  theme_bw() +
  theme(
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  )+
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


###############################################
### 5. Plot Power curves                     ###
###############################################

P2 <- ggplot(pp, aes(x = prop, y = POWER,
                     color = Method, linetype = Method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = POWER - Power_se,
                  ymax = POWER + Power_se,
                  fill = Method),
              alpha = 0.1, color = NA) +
  xlab(TeX("$\\pi_1$")) + ylab("Power") +
  ylim(0, 1) +
  theme_bw() +
  theme(
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom"
  )+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 


###############################################
### 6. Combine and export PDF                ###
###############################################

GAIF_Gaussian_Delayed <- ggarrange(
  P1, P2,
  ncol = 2, nrow = 1,
  common.legend = TRUE,
  legend = "bottom"
)

pdf("GAIF_Gaussian_Delayed.pdf", width = 10, height = 4)
GAIF_Gaussian_Delayed
dev.off()




########################################################
### Color scheme you provided (6 colors)              ###
########################################################

my_colors <- c(
  "SF-FD-d0"  = "#BC3C29FF",  
  "SF-FD-d10" = "#BC3C29AA",  
  "SF-FD-d100" = "#BC3C2955",  
  
  "LF-FD-d0"  = "#0072B5FF",  
  "LF-FD-d10" = "#0072B5AA",  
  "LF-FD-d100" = "#0072B555",  
  
  "SAFFRON" = "#E18727CC",
  "LORD++"  = "#20854ECC",
  "LOND"    = "#7876B1CC"
)


my_fills <- my_colors  # ribbon uses same color but transparent


P1 <- ggplot(pp, aes(x = prop, y = FDR,
                     color = Method, fill = Method, linetype = Method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = FDR - FDP_se,
                  ymax = FDR + FDP_se),
              alpha = 0.12, color = NA) +
  geom_hline(aes(yintercept = alpha),
             linetype = "dashed", color = "black") +
  scale_color_manual(values = my_colors) +
  scale_fill_manual(values = my_fills) +
  xlab(TeX("$\\pi_1$")) + ylab("FDR") +
  ylim(0, 0.3) +
  theme_bw() +
  theme(
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )+
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


P2 <- ggplot(pp, aes(x = prop, y = POWER,
                     color = Method, fill = Method, linetype = Method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = POWER - Power_se,
                  ymax = POWER + Power_se),
              alpha = 0.12, color = NA) +
  scale_color_manual(values = my_colors) +
  scale_fill_manual(values = my_fills) +
  xlab(TeX("$\\pi_1$")) + ylab("Power") +
  ylim(0, 0.8) +
  theme_bw() +
  theme(
    text = element_text(size = 16, family = "serif"),
    legend.position = "bottom",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )+
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

GAIF_Gaussian_Delayed <- ggarrange(
  P1, P2,
  ncol = 2, nrow = 1,
  common.legend = TRUE, legend = "bottom"
)

pdf("GAIF_Gaussian_Delayed.pdf", width = 10, height = 4)
GAIF_Gaussian_Delayed
dev.off()
write.csv(pp,"GAIF_Gaussian_Delayed_plot_data.csv")
print(pp,n=72)
