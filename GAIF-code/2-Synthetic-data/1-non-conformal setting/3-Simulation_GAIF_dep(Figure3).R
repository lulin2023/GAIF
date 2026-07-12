

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
source("local dependence func.R")




mu_c_values <- 2.5

N=1000
#prop=0.1
alpha=0.1
lambda=0.5
pi1_seq=seq(0.1,0.8,0.1)

library(latex2exp)

nr=500
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR","latex2exp"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  info<-data.frame()
  
  for (pi_1 in pi1_seq) {
   
    
    t_max <- 1000
    data <- block_cov.data_new(t_max=t_max,mean_value=mu_c_values,proportion=pi_1,rho=0.8,nbatch=10)
    
    head(data)
    #data=data[sample(nrow(data)), ]
    df <- data.frame(pval=data$p,lags=rep(10,200))
    
    rej_SAFFRON=SAFFRON(data$pvalue,alpha)
    res=CiterionCompute(rej_SAFFRON$R,data$theta,"SAFFRON")
    res["prop"]=pi_1
    
    
    info=rbind(info,res)
    
    rej_SAFFRON=SAFFRONstar(df,alpha,version = "dep")
    #rej_SAFFRON=saffronstar_dep(pval=data$pvalue,L=rep(5,200),alpha=alpha)
    res=CiterionCompute(rej_SAFFRON$R,data$theta,"SAFFRONdep")
    res["prop"]=pi_1
    
    info=rbind(info,res)
    
    
    rej_LORD=LORD(data$pvalue,alpha)
    res=CiterionCompute(rej_LORD$R,data$theta,"LORD++")
    res["prop"]=pi_1
   
    info=rbind(info,res)
    
     df <- data.frame(pval=data$pvalue,lags=rep(5,200))
    
    rej_LORD=LORDstar(df,alpha,version = "dep")
    res=CiterionCompute(rej_LORD$R,data$theta,"LORDdep")
    res["prop"]=pi_1
   
    info=rbind(info,res)
    
    # rej_ADDIS=ADDIS(data$p,alpha)
    # res=CiterionCompute(rej_ADDIS$R,data$theta,"ADDIS")
    # res["Signal"]=signal
    # info=rbind(info,res)
    
    rej_LOND=LOND(data$pvalue,alpha)
    res=CiterionCompute(rej_LOND$R,data$theta,"LOND")
    res["prop"]=pi_1
    
    info=rbind(info,res)
    
    
    rej_feedback_SA=SAFFRON_feedback(data$p,alpha=alpha,w0=alpha/2,theta=data$theta)
    res=CiterionCompute(rej_feedback_SA$R,data$theta,"SF")
    res["prop"]=pi_1
   
    info=rbind(info,res)
    
    rej_feedback_SA=SAFFRON_feedback_dep(pval=data$p,L=df$lags,alpha=alpha,w0=alpha/2,theta=data$theta)
    res=CiterionCompute(rej_feedback_SA$R,data$theta,"SFdep")
    res["prop"]=pi_1
    
    info=rbind(info,res)
    
    rej_feedback_LORD=Lord_feedback_dep(pval=data$p,L=df$lags,alpha=alpha,theta=data$theta,W0=alpha/2)
    res=CiterionCompute(rej_feedback_LORD$R,data$theta,"LFdep")
    res["prop"]=pi_1
    
    info=rbind(info,res)
    
    rej_feedback_LORD=Lord_feedback(pval=data$p,alpha=alpha,theta=data$theta,W0=alpha/2)
    res=CiterionCompute(rej_feedback_LORD$R,data$theta,"LF")
    res["prop"]=pi_1
    
    info=rbind(info,res)
    
  }
  
  
  return(info)
}
close(pb)
stopCluster(cl)

result1 <- result


pp1 <- result%>%
  group_by(Method,prop)%>%
  dplyr::summarize(FDR=mean(FDP),POWER=mean(Power),
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power)))
pp1

print(pp1,n=72)
pp1 <- result2%>%
  group_by(Method,prop)%>%
  dplyr::summarize(FDR=mean(FDP),POWER=mean(Power),
                   FDP_se = sd(FDP, na.rm = TRUE)/sqrt(length(FDP)),
                   Power_se = sd(Power, na.rm = TRUE)/sqrt(length(Power)))
pp1

pp1$Method=factor(pp1$Method,levels = c("SFdep","SF","LFdep","LF",
                                        "SAFFRONdep","SAFFRON",
                                      "LORDdep","LORD++","LOND"))
table(pp1$Method)







library(ggplot2)
library(latex2exp)
library(dplyr)

library(ggplot2)
library(latex2exp)
library(dplyr)


cols   <- c(
  "#BC3C29CC", # red
  "#0072B5CC", # blue
  "#E18727CC", # orange
  "#20854ECC", # green
  "#7876B1CC", # purple
  "#6F99ADCC", # cyan
  "#FFDC91CC", # light orange
  "#EE4C97CC", # pink
  "#8F786BCC"  # soft brown 
)

# 全部实心点型
shapes <- c(19, 17, 15, 18, 21, 22, 23, 24, 25)

pp2 <- pp1 %>%
  filter(!is.na(FDR), !is.na(FDP_se))

P1 <- ggplot(pp2, aes(x = prop, y = FDR,
                      group  = Method,
                      colour = Method,
                      fill   = Method,
                      shape  = Method)) +
  geom_ribbon(aes(ymin = FDR - FDP_se, ymax = FDR + FDP_se),
              alpha = 0.1, linetype = 1, colour = NA) +
  geom_line(aes(linetype = Method), size = 0.8, na.rm = TRUE) +
  geom_point(size = 2.5, na.rm = TRUE) +
  geom_hline(yintercept = alpha, colour = "black", linetype = "dashed") +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values   = cols) +
  scale_shape_manual(values  = shapes) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  ylim(0, 0.5) +
  theme_bw() +
  theme(
    axis.text        = element_text(size = 16),
    axis.title       = element_text(size = 20),
    legend.text      = element_text(size = 16),
    legend.title     = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    text             = element_text(size = 16, family = "serif"),
    legend.position  = "bottom"
  )

print(P1)

# nejm_9_colors <- c(
#   "#BC3C29CC", # red
#   "#0072B5CC", # blue
#   "#E18727CC", # orange
#   "#20854ECC", # green
#   "#7876B1CC", # purple
#   "#6F99ADCC", # cyan
#   "#FFDC91CC", # light orange
#   "#EE4C97CC", # pink
#   "#8F786BCC"  # soft brown 
# )




cols       <- c(
  "#BC3C29CC", # red
  "#0072B5CC", # blue
  "#E18727CC", # orange
  "#20854ECC", # green
  "#7876B1CC", # purple
  "#6F99ADCC", # cyan
  "#FFDC91CC", # light orange
  "#EE4C97CC", # pink
  "#8F786BCC"  # soft brown 
)

shapes <- c(19, 17, 15, 18, 21, 22, 23, 24, 25)

pp2 <- pp1 %>%
  filter(!is.na(POWER), !is.na(Power_se))

P2 <- ggplot(pp2, aes(x = prop, y = POWER,
                      group  = Method,
                      colour = Method,
                      fill   = Method,
                      shape  = Method)) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1, linetype = 1, colour = NA) +
  geom_line(aes(linetype = Method), size = 0.8, na.rm = TRUE) +
  geom_point(size = 2.5, na.rm = TRUE) +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values   = cols) +
  scale_shape_manual(values  = shapes) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  ylim(0, 1) +
  theme_bw() +
  theme(
    axis.text       = element_text(size = 16),
    axis.title      = element_text(size = 20),
    legend.text     = element_text(size = 16),
    legend.title    = element_text(size = 16),
    panel.grid      = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    text             = element_text(size = 16, family = "serif"),
    legend.position  = "bottom"
  )

print(P2)


GAIF_dep2 <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                      font.label = list(size = 20, face = "bold"))

pdf(file = "GAIF_dep2.pdf",width = 10,height = 4) 
GAIF_dep2
dev.off()

pp1
write.csv(pp1,"GAIF_local_dep.csv")
print(pp1,n=72)

############  plots ------------

# 分组
pp_saffron <- pp1 %>% filter(grepl("SAFFRON", Method) | grepl("SF", Method))
pp_lord    <- pp1 %>% filter(grepl("LORD", Method) | grepl("LF", Method) | Method == "LOND")

# 可选配色（按实际方法顺序调整）
cols_saffron <- c(
  "#BC3C29CC", "#0072B5CC",  # SFdep, SF
  "#7876B1CC", "#6F99ADCC"   # SAFFRONdep, SAFFRON
)
shapes_saffron <- c(19, 17, 21, 22)

cols_lord <- c(
  "#E18727CC", "#20854ECC",  # LFdep, LF
  "#FFDC91CC", "#EE4C97CC",  # LORDdep, LORD++
  "#8F786BCC"                # LOND
)
shapes_lord <- c(15, 18, 23, 24, 25)

plot_curve <- function(df, ycol, ysecol, ylab, ylim_range, cols, shapes, title = NULL, add_hline = FALSE) {
  ggplot(df, aes(x = prop, y = .data[[ycol]],
                 group = Method, colour = Method,
                 fill = Method, shape = Method)) +
    geom_ribbon(aes(ymin = .data[[ycol]] - .data[[ysecol]],
                    ymax = .data[[ycol]] + .data[[ysecol]]),
                alpha = 0.1, linetype = 1, colour = NA) +
    geom_line(aes(linetype = Method), linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2.5, na.rm = TRUE) +
    {if (add_hline) geom_hline(yintercept = alpha, colour = "black", linetype = "dashed")} +
    scale_colour_manual(values = cols) +
    scale_fill_manual(values = cols) +
    scale_shape_manual(values = shapes) +
    xlab(TeX("$\\pi_1$")) +
    ylab(ylab) +
    ylim(ylim_range[1], ylim_range[2]) +
    theme_bw() +
    theme(
      axis.text       = element_text(size = 16),
      axis.title      = element_text(size = 20),
      legend.text     = element_text(size = 16),
      legend.title    = element_text(size = 16),
      panel.grid      = element_blank(),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background  = element_rect(fill = "transparent", colour = NA),
      text             = element_text(size = 16, family = "serif"),
      legend.position  = "bottom",
      plot.title       = element_text(size = 18, hjust = 0.5)
    ) +
    ggtitle(title)
}

# SAFFRON 系列图
pp_saffron_fdr   <- pp_saffron %>% filter(!is.na(FDR), !is.na(FDP_se))
pp_saffron_power <- pp_saffron %>% filter(!is.na(POWER), !is.na(Power_se))

p_saf_fdr   <- plot_curve(pp_saffron_fdr,   "FDR", "FDP_se", "FDR",   c(0, 0.5), cols_saffron, shapes_saffron, add_hline = TRUE)
p_saf_power <- plot_curve(pp_saffron_power, "POWER", "Power_se", "Power", c(0, 1), cols_saffron, shapes_saffron)

# LORD 系列图
pp_lord_fdr   <- pp_lord %>% filter(!is.na(FDR), !is.na(FDP_se))
pp_lord_power <- pp_lord %>% filter(!is.na(POWER), !is.na(Power_se))

p_lord_fdr   <- plot_curve(pp_lord_fdr,   "FDR", "FDP_se", "FDR",   c(0, 0.5), cols_lord, shapes_lord, add_hline = TRUE)
p_lord_power <- plot_curve(pp_lord_power, "POWER", "Power_se", "Power", c(0, 1), cols_lord, shapes_lord)

plot_SAFFRON <- ggarrange(p_saf_fdr, p_saf_power, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                  font.label = list(size = 16, face = "bold"))


plot_LORD <- ggarrange(p_lord_fdr ,p_lord_power, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                               font.label = list(size = 16, face = "bold"))



plot_GAIF_dep <- ggarrange(plot_LORD,plot_SAFFRON, ncol=1, nrow=2,
                          common.legend = TRUE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))
pdf(file = "GAIF_dep.pdf",width = 10,height = 8) 
plot_GAIF_dep
dev.off()



#########。20260619 mFDR ########


CriterionCompute_VR <- function(rejection, theta, method_name = " ") {
  V <- sum((1 - theta) * rejection)           # 假发现数
  R <- sum(rejection)                         # 总拒绝数
  FDP <- V / max(R, 1)                        # 假发现比例（FDR）
  Power <- sum(theta * rejection) / sum(theta)  # Power
  
  return(data.frame(FDR = FDP, Power = Power, V = V, R = R, Method = method_name))
}


alpha <- 0.1
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=100
cl = makeCluster(10)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "onlineFDR","randomForest","caret","nnet",
                                                                 "glmnet","reshape2","kedd","kernlab","e1071","ks",
                                                                 "tidyverse","lsa","magrittr","doSNOW","pbmcapply"),
                  .errorhandling = "remove", .options.snow = opts)%dopar% {
                    info<-data.frame()
                    
                    for (pi_1 in pi1_seq) {
                      # Z0=rnorm(N*(1-prop),0,1)
                      # Z1=rnorm(N*(prop),signal,1)
                      
                      #data=data.frame(Z=c(Z0,Z1),p=1-pnorm(c(Z0,Z1)),theta=c(rep(0,N*(1-prop)),rep(1,N*(prop) ) ) )
                      
                      # data <- generate_data_Gaussian(N, pi_1, mu_c_values)
                      
                      t_max <- 200
                      data <- block_cov.data_new(t_max=t_max,mean_value=mu_c_values,proportion=pi_1,rho=0.8,nbatch=10)
                      
                      head(data)
                      #data=data[sample(nrow(data)), ]
                      df <- data.frame(pval=data$p,lags=rep(10,200))
                      
                      rej_SAFFRON=SAFFRON(data$pvalue,alpha)
                      res=CriterionCompute_VR(rej_SAFFRON$R,data$theta,"SAFFRON")
                      res["prop"]=pi_1
                      
                      
                      info=rbind(info,res)
                      
                      rej_SAFFRON=SAFFRONstar(df,alpha,version = "dep")
                      #rej_SAFFRON=saffronstar_dep(pval=data$pvalue,L=rep(5,200),alpha=alpha)
                      res=CriterionCompute_VR(rej_SAFFRON$R,data$theta,"SAFFRONdep")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      
                      rej_LORD=LORD(data$pvalue,alpha)
                      res=CriterionCompute_VR(rej_LORD$R,data$theta,"LORD++")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      df <- data.frame(pval=data$pvalue,lags=rep(5,200))
                      
                      rej_LORD=LORDstar(df,alpha,version = "dep")
                      res=CriterionCompute_VR(rej_LORD$R,data$theta,"LORDdep")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      # rej_ADDIS=ADDIS(data$p,alpha)
                      # res=CiterionCompute(rej_ADDIS$R,data$theta,"ADDIS")
                      # res["Signal"]=signal
                      # info=rbind(info,res)
                      
                      rej_LOND=LOND(data$pvalue,alpha)
                      res=CriterionCompute_VR(rej_LOND$R,data$theta,"LOND")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      
                      rej_feedback_SA=SAFFRON_feedback(data$p,alpha=alpha,w0=alpha/2,theta=data$theta)
                      res=CriterionCompute_VR(rej_feedback_SA$R,data$theta,"SF")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      rej_feedback_SA=SAFFRON_feedback_dep(pval=data$p,L=df$lags,alpha=alpha,w0=alpha/2,theta=data$theta)
                      res=CriterionCompute_VR(rej_feedback_SA$R,data$theta,"SFdep")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      rej_feedback_LORD=Lord_feedback_dep(pval=data$p,L=df$lags,alpha=alpha,theta=data$theta,W0=alpha/2)
                      res=CriterionCompute_VR(rej_feedback_LORD$R,data$theta,"LFdep")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                      rej_feedback_LORD=Lord_feedback(pval=data$p,alpha=alpha,theta=data$theta,W0=alpha/2)
                      res=CriterionCompute_VR(rej_feedback_LORD$R,data$theta,"LF")
                      res["prop"]=pi_1
                      
                      info=rbind(info,res)
                      
                    }
                    
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)

pp <- result %>%
  group_by(Method, prop) %>%
  summarize(
    mFDR = mean(V) / pmax(mean(R), 1),
    Fdr = mean(FDR),
    POWER = mean(Power),
    FDP_se = sd(FDR, na.rm = TRUE) / sqrt(length(FDR)),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(length(Power)),
    .groups = "drop"
  )

pp

pp$Method=factor(pp$Method,levels =  c("SFdep","SF","LFdep","LF",
                                       "SAFFRONdep","SAFFRON",
                                       "LORDdep","LORD++","LOND"))

#write.csv(pp,"Gaussian_dep_mFDR.csv")


library(ggplot2)
library(latex2exp)
library(dplyr)

library(ggplot2)
library(latex2exp)
library(dplyr)


cols   <- c(
  "#BC3C29CC", # red
  "#0072B5CC", # blue
  "#E18727CC", # orange
  "#20854ECC", # green
  "#7876B1CC", # purple
  "#6F99ADCC", # cyan
  "#FFDC91CC", # light orange
  "#EE4C97CC", # pink
  "#8F786BCC"  # soft brown 
)

# 全部实心点型
shapes <- c(19, 17, 15, 18, 21, 22, 23, 24, 25)

pp2 <- pp %>%
  filter(!is.na(mFDR), !is.na(FDP_se))

P1 <- ggplot(pp2, aes(x = prop, y = mFDR,
                      group  = Method,
                      colour = Method,
                      fill   = Method,
                      shape  = Method)) +
  geom_ribbon(aes(ymin = mFDR - FDP_se, ymax = mFDR + FDP_se),
              alpha = 0.1, linetype = 1, colour = NA) +
  geom_line(aes(linetype = Method), size = 0.8, na.rm = TRUE) +
  geom_point(size = 2.5, na.rm = TRUE) +
  geom_hline(yintercept = alpha, colour = "black", linetype = "dashed") +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values   = cols) +
  scale_shape_manual(values  = shapes) +
  xlab(TeX("$\\pi_1$")) +
  ylab("mFDR") +
  ylim(0, 1) +
  theme_bw() +
  theme(
    axis.text        = element_text(size = 16),
    axis.title       = element_text(size = 20),
    legend.text      = element_text(size = 16),
    legend.title     = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    text             = element_text(size = 16, family = "serif"),
    legend.position  = "bottom"
  )

print(P1)

# nejm_9_colors <- c(
#   "#BC3C29CC", # red
#   "#0072B5CC", # blue
#   "#E18727CC", # orange
#   "#20854ECC", # green
#   "#7876B1CC", # purple
#   "#6F99ADCC", # cyan
#   "#FFDC91CC", # light orange
#   "#EE4C97CC", # pink
#   "#8F786BCC"  # soft brown 
# )

P2 <- ggplot(pp2, aes(x = prop, y = Fdr,
                      group  = Method,
                      colour = Method,
                      fill   = Method,
                      shape  = Method)) +
  geom_ribbon(aes(ymin = Fdr - FDP_se, ymax = Fdr + FDP_se),
              alpha = 0.1, linetype = 1, colour = NA) +
  geom_line(aes(linetype = Method), size = 0.8, na.rm = TRUE) +
  geom_point(size = 2.5, na.rm = TRUE) +
  geom_hline(yintercept = alpha, colour = "black", linetype = "dashed") +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values   = cols) +
  scale_shape_manual(values  = shapes) +
  xlab(TeX("$\\pi_1$")) +
  ylab("FDR") +
  ylim(0, 1) +
  theme_bw() +
  theme(
    axis.text        = element_text(size = 16),
    axis.title       = element_text(size = 20),
    legend.text      = element_text(size = 16),
    legend.title     = element_text(size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    text             = element_text(size = 16, family = "serif"),
    legend.position  = "bottom"
  )

print(P2)



plot_GAIF_dep_mFDR <- ggarrange(P1,P2, ncol=2, nrow=1,
                           common.legend = TRUE, legend="bottom",
                           font.label = list(size = 16, face = "bold"))
pdf(file = "GAIF_dep_mFDR.pdf",width = 10,height = 4) 
plot_GAIF_dep_mFDR
dev.off()



cols       <- c(
  "#BC3C29CC", # red
  "#0072B5CC", # blue
  "#E18727CC", # orange
  "#20854ECC", # green
  "#7876B1CC", # purple
  "#6F99ADCC", # cyan
  "#FFDC91CC", # light orange
  "#EE4C97CC", # pink
  "#8F786BCC"  # soft brown 
)

shapes <- c(19, 17, 15, 18, 21, 22, 23, 24, 25)

pp2 <- pp1 %>%
  filter(!is.na(POWER), !is.na(Power_se))

P2 <- ggplot(pp2, aes(x = prop, y = POWER,
                      group  = Method,
                      colour = Method,
                      fill   = Method,
                      shape  = Method)) +
  geom_ribbon(aes(ymin = POWER - Power_se, ymax = POWER + Power_se),
              alpha = 0.1, linetype = 1, colour = NA) +
  geom_line(aes(linetype = Method), size = 0.8, na.rm = TRUE) +
  geom_point(size = 2.5, na.rm = TRUE) +
  scale_colour_manual(values = cols) +
  scale_fill_manual(values   = cols) +
  scale_shape_manual(values  = shapes) +
  xlab(TeX("$\\pi_1$")) +
  ylab("Power") +
  ylim(0, 1) +
  theme_bw() +
  theme(
    axis.text       = element_text(size = 16),
    axis.title      = element_text(size = 20),
    legend.text     = element_text(size = 16),
    legend.title    = element_text(size = 16),
    panel.grid      = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    text             = element_text(size = 16, family = "serif"),
    legend.position  = "bottom"
  )

print(P2)

