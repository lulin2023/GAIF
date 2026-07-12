###################  mFDR ---------------------------


CriterionCompute_VR <- function(rejection, theta, method_name = " ") {
  V <- sum((1 - theta) * rejection)           
  R <- sum(rejection)                         
  FDP <- V / max(R, 1)                        
  Power <- sum(theta * rejection) / sum(theta)  
  
  return(data.frame(FDR = FDP, Power = Power, V = V, R = R, Method = method_name))
}


alpha <- 0.2
T <- 1000

pi1_seq=seq(0.1,0.8,0.1)

nr=500
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
                      
                      
                      # generate data
                      data <- data_generation_classication1(N=T,mu1= c(1,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=pi_1,pi=pi_1)
                      
                      # generate history data and estimate K (diversity threshold)---
                      his_data <- data_generation_classication1(N=n,mu1= c(1,0,0,0),mu2=c(0,0,-2,-2),p=4,propotion=pi_1,pi=pi_1)
                      
                      p <- ncol(his_data)-1 # dimension of covariates
                      
                      
                      
                      ### some data notations, and index for null data-----
                      datawork=DataSplit(his_data,n,0,n_cal)
                      data_train=datawork$data_train
                      
                      data_cal=datawork$data_cal
                      
                      
                      Null_cal=NullIndex(data_cal$y,Value)
                      
                      
                      X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
                      Y_train=as.matrix(data_train$y)
                      X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
                      Y_cal=as.matrix(data_cal$y)
                      
                      
                      
                      
                      data_test=data
                      Null_test=NullIndex(data_test$y,Value)
                      Alter_test=setdiff(1:length(data_test$y),Null_test)
                      X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
                      Y_test=as.matrix(data_test$y)
                      
                      
                      
                      #theta <- data_test$y
                      theta <- rep(1,T)
                      theta[Null_test]=0
                      
                      # model and estimating locfdr -----
                      
                      model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
                      W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
                      W_test=Pred(algo,model,X_test) #predict classfication score of test data
                      
                      
                      #calculate p-values
                      
                      pval <- confomalPvalue_online(W_cal,W_test,Null_cal,Value,theta)
                      
                      
                      
                      
                      rej_feedback_LORD=Lord_feedback(pval,alpha,theta,W0=alpha/2)
                      res <- CriterionCompute_VR(rej_feedback_LORD$R,theta,"LF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_feedback_LORD_conse=Lord_feedback_conservative(pval,alpha,theta,W0=alpha/2)
                      res <- CriterionCompute_VR(rej_feedback_LORD_conse$R,theta,"LFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_LORD=LORD(pval,alpha,version = "++")
                      res <- CriterionCompute_VR(rej_LORD$R,theta,"LORD++")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      
                      rej_SAFFRON=SAFFRON(pval,alpha,lambda=0.5)
                      res <- CriterionCompute_VR(rej_SAFFRON$R,theta,"SAFFRON")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_SA=SAFFRON_feedback(pval,alpha=alpha,theta=theta,w0=alpha/2,lambda=0.5)
                      res <-CriterionCompute_VR(rej_feedback_SA$R,theta,"SF")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_feedback_SA_conse=SAFFRON_feedback_conservative(pval,alpha,theta,w0=alpha/2,lambda=0.5)
                      res <- CriterionCompute_VR(rej_feedback_SA_conse$R,theta,"SFS")
                      res["prop"]=pi_1
                      info=rbind(info,res)
                      
                      rej_LOND=LOND(pval,alpha)
                      res=CriterionCompute_VR(rej_LOND$R,theta,"LOND")
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
    FDR = mean(FDR),
    POWER = mean(Power),
    FDP_se = sd(FDR, na.rm = TRUE) / sqrt(length(FDR)),
    Power_se = sd(Power, na.rm = TRUE) / sqrt(length(Power)),
    .groups = "drop"
  )

pp

pp$Method=factor(pp$Method,levels =  c("SF","LF","SFS","LFS","SAFFRON",
                                       "LORD++","LOND"))

write.csv(pp,"classification_mFDR.csv")


P1 <- ggplot(data = pp,aes(x = prop, y = mFDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("mFDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
  ))+
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


P2 <- ggplot(data = pp,aes(x = prop, y = FDR, group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.5)+
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1))+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab( TeX("$pi_1$"))+
  ylab("FDR")+
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1), limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.9)+
  scale_fill_manual(values=c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
  ))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 
P2 <- P2 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

P2

cla_plots_vary_prop_mFDR <- ggarrange(P1, P2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                      font.label = list(size = 20, face = "bold"))

pdf(file = "cla_plots_vary_prop_mFDR.pdf",width = 10,height = 4) 
cla_plots_vary_prop_mFDR
dev.off()

write.csv(pp,"cla_plots_vary_prop_mFDR_plot_data.csv")



############ plots ------------------


library(ggplot2)
library(ggpubr)
library(ggsci)
library(latex2exp)
library(dplyr)
library(tidyr)
library(scales)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))


pp1 <- read.csv("cla_plots_vary_prop_mFDR_plot_data.csv")[,-1]
pp2 <- read.csv("reg_plots_vary_prop_mFDR_plot_data.csv")[,-1]

pp1$Scenario <- "Scenario IV"
pp2$Scenario <- "Scenario V"

pp <- rbind(pp1, pp2)


pp$Method <- factor(pp$Method,
                    levels = c("SF","LF","SFS","LFS","SAFFRON","LORD++","LOND"))


pp_long <- pp %>%
  pivot_longer(cols = c(mFDR, Fdr),
               names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels=c("mFDR","FDR")))


alpha <- 0.2   

# ---- 绘图 ----
p_all <- ggplot(pp_long,
                aes(x = prop, y = Value, group = Method,
                    color = Method, shape = Method, fill = Method)) +
  geom_point(size=2.5) +
  geom_line(aes(linetype=Method), linewidth=0.8) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1)) +
  scale_x_continuous(breaks = seq(0.1, 0.8, 0.1),
                     limits = c(0.1, 0.8)) +
  scale_y_continuous(limits = c(0, 0.4),
                     labels = scales::number_format(accuracy = 0.1)) +
  geom_hline(yintercept = alpha, linetype = "dashed", color = "black") +  # ✅ 阈值线
  xlab(TeX("$\\pi_1$")) +
  ylab("") +
  facet_grid(Metric ~ Scenario, scales = "fixed") +   # ✅ 行=Metric，列=Scenario
  theme_bw() +
  scale_color_nejm(palette = "default", alpha = 0.9) +
  scale_fill_manual(values=c(
    "#BC3C29CC", # red
    "#0072B5CC", # blue
    "#E18727CC", # orange
    "#20854ECC", # green
    "#7876B1CC", # purple
    "#6F99ADCC", # cyan
    "#FFDC91CC", # light orange
    "#EE4C97CC", # pink
    "#8F786BCC"  # soft brown 
  )) +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 18),
        strip.text = element_text(size=16, face="bold"),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        panel.grid.major=element_blank(),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        text=element_text(size=14, family="serif"),
        legend.position = "bottom")

# 输出 PDF
pdf("Scenario_IV_V_mFDR_FDR.pdf", width = 8, height = 8)
print(p_all)
dev.off()


setwd(dirname(rstudioapi::getSourceEditorContext()$path))


pp1 <- read.csv("cla_plots_vary_prop_mFDR_plot_data.csv")[,-1]
pp2 <- read.csv("reg_plots_vary_prop_mFDR_plot_data.csv")[,-1]

pp1$Scenario <- "Scenario IV"
pp2$Scenario <- "Scenario V"

pp <- rbind(pp1, pp2)

# Method 因子化，保证顺序
pp$Method <- factor(pp$Method,
                    levels = c("SF","LF","SFS","LFS","SAFFRON","LORD++","LOND"))

# ---- 转成长格式 ----
pp_long <- pp %>%
  pivot_longer(cols = c(mFDR, Fdr),
               names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels=c("mFDR","Fdr")))

# ---- 设置 alpha 阈值 ----
alpha <- 0.2

# ---- 通用绘图函数 ----
make_plot <- function(data, ylab_text="") {
  ggplot(data,
         aes(x = prop, y = Value, group = Method,
             color = Method, shape = Method, fill = Method)) +
    geom_point(size=2.5) +
    geom_line(aes(linetype=Method), linewidth=0.8) +
    scale_shape_manual(values = c(21, 22, 23, 24, 25, 0, 1)) +
    scale_x_continuous(breaks = seq(0.1, 0.8, 0.1),
                       limits = c(0.1, 0.8)) +
    scale_y_continuous(limits = c(0, 0.4),
                       labels = scales::number_format(accuracy = 0.1)) +
    geom_hline(yintercept = alpha, linetype = "dashed", color = "black") +
    xlab(TeX("$\\pi_1$")) +
    ylab(ylab_text) +
    theme_bw() +
    scale_color_nejm(palette = "default", alpha = 0.9) +
    scale_fill_manual(values=c(
      "#BC3C29CC", "#0072B5CC", "#E18727CC",
      "#20854ECC", "#7876B1CC", "#6F99ADCC",
      "#FFDC91CC", "#EE4C97CC", "#8F786BCC"
    )) +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 20),
          legend.text = element_text(size = 16),
          legend.title = element_text(size = 16),
          panel.grid.major=element_blank(),
          panel.background = element_rect(fill = "transparent",colour = NA),
          plot.background = element_rect(fill = "transparent",colour = NA),
          panel.grid.minor = element_blank(),
          text=element_text(size=16, family="serif"),
          legend.position = "bottom")
}

# ---- mFDR 图 ----
p_mfdr_iv <- make_plot(subset(pp_long, Metric=="mFDR" & Scenario=="Scenario IV"), "mFDR") +
  ggtitle("Scenario IV") +
  theme(plot.title = element_text(hjust=0.5, size=18, face="bold"))

p_mfdr_vi <- make_plot(subset(pp_long, Metric=="mFDR" & Scenario=="Scenario V"), "mFDR") +
  ggtitle("Scenario V") +
  theme(plot.title = element_text(hjust=0.5, size=18, face="bold"))

# ---- FDR 图 ----
p_fdr_iv <- make_plot(subset(pp_long, Metric=="Fdr" & Scenario=="Scenario IV"), "FDR")
p_fdr_vi <- make_plot(subset(pp_long, Metric=="Fdr" & Scenario=="Scenario V"), "FDR")

# ---- 拼接 ----
p_mFDR <- ggarrange(p_mfdr_iv, p_mfdr_vi, ncol=2, nrow=1,
                    common.legend = TRUE, legend="bottom")

p_FDR <- ggarrange(p_fdr_iv, p_fdr_vi, ncol=2, nrow=1,
                   common.legend = TRUE, legend="bottom")

p_all <- ggarrange(
  p_mfdr_iv, p_mfdr_vi,
  p_fdr_iv, p_fdr_vi,
  ncol=2, nrow=2,
  common.legend = TRUE, legend="bottom"
)


pdf("Scenario_IV_V_mFDR.pdf", width = 8, height = 4)
print(p_mFDR)
dev.off()

pdf("Scenario_IV_V_FDR.pdf", width = 8, height = 4)
print(p_FDR)
dev.off()

pdf("Scenario_IV_V_mFDR_FDR.pdf", width = 10, height = 8)
print(p_all)
dev.off()

