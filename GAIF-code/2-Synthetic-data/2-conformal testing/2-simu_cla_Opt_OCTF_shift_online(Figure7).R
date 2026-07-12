############ distribution shift 
mvrnorm_new <- function(theta,mu1,mu2,d){
  data <- mvrnorm(n=2,(1-theta)*mu1+theta*mu2,Sigma=diag(d))[1,]
  return(data)
}

data_gen_cla_shift_smooth <- function(m = 1000, 
                                      pattern = c('constant', 'blocks', 'linear-des','linear-inc', 'sine'), 
                                      d = 4, 
                                      mu1 = c(5, 0, 0, 0), 
                                      mu2 = c(0, 0, -3, -2)) {
  pattern <- match.arg(pattern)
  p <- rep(0, m)
  
  if (pattern == 'constant') {
    p <- rep(0.2, m)
    
  } else if (pattern == 'sine') {
    p <- (sin(100 * pi * (1:m) / m) + 1) / 4
    
  } 
  Y <- map_dbl(p, ~sample(c(0,1), 1, prob = c(1 - .x, .x)))
  
  theta <- Y
  X <- matrix(unlist(lapply(theta, function(y) mvrnorm_new(y, mu1, mu2, d))), length(theta), d, byrow = TRUE)
  data <- as.data.frame(cbind(X, Y))
  names(data)[d + 1] <- 'y'
  return(data)
}



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

##################  ----------------

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))
algo1<- new("SVM") #algorithm used for classification or regression
lambda1<- 2 #specific parameter for the algorithm
algo2<- new("RFc") #algorithm used for classification or regression
lambda2<- 600 #specific parameter for the algorithm
algo3<- new("NN") #algorithm used for classification or regression




################# online plot------------


alpha <- 0.05
T <- 1000
n <- 1000
n_train<- round(n/2) # number of data used for training model
n_cal <- round(n/2)

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
                    
                    
                    
                    
                    data <- data_gen_cla_shift_smooth(m=T,pattern = "sine",d=4,mu1=c(3,0,0,0),mu2 = c(0,0,-1,-1))
                    his_data <- data_gen_cla_shift_smooth(m=n,pattern = "constant",d=4,mu1=c(3,0,0,0),mu2 = c(0,0,-1,-1))
                    
                    
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
                    
                    theta <- Y_test
                    
                    
                    # training K = 3 models and calculate conformal p-values -----
                    
                    ## Model1: SVM
                    
                    
                    model1=fitting(algo1,X_train,Y_train,lambda1) #estimate model by training data
                    W_cal1=Pred(algo1,model1,X_cal)[,2] #predict classfication score of calibration data
                    W_test1=Pred(algo1,model1,X_test)[,2] #predict classfication score of test data
                    
                    ## Model2: RF
                    
                    
                    model2=fitting(algo2,X_train,Y_train,lambda2) #estimate model by training data
                    W_cal2=Pred(algo2,model2,X_cal) #predict classfication score of calibration data
                    W_test2=Pred(algo2,model2,X_test) #predict classfication score of test data
                    
                    ## Model3: NN
                    
                    
                    model3=fitting(algo3,X_train,Y_train,lambda) #estimate model by training data
                    W_cal3=Pred(algo3,model3,X_cal) #predict classfication score of calibration data
                    W_test3=Pred(algo3,model3,X_test) #predict classfication score of test data
                    
                    
                    
                    
                    W_cal_all <- as.data.frame(cbind(W_cal1,W_cal2,W_cal3))
                    
                    
                    W_test_all <- as.data.frame(cbind(W_test1,W_test2,W_test3))
                    
                    
                    
                    pvals_random <- confomalPvalue_online_random(W_cal_all, W_test_all, Null_cal, Value, theta)
                    
                    pvals_opt <- confomalPvalue_online_opt_nonNull_EWMA(W_cal_all, W_test_all, Null_cal, Value,theta,lambda = 0.95, L = 200)
                    
                    
                    rej_opt.lordf <- Lord_feedback(pvals_opt,alpha,theta,W0=alpha/2)
                    res_Opt_LF_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.lordf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    
                    result.LF.opt <- list(FDP=res_Opt_LF_online$FDP,
                                          Power=res_Opt_LF_online$Power,
                                          time=res_Opt_LF_online$time,
                                          method="Opt-LF")
                    
                    rej_random.lordf <- Lord_feedback(pvals_random,alpha,theta,W0=alpha/2)
                    res_Ran_LF_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.lordf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.LF.ran <- list(FDP=res_Ran_LF_online$FDP,
                                          Power=res_Ran_LF_online$Power,
                                          time=res_Ran_LF_online$time,
                                          method="Ran-LF")
                    
                    
                    rej_opt.lordf.conse <- Lord_feedback_conservative(pvals_opt,alpha,theta,W0=alpha/2)
                    res_Opt_LFS_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.lordf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.LFS.opt <- list(FDP=res_Opt_LFS_online$FDP,
                                           Power=res_Opt_LFS_online$Power,
                                           time=res_Opt_LFS_online$time,
                                           method="Opt-LFS")
                    
                    rej_random.lordf.conse <- Lord_feedback_conservative(pvals_random,alpha,theta,W0=alpha/2)
                    res_Ran_LFS_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.lordf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.LFS.ran <- list(FDP=res_Ran_LFS_online$FDP,
                                           Power=res_Ran_LFS_online$Power,
                                           time=res_Ran_LFS_online$time,
                                           method="Ran-LFS")
                    
                    rej_opt.saf <- SAFFRON_feedback(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                    res_Opt_SF_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.saf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.SF.opt <- list(FDP=res_Opt_SF_online$FDP,
                                          Power=res_Opt_SF_online$Power,
                                          time=res_Opt_SF_online$time,
                                          method="Opt-SF")
                    
                    
                    
                    rej_random.saf <- SAFFRON_feedback(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                    res_Ran_SF_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.saf$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.SF.ran <- list(FDP=res_Ran_SF_online$FDP,
                                          Power=res_Ran_SF_online$Power,
                                          time=res_Ran_SF_online$time,
                                          method="Ran-SF")
                    
                    
                    rej_opt.saf.conse <- SAFFRON_feedback_conservative(pvals_opt,alpha=alpha,theta=theta,w0=alpha/2)
                    res_Opt_SFS_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_opt.saf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    result.SFS.opt <- list(FDP=res_Opt_SFS_online$FDP,
                                           Power=res_Opt_SFS_online$Power,
                                           time=res_Opt_SFS_online$time,
                                           method="Opt-SFS")
                    
                    
                    rej_random.saf.conse <- SAFFRON_feedback_conservative(pvals_random,alpha=alpha,theta=theta,w0=alpha/2)
                    res_Ran_SFS_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_random.saf.conse$R , .x))%>% unlist  %>% split(.,names(.))
                    result.SFS.ran <- list(FDP=res_Ran_SFS_online$FDP,
                                           Power=res_Ran_SFS_online$Power,
                                           time=res_Ran_SFS_online$time,
                                           method="Ran-SFS")
                    
                    rej_LORD_random <- LORD(pvals_random,alpha)
                    res_Ran_LORD_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_LORD_random$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.LORD.ran <- list(FDP=res_Ran_LORD_online$FDP,
                                            Power=res_Ran_LORD_online$Power,
                                            time=res_Ran_LORD_online$time,
                                            method="LORD++")
                    
                    rej_SA_random <- SAFFRON(pvals_random,alpha)
                    res_Ran_SA_online <- 1:T %>% map(~CiterionCompute_each(Alter_test, decisions=rej_SA_random$R , .x))%>% unlist  %>% split(.,names(.))
                    
                    result.SA.ran <- list(FDP=res_Ran_SA_online$FDP,
                                          Power=res_Ran_SA_online$Power,
                                          time=res_Ran_SA_online$time,
                                          method="SAFFRON")
                    
                    info=list(
                      result.LF.opt, result.LF.ran,
                      result.LFS.opt, result.LFS.ran,
                      result.SF.opt, result.SF.ran,
                      result.SFS.opt, result.SFS.ran,
                      result.LORD.ran,result.SA.ran
                    ) 
                    
                    
                    return(info)
                  }

close(pb)
stopCluster(cl)

result

str(result[[1]])



pp <- result %>%
  map_dfr(~{
    data <- bind_rows(
      tibble(FDP = .x$FDP, Power = .x$Power, 
             Method = .x$method, Time=.x$time)
    )
    return(data)
  })

head(pp)
summary(pp)


pp$Method=factor(pp$Method,levels = c("Opt-SF", "Ran-SF",
                                      "Opt-SFS", "Ran-SFS",
                                      "Opt-LF", "Ran-LF",
                                      "Opt-LFS", "Ran-LFS",
                                      "SAFFRON", "LORD++"))


pp <- pp[!pp$Method %in% c("Opt-SFS", "Ran-SFS","Opt-LFS", "Ran-LFS"), ]

# ==== Load packages ====
library(ggplot2)
library(dplyr)
library(patchwork)

# ==== Define time points ====
t <- c(seq(10, 1000, 100),1000)

# ==== Method order and colors ====
method_levels <- c("Opt-SF", "Ran-SF",
                   "Opt-SFS", "Ran-SFS",
                   "Opt-LF", "Ran-LF",
                   "Opt-LFS", "Ran-LFS",
                   "SAFFRON", "LORD++")

method_colors <- c(
  "#BC3C29", "#F4A582",
  "#E18727", "#FDBF6F",
  "#0072B5", "#A6CEE3",
  "#20854E", "#B2DF8A",
  "#984EA3", "#CAB2D6"
)
names(method_colors) <- method_levels

shape_values <- c(21, 21, 22, 22, 23, 23, 24, 24, 25, 25)
names(shape_values) <- method_levels

linetype_values <- c("solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "solid", "dashed",
                     "dashed", "dashed")
names(linetype_values) <- method_levels

# ==== Set factor order ====
pp$Method <- factor(pp$Method, levels = method_levels)

# ==== Split data ====
pp_saffron <- pp %>% filter(Method %in% c("Opt-SF", "Ran-SF", "Opt-SFS", "Ran-SFS", "SAFFRON"))
pp_lord    <- pp %>% filter(Method %in% c("Opt-LF", "Ran-LF", "Opt-LFS", "Ran-LFS", "LORD++"))

# ==== Compute summary ====
summary_by_group <- function(data) {
  data %>%
    filter(Time %in% t) %>%
    group_by(Time, Method) %>%
    summarize(
      FDP_avg = mean(FDP, na.rm = TRUE),
      Power_avg = mean(Power, na.rm = TRUE),
      FDP_se = sd(FDP, na.rm = TRUE)/sqrt(n()),
      Power_se = sd(Power, na.rm = TRUE)/sqrt(n()),
      .groups = "drop"
    )
}

pp_saffron_summary <- summary_by_group(pp_saffron)
pp_lord_summary <- summary_by_group(pp_lord)

print(pp_saffron_summary,n=45)

# ==== Plotting function ====
plot_metric <- function(df, y_avg, y_se, y_label, y_lim, add_alpha_line = FALSE) {
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
      legend.text = element_text(size = 16),
      legend.title = element_text(size = 16),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      text = element_text(size = 16, family = "serif")
    )
  
  if (add_alpha_line) {
    p <- p + geom_hline(aes(yintercept = alpha), colour = "black", linetype = "dashed")
  }
  
  return(p)
}


p_saf_fdp   <- plot_metric(pp_saffron_summary, "FDP_avg", "FDP_se", "FDR", c(0, 0.3), add_alpha_line = TRUE)
p_saf_power <- plot_metric(pp_saffron_summary, "Power_avg", "Power_se", "Power", c(0, 0.5), add_alpha_line = FALSE)

p_lord_fdp   <- plot_metric(pp_lord_summary, "FDP_avg", "FDP_se", "FDR", c(0, 0.3), add_alpha_line = TRUE)
p_lord_power <- plot_metric(pp_lord_summary, "Power_avg", "Power_se", "Power", c(0, 0.5), add_alpha_line = FALSE)

p_saf_fdp 
p_saf_power
p_lord_fdp  
p_lord_power 


plot_cla_Opt_SAFFRON <- ggarrange(p_saf_fdp, p_saf_power, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                                  font.label = list(size = 16, face = "bold"))
pdf(file = "plot_cla_Opt_SAFFRON.pdf",width = 10,height = 4) 
plot_cla_Opt_SAFFRON
dev.off()
dev.new()


plot_cla_Opt_LORD <- ggarrange(p_lord_fdp ,p_lord_power, ncol=2, nrow=1, common.legend = TRUE, legend="bottom",
                               font.label = list(size = 16, face = "bold"))

pdf(file = "plot_cla_Opt_LORD.pdf",width = 10,height = 4) 
plot_cla_Opt_LORD
dev.off()
dev.new()

plot_cla_Opt <- ggarrange(plot_cla_Opt_LORD,plot_cla_Opt_SAFFRON, ncol=2, nrow=1,
                          common.legend = FALSE, legend="bottom",
                          font.label = list(size = 16, face = "bold"))
pdf(file = "plot_cla_Opt_vary_prop_sine_shift.pdf",width = 12,height = 4) 
plot_cla_Opt 
dev.off()
dev.new()

t <- 1000
pp_saffron_summary_stop <- summary_by_group(pp_saffron)
pp_lord_summary_stop <- summary_by_group(pp_lord)

pp_saffron_summary_stop %>%
  arrange(Method)



# ==== Filter methods ====
pp_combined <- pp %>% 
  filter(Method %in% c("Opt-SF", "Ran-SF", "Opt-LF", "Ran-LF", "SAFFRON", "LORD++"))

pp_combined$Method <- factor(pp_combined$Method, levels = c(
  "Opt-SF", "Ran-SF", "Opt-LF", "Ran-LF", "SAFFRON", "LORD++"
))

# ==== Colors, shapes, linetypes ====
method_levels_combined <- c("Opt-SF", "Ran-SF", "Opt-LF", "Ran-LF", "SAFFRON", "LORD++")

method_colors_combined <- c(
  "#BC3C29", "#F4A582",
  "#0072B5", "#A6CEE3",
  "#984EA3", "#20854E"
)
names(method_colors_combined) <- method_levels_combined

shape_values_combined <- c(21, 21, 23, 23, 25, 25)
names(shape_values_combined) <- method_levels_combined

linetype_values_combined <- c("solid", "dashed", "solid", "dashed", "solid", "dashed")
names(linetype_values_combined) <- method_levels_combined

# ==== Compute summary ====
pp_combined_summary <- pp_combined %>%
  filter(Time %in% t) %>%
  group_by(Time, Method) %>%
  summarize(
    FDP_avg   = mean(FDP, na.rm = TRUE),
    Power_avg = mean(Power, na.rm = TRUE),
    FDP_se    = sd(FDP, na.rm = TRUE) / sqrt(n()),
    Power_se  = sd(Power, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# ==== Plotting function ====
plot_metric_combined <- function(df, y_avg, y_se, y_label, y_lim, add_alpha_line = FALSE) {
  p <- ggplot(df, aes(x = Time, y = .data[[y_avg]], group = Method)) +
    geom_point(aes(color = Method, fill = Method, shape = Method), size = 2.5) +
    geom_ribbon(aes(
      ymin = .data[[y_avg]] - .data[[y_se]],
      ymax = .data[[y_avg]] + .data[[y_se]],
      fill = Method
    ), alpha = 0.1, color = NA) +
    geom_line(aes(linetype = Method, color = Method), linewidth = 0.8) +
    scale_color_manual(values = method_colors_combined) +
    scale_fill_manual(values = method_colors_combined) +
    scale_shape_manual(values = shape_values_combined) +
    scale_linetype_manual(values = linetype_values_combined) +
    ylim(y_lim[1], y_lim[2]) +
    xlab("Time") + ylab(y_label) +
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
      legend.position  = "bottom",
      axis.text.x      = element_text(angle = 45, hjust = 1),
      text             = element_text(size = 16, family = "serif")
    )
  
  if (add_alpha_line) {
    p <- p + geom_hline(yintercept = alpha, colour = "black", linetype = "dashed")
  }
  return(p)
}

# ==== Plot ====
p_fdp   <- plot_metric_combined(pp_combined_summary, "FDP_avg", "FDP_se",   "FDR",   c(0, 0.3), add_alpha_line = TRUE)
p_power <- plot_metric_combined(pp_combined_summary, "Power_avg", "Power_se", "Power", c(0, 1),  add_alpha_line = FALSE)

# ==== Combine plots and export ====
plot_combined <- ggarrange(
  p_fdp, p_power,
  ncol = 2, nrow = 1,
  common.legend = TRUE, legend = "bottom",
  font.label = list(size = 16, face = "bold")
)

pdf(file = "plot_combined.pdf", width = 10, height = 4)
plot_combined
dev.off()