########## Illustration figure: repeated experiments ##########

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
library(scales)
library(dplyr)
library(tidyr)
library(patchwork)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("functions_OnSel.R")
source("algoclass_OnSel.R")
source("SAFFRON_feedback functions.R")

generate_data_Gaussian <- function(T, pi_1, mu_c_values) {
  mu_t <- numeric(T)
  p_t <- numeric(T)
  Z_t <- numeric(T)
  theta <- numeric(T)
  
  for (t in 1:T) {
    if (runif(1) <= pi_1) {
      mu_t[t] <- sample(mu_c_values, 1)
      theta[t] <- 1
    } else {
      mu_t[t] <- 0
      theta[t] <- 0
    }
    
    Z_t[t] <- rnorm(1, mu_t[t], 1)
    p_t[t] <- pnorm(-Z_t[t])
  }
  
  return(data.frame(Z = Z_t, p = p_t, theta = theta))
}

N <- 1000
T <- 1000
alpha <- 0.1
lambda <- 0.5

########## Repeated experiments (single pi_1) ##########

pi_1 <- 0.4
num_experiments <- 100
mu_c_values <- 2

alpha_values <- data.frame()

for (i in 1:num_experiments) {
  data <- generate_data_Gaussian(N, pi_1, mu_c_values)
  data <- data[sample(nrow(data)), ]
  
  rej_SAFFRON <- SAFFRON(data$p, alpha)
  rej_LORD <- LORD(data$p, alpha)
  rej_LOND <- LOND(data$p, alpha)
  rej_feedback_SA <- SAFFRON_feedback(data$p, alpha = alpha, w0 = alpha/2, theta = data$theta)
  rej_feedback_LORD <- Lord_feedback(data$p, alpha, data$theta, W0 = alpha/2)
  
  alpha_experiment <- data.frame(
    SAFFRONF = rej_feedback_SA$alphai,
    SAFFRON = rej_SAFFRON$alphai,
    LORDF = rej_feedback_LORD$alphai,
    LORD = rej_LORD$alphai,
    LOND = rej_LOND$alphai,
    Time = 1:T
  )
  
  alpha_values <- rbind(alpha_values, alpha_experiment)
  print(i)
}

alpha_mean <- alpha_values %>%
  group_by(Time) %>%
  summarise(
    SAFFRONF = mean(SAFFRONF, na.rm = TRUE),
    SAFFRON = mean(SAFFRON, na.rm = TRUE),
    LORD = mean(LORD, na.rm = TRUE),
    LORDF = mean(LORDF, na.rm = TRUE),
    LOND = mean(LOND, na.rm = TRUE)
  )

t <- seq(100, 1000, 10)

alpha_subset <- alpha_mean %>%
  filter(Time %in% t)

alpha_long <- pivot_longer(alpha_subset,
                           cols = -Time,
                           names_to = "Method",
                           values_to = "Alpha")

alpha_long$Method[which(alpha_long$Method == "LORD")] <- "LORD++"
alpha_long$Method[which(alpha_long$Method == "SAFFRONF")] <- "SF"
alpha_long$Method[which(alpha_long$Method == "LORDF")] <- "LF"

alpha_long$Method <- factor(alpha_long$Method,
                            levels = c("SF", "LF", "SAFFRON", "LORD++", "LOND"))

#write.csv(alpha_long, "alpha_long.csv")

p_Gaussian <- ggplot(alpha_long, aes(x = Time, y = Alpha, color = Method, shape = Method)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2, alpha = 0.6) +
  scale_color_nejm(palette = c("default"), alpha = 0.8) +
  scale_y_continuous(
    labels = label_number(scale = 1e3, accuracy = 1),
    limits = c(0, NA)
  ) +
  labs(
    x = "Time",
    y = expression(alpha[t]~"(×10"^{-2}*")")
  ) +
  theme_minimal(base_family = "serif") +
  theme(
    axis.text = element_text(size = 16),
    axis.title = element_text(size = 20),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16),
    panel.grid.major = element_line(colour = NA),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    text = element_text(size = 16, family = "serif"),
    legend.position = "top",
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(5, "pt"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8)
  )

p_Gaussian

pdf(file = "alpha_Gaussian.pdf", width = 10, height = 4)
p_Gaussian
dev.off()

########## Repeated experiments across multiple pi_1 values ##########

pi_1_values <- c(0.2, 0.4, 0.6, 0.8)
num_experiments <- 500
mu_c_values <- 2
t_seq <- seq(100, 1000, 50)

run_experiments <- function(pi_1) {
  alpha_values <- data.frame()
  
  for (i in 1:num_experiments) {
    data <- generate_data_Gaussian(N, pi_1, mu_c_values)
    data <- data[sample(nrow(data)), ]
    
    rej_SAFFRON <- SAFFRON(data$p, alpha)
    rej_LORD <- LORD(data$p, alpha)
    rej_LOND <- LOND(data$p, alpha)
    rej_feedback_SA <- SAFFRON_feedback(data$p, alpha = alpha, w0 = alpha/2, theta = data$theta)
    rej_feedback_LORD <- Lord_feedback(data$p, alpha, data$theta, W0 = alpha/2)
    
    alpha_experiment <- data.frame(
      SAFFRONF = rej_feedback_SA$alphai,
      SAFFRON = rej_SAFFRON$alphai,
      LORDF = rej_feedback_LORD$alphai,
      LORD = rej_LORD$alphai,
      LOND = rej_LOND$alphai,
      Time = 1:T
    )
    
    alpha_values <- rbind(alpha_values, alpha_experiment)
    cat(sprintf("pi_1 = %.1f | experiment %d/%d\n", pi_1, i, num_experiments))
  }
  
  alpha_mean <- alpha_values %>%
    group_by(Time) %>%
    summarise(
      SAFFRONF = mean(SAFFRONF, na.rm = TRUE),
      SAFFRON = mean(SAFFRON, na.rm = TRUE),
      LORD = mean(LORD, na.rm = TRUE),
      LORDF = mean(LORDF, na.rm = TRUE),
      LOND = mean(LOND, na.rm = TRUE)
    )
  
  alpha_long <- alpha_mean %>%
    filter(Time %in% t_seq) %>%
    pivot_longer(cols = -Time, names_to = "Method", values_to = "Alpha")
  
  alpha_long$Method[alpha_long$Method == "LORD"] <- "LORD++"
  alpha_long$Method[alpha_long$Method == "SAFFRONF"] <- "SF"
  alpha_long$Method[alpha_long$Method == "LORDF"] <- "LF"
  
  alpha_long$Method <- factor(
    alpha_long$Method,
    levels = c("SF", "LF", "SAFFRON", "LORD++", "LOND")
  )
  
  alpha_long$pi_1 <- pi_1
  alpha_long
}

all_results <- lapply(pi_1_values, run_experiments)
all_data <- bind_rows(all_results)

#write.csv(all_data, "alpha_long_all_pi1.csv", row.names = FALSE)

make_plot <- function(df, pi_1_val, show_legend = FALSE) {
  p <- ggplot(df, aes(x = Time, y = Alpha, color = Method, shape = Method)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2, alpha = 0.6) +
    scale_color_nejm(palette = "default", alpha = 0.8) +
    scale_y_continuous(
      labels = label_number(scale = 1e3, accuracy = 0.01),
      limits = c(0, NA)
    ) +
    labs(
      title = bquote(pi[1] == .(pi_1_val)),
      x = "Time",
      y = expression(alpha[t]~"(×10"^{-3}*")")
    ) +
    theme_minimal(base_family = "serif") +
    theme(
      plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
      axis.text = element_text(size = 13),
      axis.title = element_text(size = 15),
      legend.text = element_text(size = 13),
      legend.title = element_text(size = 13),
      panel.grid.major = element_line(colour = NA),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      text = element_text(size = 13, family = "serif"),
      legend.position = "bottom",
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black"),
      axis.ticks.length = unit(5, "pt"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8)
    )
  p
}

plots <- lapply(seq_along(pi_1_values), function(i) {
  df <- all_data %>% filter(pi_1 == pi_1_values[i])
  make_plot(df, pi_1_values[i], show_legend = (i == 1))
})

combined <- (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

combined

pdf("alpha_all_pi1.pdf", width = 12, height = 4)
print(combined)
dev.off()