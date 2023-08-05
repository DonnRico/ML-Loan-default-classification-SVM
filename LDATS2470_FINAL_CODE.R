flibrary(na.tools)
library(readr)
library(Hmisc)
library(tigerstats)
library(text)
library(usethis)
library(factoextra)
library(epiDisplay)
library(e1071)
library(forcats)
library(caTools)
library(dplyr)
library(gitcreds)
library(caret)
library(corrplot)
library(lubridate)
library(tibble)
library(latex2exp)
#-------------------------------------------------------------------------------------------------------------#
#                                                                                                             #                   
#                                     DATA PREPROCESSED                                                   #
#                                                                                                             #
#-------------------------------------------------------------------------------------------------------------#

new_data <- read_csv("data_preprocessed.csv")
new_data <- new_data[, -c(1)]


new_data$term <- as.factor(new_data$term)
new_data$emp_length <- as.factor(new_data$emp_length)
new_data$home_ownership <- as.factor(new_data$home_ownership)
new_data$verification_status <- as.factor(new_data$verification_status)
new_data$loan_status <- as.factor(new_data$loan_status)
new_data$addr_state <- as.factor(new_data$addr_state)
new_data$purpose <- as.factor(new_data$purpose)
new_data$pub_rec_bankruptcies <- factor(ifelse(new_data$pub_rec_bankruptcies > 0, 1, 0))
new_data$earliest_cr_line_year <- as.factor(new_data$earliest_cr_line_year)
new_data$earliest_cr_line_month <- as.factor(new_data$earliest_cr_line_month)
new_data$last_pymnt_d_year <- as.factor(new_data$last_pymnt_d_year)
new_data$last_pymnt_d_month <- as.factor(new_data$last_pymnt_d_month)
new_data$last_credit_pull_d_year <- as.factor(new_data$last_credit_pull_d_year)
new_data$last_credit_pull_d_month <- as.factor(new_data$last_credit_pull_d_month)
new_data$issue_d_year <- as.factor(new_data$issue_d_year)
new_data$issue_d_month <- as.factor(new_data$issue_d_month)


####################
# clustering in HD #
####################


data_clustering <- new_data[,colnames(new_data)[grepl('factor|num',sapply(new_data,class))]]


data_clustering <- new_data[,colnames(new_data)[grepl('num',sapply(new_data,class))]]


var_standardization <- function(data,variable) {
  y <- paste("data$",sep = "",variable)
  x <- eval(parse(text=y))
  x <- (x-min(x))/(max(x)-min(x))
  
  name <- paste("Scaled_",sep = "", variable)
  
  data <- data[, ! names(data) %in% variable, drop = F]
  data <- data.frame(data,x)
  names(data)[names(data) == 'x'] <- name
  return(data)
}

for (i in colnames(data_clustering)) {
  data_clustering <- var_standardization(data_clustering, i)
}

loan_status <- new_data$loan_status

data_clustering <- cbind(loan_status, data_clustering)




#-------------------------------------------------------------------------------------------------------------#
#                                                                                                             #                   
#                                     GS + CV LINEAR KERNEL                                                   #
#                                                                                                             #
#-------------------------------------------------------------------------------------------------------------#



folds <- createFolds(data_clustering$loan_status, k = 5)


# Define lambda and alpha values to search over
Cost <- c(1000000000)


# Initialize matrix to store cross-validation results
cv_results_svm <- matrix(NA, nrow = length(Cost), ncol = 8)
colnames(cv_results_svm) <- c("Cost","support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r")
row_idx <- 1
set.seed(87031800)
results_list <- list()

# Loop over lambda and alpha values
for (j in 1:length(Cost)) {
  # Initialize vector to store Dev.Bern results for this combination of lambda and alpha
  support_V <- rep(NA, 5)
  Sensitivity <- rep(NA,5)
  F1_score <- rep(NA, 5)
  Balanced_Accuracy <- rep(NA,5)
  Precision <- rep(NA, 5)
  Recall <- rep(NA, 5)
  Detection_r <- rep(NA, 5)
  
  # Loop over folds
  nfolds <- 0
  for (fold in 1:5) {
    # Split data into training and test sets for this fold
    train_idx <- setdiff(seq_len(nrow(data_clustering)), folds[[fold]])
    test_idx <- folds[[fold]]
    x_train <- data_clustering[train_idx, ]
    y_train <- data_clustering$loan_status[train_idx]
    x_test <- data_clustering[test_idx, ]
    y_test <- data_clustering$loan_status[test_idx]
    print("go")
    print(Cost[j])
    # Fit model with elastic net regularization
    final_model <- svm(loan_status ~ ., data = x_train, kernel = "linear", cost = Cost[j])  
    print("fin")
    # Compute odds
    
    y_pred <- predict(final_model, newdata = x_test)
    confu <- confusionMatrix(data = y_pred, y_test)
    print(confu)
    sensi <- confu$byClass[["Sensitivity"]]
    f1 <- confu$byClass[["F1"]]
    balanced <- confu$byClass[["Balanced Accuracy"]]
    prec <- confu$byClass[["Precision"]]
    rec <- confu$byClass[["Recall"]]
    det <- confu$byClass[["Detection Rate"]]
    
    # Store results for this fold
    support_V[fold] <- final_model$tot.nSV
    print(support_V)
    Sensitivity[fold] <- sensi
    print(Sensitivity)
    F1_score[fold] <- f1
    print(F1_score)
    Balanced_Accuracy[fold] <- balanced
    print(Balanced_Accuracy)
    Precision[fold] <- prec
    print(Precision)
    Recall[fold] <- rec
    print(Recall)
    Detection_r[fold] <- det
    nfolds <- nfolds + 1
  }
  results_list <- cbind(results_list, list(support_V, Sensitivity, F1_score, Balanced_Accuracy, Precision, Recall, Detection_r))
  print(results_list)
  # Compute mean Dev.Bern loss across folds for this combination of lambda and alpha
  mean_support_V <- mean(support_V)
  mean_Sensitivity <- mean(Sensitivity)
  mean_F1_score <- mean(F1_score)
  mean_Balanced_Accuracy <- mean(Balanced_Accuracy)
  mean_Precision <- mean(Precision)
  mean_Recall <- mean(Recall)
  mean_Detection_r <- mean(Detection_r)
  
  # Store results in cv_results matrix
  cv_results_svm[row_idx, 1] <- Cost[j]
  cv_results_svm[row_idx, 2:8] <- c(mean_support_V, mean_Sensitivity, mean_F1_score, mean_Balanced_Accuracy, mean_Precision,mean_Recall, mean_Detection_r)
  print(cv_results_svm)
  # Increment row index
  row_idx <- row_idx + 1
}


to_plot <- as.data.frame(do.call(cbind, results_list))
new_col_names <- c(
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_1", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_2", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_3", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_4", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_5", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_6", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_7", sep = "_"),
  paste(rep(c("Support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r"), each = 1), "_8", sep = "_")
)

# Assign new column names to data frame
colnames(to_plot) <- new_col_names

write.csv(to_plot, file = "to_plot.csv", row.names = FALSE)
#-------------------------------------------------------------------------------------------------------------#
#                                                                                                             #                   
#                                     GS + CV POLYNOMIAL KERNEL                                               #
#                                                                                                             #
#-------------------------------------------------------------------------------------------------------------#


folds <- createFolds(data_clustering$loan_status, k = 5)

# Define lambda and alpha values to search over
Cost <- c(1, 5,7,9,11,13,15,17,19, 21, 25)
Degree <- c(1,2,3)
gamma <- c(0.0001, 0.0005,0.001, 0.005, 0.01, 0.05,0.1,0.15,0.5,0.75,1)
Coefs <- c(0,1)
# Initialize matrix to store cross-validation results
cv_results_svm <- matrix(NA, nrow = length(Cost)*length(Degree)*length(gamma)*length(Coefs), ncol = 14)
colnames(cv_results_svm) <- c("Cost","Degree","Gamma","Coef0", "margin", "support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r", "Lsv", "Rsv")
row_idx <- 1
set.seed(87031800)
results_list <- list()

# Loop over lambda and alpha values
for (i in 1:length(Cost)) {
  for (j in 1:length(Degree)) {
    for (z in 1:length(gamma)){
      for (a in 1:length(Coefs)) {
        # Initialize vector to store Dev.Bern results for this combination of lambda and alpha
        support_V <- rep(NA, 5)
        Sensitivity <- rep(NA,5)
        F1_score <- rep(NA, 5)
        Balanced_Accuracy <- rep(NA,5)
        Precision <- rep(NA, 5)
        Recall <- rep(NA, 5)
        Detection_r <- rep(NA, 5)
        margin <- rep(NA,5)
        Lsv <- rep(NA,5)
        Rsv <- rep(NA, 5)
        
        # Loop over folds
        nfolds <- 0
        for (fold in 1:5) {
          # Split data into training and test sets for this fold
          train_idx <- setdiff(seq_len(nrow(data_clustering)), folds[[fold]])
          test_idx <- folds[[fold]]
          x_train <- data_clustering[train_idx, ]
          y_train <- data_clustering$loan_status[train_idx]
          x_test <- data_clustering[test_idx, ]
          y_test <- data_clustering$loan_status[test_idx]
          print("go")
          print(c(Cost[i],Degree[j], gamma[z], Coefs[a]))
          # Fit model with elastic net regularization
          final_model <- svm(loan_status ~ ., data = x_train, kernel = "polynomial", cost = Cost[i], degree = Degree[j], gamma = gamma[z], coef0 = Coefs[a])  
          print("fin")
          # Compute odds
          
          y_pred <- predict(final_model, newdata = x_test)
          confu <- confusionMatrix(data = y_pred, y_test)
          print(confu)
          sensi <- confu$byClass[["Sensitivity"]]
          f1 <- confu$byClass[["F1"]]
          balanced <- confu$byClass[["Balanced Accuracy"]]
          prec <- confu$byClass[["Precision"]]
          rec <- confu$byClass[["Recall"]]
          det <- confu$byClass[["Detection Rate"]]
          
          # Store results for this fold
          margin[fold] <- 2/norm(final_model$coefs)
          Lsv[fold] <- final_model$nSV[1]
          Rsv[fold] <- final_model$nSV[2]
          support_V[fold] <- final_model$tot.nSV
          print(support_V)
          Sensitivity[fold] <- sensi
          print(Sensitivity)
          F1_score[fold] <- f1
          print(F1_score)
          Balanced_Accuracy[fold] <- balanced
          print(Balanced_Accuracy)
          Precision[fold] <- prec
          print(Precision)
          Recall[fold] <- rec
          print(Recall)
          Detection_r[fold] <- det
          nfolds <- nfolds + 1
        }
        results_list <- cbind(results_list, list(margin, support_V, Sensitivity, F1_score, Balanced_Accuracy, Precision, Recall, Detection_r, Lsv, Rsv))
        print(results_list)
        # Compute mean Dev.Bern loss across folds for this combination of lambda and alpha
        mean_support_V <- mean(support_V)
        mean_Sensitivity <- mean(Sensitivity)
        mean_F1_score <- mean(F1_score)
        mean_Balanced_Accuracy <- mean(Balanced_Accuracy)
        mean_Precision <- mean(Precision)
        mean_Recall <- mean(Recall)
        mean_Detection_r <- mean(Detection_r)
        mean_margin <- mean(margin)
        mean_Lsv <- mean(Lsv)
        mean_Rsv <- mean(Rsv)
        
        # Store results in cv_results matrix
        cv_results_svm[row_idx, 1:4] <- c(Cost[i], Degree[j], gamma[z], Coefs[a])
        cv_results_svm[row_idx, 5:14] <- c(mean_margin, mean_support_V, mean_Sensitivity, mean_F1_score, mean_Balanced_Accuracy, mean_Precision,mean_Recall, mean_Detection_r, mean_Lsv, mean_Rsv)
        print(cv_results_svm)
        # Increment row index
        row_idx <- row_idx + 1
      }
    }
  }
}


#-------------------------------------------------------------------------------------------------------------#
#                                                                                                             #                   
#                                     PCA projection from HD to LD                                            #
#                                                                                                             #
#-------------------------------------------------------------------------------------------------------------#


library(e1071) # required package for SVM
library(plot3D) # required package for data visualization

# Preproc, dividing x = features and y = target for pca in future step
data_PCA <- data_clustering

x <- data_PCA[, -1]
y <- data_PCA$loan_status

# Train an SVM on the transformed data
svm <- svm(loan_status~ ., data = data_PCA, kernel = "linear", cost = 50)

# Predict SVM output for each data point
pred_labels <- predict(svm, data_PCA)
confusionMatrix(pred_labels, data_PCA$loan_status)
# Create a color vector for the data points based on the predicted labels
colors <- rep(3, length(pred_labels)) # initialize colors vector with 3s
for(i in 1:length(pred_labels)) {
  if(pred_labels[i] == "Charged Off" && y[i] == "Charged Off") {
    colors[i] <- 1
  } else if(pred_labels[i] == "Fully Paid" && y[i] == "Fully Paid") {
    colors[i] <- 2
  }
}
to_comp <- table(colors)
to_comp

# PCA
pca <- prcomp(x)
pca.x <- pca$x[,1:3]
library(plot3D)
library(rgl)
library(plotly)
summary(pca)
library(factoextra)

png(filename = "contributions.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
fviz_eig(pca)
dev.off()

fviz_pca_var(pca,col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)
pca

plot3d(pca$rotation[,1:3], 
       texts=rownames(pca$rotation), 
       col="red", cex=0.8)

text3d(pca$rotation[,1:3], 
       texts=rownames(pca$rotation), 
       col="red", cex=0.8)

coords <- NULL
for (i in 1:nrow(pca$rotation)) {
  coords <- rbind(coords, 
                  rbind(c(0,0,0),
                        pca$rotation[i,1:3]))
}

lines3d(coords, 
        col="red", 
        lwd=1)

var <- get_pca_var(pca)
head(var$cos2, 4)
library("corrplot")
corrplot(var$cos2, is.corr=FALSE)
fviz_cos2(pca, choice = "var", axes = 1:3)
## Plot for true positive etc...

pca.fin <- as.data.frame(pca.x)
pca.fin <- cbind(pca.fin, colors)
pca.fin$colors <- factor(pca.fin$colors, levels = c("1", "2", "3"), labels = c("True Negative", "True Positive", "False Positive"))

fig1 <- plot_ly(pca.fin, x = ~PC1, y = ~PC2, z = ~PC3, color = ~colors, colors = c('#BF382A', 'blue', 'green'))
fig1 <- fig1 %>% add_markers()
fig1 <- fig1 %>% layout(scene = list(xaxis = list(title = 'PC1'),
                                     yaxis = list(title = 'PC2'),
                                     zaxis = list(title = 'PC3')),
                        legend = list(font = list(size = 30),
                                      itemsizing = "constant",
                                      itemwidth = 50,
                                      itemheight = 20))

fig1


## Plot for support vectors

supportvectors <- svm$index

pca.fin2 <- as.data.frame(pca.x)
colors <- rep(1, length(pca.fin2))
pca.fin2 <- cbind(pca.fin2, colors)
sv_indices <- match(supportvectors, rownames(pca.fin))

pca.fin2$colors[sv_indices] <- 2
pca.fin2$colors <- factor(pca.fin2$colors, levels = c("1", "2"), labels = c("Non sV", "Support Vector"))

fig2 <- plot_ly(pca.fin2, x = ~PC1, y = ~PC2, z = ~PC3, color = ~colors, colors = c('#BF382A', 'blue'))
fig2 <- fig2%>% add_markers()
fig2 <- fig2 %>% layout(scene = list(xaxis = list(title = 'PC1'),
                                     yaxis = list(title = 'PC2'),
                                     zaxis = list(title = 'PC3')),
                        legend = list(font = list(size = 30),
                                      itemsizing = "constant",
                                      itemwidth = 50,
                                      itemheight = 20))
fig2

#### Plot across folds

to_plot <- read.csv("to_plot.csv") ## dataset of transformed kfolds
cols <- grep("^Balanced", colnames(to_plot)) ## For balanced Accuracy
x <- seq_along(to_plot[,1])
matplot(x, to_plot[,cols], type = "l", xlab = "Index des lignes", ylab = "Valeur") # pas ouf, mieux après avec ggplot

theme_set(theme_bw())
library(tidyr)


cols <- grep("^Balanced", colnames(to_plot)) ## pour ggplot
to_plot_gg <- to_plot[,cols]
to_plot_gg$row_num <- 1:nrow(to_plot_gg)
df_long <- gather(to_plot_gg, key = "variable", value = "value", -row_num) # long format oklm

png(filename = "Balanced_acc.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
ggplot(df_long, aes(x = row_num, y = value, color = variable)) + 
  geom_line() + 
  labs(x = "folds", y = "Balanced Accuracy across folds", color = "Legend title") + theme(legend.position = "top", legend.direction = "horizontal")
dev.off() ## end the process the begin the next plot

cols <- grep("^Detection", colnames(to_plot))
to_plot_gg <- to_plot[,cols]
to_plot_gg$row_num <- 1:nrow(to_plot_gg)
df_long <- gather(to_plot_gg, key = "variable", value = "value", -row_num)


png(filename = "Detection_r.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
ggplot(df_long, aes(x = row_num, y = value, color = variable)) + 
  geom_line() + 
  labs(x = "folds", y = "Detection rate across folds", color = "Legend title") + theme(legend.position = "top", legend.direction = "horizontal")
dev.off()


#-------------------------------------------------------------------------------------------------------------#
#                                                                                                             #                   
#                                    HARD MARGIN LINEAR RBF POLYNO                                            #
#                                                                                                             #
#-------------------------------------------------------------------------------------------------------------#


folds <- createFolds(data_clustering$loan_status, k = 5)

# Define lambda and alpha values to search over
kernel <- c("linear", "radial", "polynomial")
# Initialize matrix to store cross-validation results
cv_results_svm <- matrix(NA, nrow = length(kernel), ncol = 15)

colnames(cv_results_svm) <- c("kernel","Cost","Gamma", "Degree", "Coef0", "margin", "support_V", "Sensitivity", "F1_score", "Balanced_Accuracy", "Precision", "Recall", "Detection_r", "Lsv", "Rsv")
row_idx <- 1
set.seed(87031800)
results_list <- list()

# Loop over lambda and alpha values
for (b in 1:length(kernel)) {
  # Initialize vector to store Dev.Bern results for this combination of lambda and alpha
  support_V <- rep(NA, 5)
  Sensitivity <- rep(NA,5)
  F1_score <- rep(NA, 5)
  Balanced_Accuracy <- rep(NA,5)
  Precision <- rep(NA, 5)
  Recall <- rep(NA, 5)
  Detection_r <- rep(NA, 5)
  margin <- rep(NA,5)
  Lsv <- rep(NA,5)
  Rsv <- rep(NA, 5)
  
  # Loop over folds
  nfolds <- 0
  for (fold in 1:5) {
    # Split data into training and test sets for this fold
    train_idx <- setdiff(seq_len(nrow(data_clustering)), folds[[fold]])
    test_idx <- folds[[fold]]
    x_train <- data_clustering[train_idx, ]
    y_train <- data_clustering$loan_status[train_idx]
    x_test <- data_clustering[test_idx, ]
    y_test <- data_clustering$loan_status[test_idx]
    print("go")
    
    if (kernel[b] == "linear") {
      print(c(100000000,"other parameters null", kernel[b]))
      final_model <- svm(loan_status ~ ., data = x_train, kernel = kernel[b], cost = 100000000)
      gamma_wesh <- 0
      degree_wesh <- 0
      coef_wesh <- 0
      print("fin linear")
    }
    else if (kernel[b] == "radial") {
      print(c(100000000, 0.03020, "other parameters null", kernel[b]))
      final_model <- svm(loan_status ~ ., data = x_train, kernel = kernel[b], cost = 100000000, gamma = 0.03020)  
      gamma_wesh <- 0.03020
      degree_wesh <- 0
      coef_wesh <- 0
      print("fin radial")
    }
    
    else if (kernel[b] == "polynomial") {
      print(c(100000000, 10, 4, 1, "ALL IN", kernel[b]))
      final_model <- svm(loan_status ~ ., data = x_train, kernel = kernel[b], cost = 100000000, degree = 4, gamma = 10, coef0 = 1) 
      gamma_wesh <- 10
      degree_wesh <- 4
      coef_wesh <- 1
      print("fin polynomial")
    } 
    # Compute odds
    
    y_pred <- predict(final_model, newdata = x_test)
    confu <- confusionMatrix(data = y_pred, y_test)
    print(confu)
    sensi <- confu$byClass[["Sensitivity"]]
    f1 <- confu$byClass[["F1"]]
    balanced <- confu$byClass[["Balanced Accuracy"]]
    prec <- confu$byClass[["Precision"]]
    rec <- confu$byClass[["Recall"]]
    det <- confu$byClass[["Detection Rate"]]
    
    # Store results for this fold
    margin[fold] <- 2/norm(final_model$coefs)
    Lsv[fold] <- final_model$nSV[1]
    Rsv[fold] <- final_model$nSV[2]
    support_V[fold] <- final_model$tot.nSV
    print(support_V)
    Sensitivity[fold] <- sensi
    print(Sensitivity)
    F1_score[fold] <- f1
    print(F1_score)
    Balanced_Accuracy[fold] <- balanced
    print(Balanced_Accuracy)
    Precision[fold] <- prec
    print(Precision)
    Recall[fold] <- rec
    print(Recall)
    Detection_r[fold] <- det
    nfolds <- nfolds + 1
  }
  results_list <- cbind(results_list, list(margin, support_V, Sensitivity, F1_score, Balanced_Accuracy, Precision, Recall, Detection_r, Lsv, Rsv))
  print(results_list)
  # Compute mean Dev.Bern loss across folds for this combination of lambda and alpha
  mean_support_V <- mean(support_V)
  mean_Sensitivity <- mean(Sensitivity)
  mean_F1_score <- mean(F1_score)
  mean_Balanced_Accuracy <- mean(Balanced_Accuracy)
  mean_Precision <- mean(Precision)
  mean_Recall <- mean(Recall)
  mean_Detection_r <- mean(Detection_r)
  mean_margin <- mean(margin)
  mean_Lsv <- mean(Lsv)
  mean_Rsv <- mean(Rsv)
  
  # Store results in cv_results matrix
  cv_results_svm[row_idx, 1:5] <- c(kernel[b], 100000000, gamma_wesh, degree_wesh, coef_wesh)
  cv_results_svm[row_idx, 6:15] <- c(mean_margin, mean_support_V, mean_Sensitivity, mean_F1_score, mean_Balanced_Accuracy, mean_Precision,mean_Recall, mean_Detection_r, mean_Lsv, mean_Rsv)
  print(cv_results_svm)
  # Increment row index
  row_idx <- row_idx + 1
}

################################################### RBF

folds <- createFolds(data_clustering$loan_status, k = 5)

# Define lambda and alpha values to search over
cost <- c(1150.13,1656.18,1987.42)
gamma = c(0.005, 0.0302, 0.5)
previous_error  = Inf
previous_acc = 0
previous_detection_r = Inf


decay = 0
conver = FALSE
# Initialize matrix to store cross-validation results
cv_results<- matrix(NA, nrow = 9, ncol = 13)
colnames(cv_results) <- c("Cost","gamma","margin",
                          "support_V", "Sensitivity", "F1_score",
                          "Balanced_Accuracy", "Precision", "Recall", 
                          "Detection_r","L_SV","R_SV",
                          "error")

results_list <- list()
full_results = list()
#convergebce loop
while (conver == FALSE) {
  print(cost)
  print(gamma)
  set.seed(696460)
  
  row_idx <- 1
  # Loop over gamma and alpha values
  for (c in 1:length(cost)) {
    for (gam in 1:length(gamma)){ 
      # Initialize vector to store Dev.Bern results for this combination of lambda and alpha
      support_V <- rep(NA, 5)
      Sensitivity <- rep(NA,5)
      F1_score <- rep(NA, 5)
      Balanced_Accuracy <- rep(NA,5)
      Precision <- rep(NA, 5)
      Recall <- rep(NA, 5)
      Detection_r <- rep(NA, 5)
      
      margin = rep(NA,5)
      L_SV = rep(NA,5)
      R_SV = rep(NA,5)
      error = rep(NA,5)
      
      # Loop over folds
      nfolds <- 0
      for (fold in 1:5) {
        # Split data into training and test sets for this fold
        train_idx <- setdiff(seq_len(nrow(data_clustering)), folds[[fold]])
        test_idx <- folds[[fold]]
        x_train <- data_clustering[train_idx, ]
        y_train <- data_clustering$loan_status[train_idx]
        x_test <- data_clustering[test_idx, ]
        y_test <- data_clustering$loan_status[test_idx]
        
        # Fit model with elastic net regularization
        print(c("go in",fold))
        
        final_model <- svm(loan_status ~ ., data = x_train, kernel = "radial", cost = cost[c], gamma = gamma[gam], 
                           type = "C-classification", loss = "squared-hinge")  
        predictions = predict(final_model, x_test)
        error[fold] = mean(predictions != y_test)
        
        print("out")
        
        # Compute odds
        
        y_pred <- predict(final_model, newdata = x_test)
        confu <- confusionMatrix(data = y_pred, y_test)
        
        sensi <- confu$byClass[["Sensitivity"]]
        f1 <- confu$byClass[["F1"]]
        balanced <- confu$byClass[["Balanced Accuracy"]]
        prec <- confu$byClass[["Precision"]]
        rec <- confu$byClass[["Recall"]]
        det <- confu$byClass[["Detection Rate"]]
        
        
        # Store results for this fold
        support_V[fold] <- final_model$tot.nSV
        Sensitivity[fold] <- sensi
        F1_score[fold] <- f1
        Balanced_Accuracy[fold] <- balanced
        Precision[fold] <- prec
        Recall[fold] <- rec
        Detection_r[fold] <- det
        
        margin[fold] = 2/norm(final_model$coefs)
        L_SV = final_model$nSV[1]
        R_SV = final_model$nSV[2]
        nfolds <- nfolds + 1
      }
      #end fold loop
      print("out of fold")
      
      results_list = cbind(results_list,list(support_V, Sensitivity, F1_score, Balanced_Accuracy, Precision, Recall, Detection_r,margin,error,L_SV,R_SV))
      
      print(results_list)
      # Compute mean Dev.Bern loss across folds for this combination of lambda and alpha
      mean_support_V <- mean(support_V)
      mean_Sensitivity <- mean(Sensitivity)
      mean_F1_score <- mean(F1_score)
      mean_Balanced_Accuracy <- mean(Balanced_Accuracy)
      mean_Precision <- mean(Precision)
      mean_Recall <- mean(Recall)
      mean_Detection_r <- mean(Detection_r)
      
      mean_margin = mean(margin)
      mean_L_SV = mean(L_SV)
      mean_R_SV = mean(R_SV)
      mean_error = mean(error)
      
      
      # Store results in cv_results matrix
      cv_results[row_idx, 1:2] <- c(cost[c],gamma[gam])
      cv_results[row_idx, 3:13] <- c(mean_margin, mean_support_V, mean_Sensitivity,
                                     mean_F1_score, mean_Balanced_Accuracy, mean_Precision,
                                     mean_Recall, mean_Detection_r,mean_L_SV,
                                     mean_R_SV,mean_error)
      print(cv_results)
      # Increment row index
      row_idx <- row_idx + 1
    }
  }
  #######end double if c, gam
  
  print("end double")
  #to keep the previous retults
  full_results = cbind(full_results,cv_results)
  print(full_results)
  
  #finding the optimal parameter based on the smallest detection_rate
  #we find the index and take the minimal difference
  opt_detec = abs(0.1406 - cv_results[,10])
  opt_index = which.min(opt_detec)
  best_detection_r = opt_detec[opt_index]
  
  print(c("opt_detec"))
  print(c("opt_index",opt_index))
  #we assign the best hyperparameters with the opt index
  best_cost = cv_results[opt_index,1]
  best_gamma = cv_results[opt_index,2]
  best_acc = cv_results[opt_index,7]
  
  
  #update the new cost and gamma
  # care implementation of the decay hasn't been tested 
  cost = c(best_cost - (best_cost /5),best_cost, best_cost + (best_cost/5))
  gamma = c(best_gamma- (best_gamma /5),best_gamma, best_gamma + (best_gamma/5))
  
  print(c(cost,gamma))
  
  #test of convergence based on the detection_rate
  if (best_detection_r >= previous_detection_r) {
    print("it converged")
    conver <- TRUE
  }
}


####### graphs

library(tidyr)

##about evolution of the balanced_accuracy and detection rate trhourgh the folds


cols <- grep("^Balanced", colnames(to_plot)) ## pour ggplot
to_plot_gg <- to_plot[,cols]
to_plot_gg$row_num <- 1:nrow(to_plot_gg)
df_long <- gather(to_plot_gg, key = "variable", value = "value", -row_num) # long format oklm

png(filename = "Balanced_acc.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
ggplot(df_long, aes(x = row_num, y = value, color = variable)) + 
  geom_line() + 
  labs(x = "folds", y = "Balanced Accuracy across folds", color = "Legend title") + theme(legend.position = "None", legend.direction = "horizontal")
dev.off() ## end the process the begin the next plot

cols <- grep("^Detection", colnames(to_plot))
to_plot_gg <- to_plot[,cols]
to_plot_gg$row_num <- 1:nrow(to_plot_gg)
df_long <- gather(to_plot_gg, key = "variable", value = "value", -row_num)
df_long

png(filename = "Detection_r.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
ggplot(df_long, aes(x = row_num, y = value, color = variable)) + 
  geom_line() + 
  labs(x = "folds", y = "Detection rate across folds", color = "Legend title") + theme(legend.position = "top", legend.direction = "horizontal")
dev.off()


####--------graph evolution of balanced_acc en fonction de l'augmentation des costs et gamma---

Orta = full_results[,which(colnames(full_results)%in%(c("Cost","gamma", "Balanced_Accuracy")))]
new_col_names_CGB = character()
for (a in 1:24) {
  new_col_names_CGB <- c(new_col_names_CGB,
                         paste(rep(c("Cost","gamma", "Balanced_Accuracy"), each = 1), a, sep = "_"))
}
colnames(Orta) = new_col_names_CGB
Orta <- as.data.frame(Orta)

cols_balance <- grep("^Balanced", colnames(Orta))
balance <- Orta[, cols_balance]
Orta <- Orta[, -cols_balance]

#########
num_cols <- length(Orta) 
new_data <- list()

for (i in seq(1, num_cols-1, by=2)) {
  new_col <- apply(Orta[, i:(i+1)], 1, function(x) paste0(na.omit(x), collapse = '_'))
  new_data[[paste0('new_col_', (i+1)/2)]] <- new_col
}

new_df <- as.data.frame(new_data)

new_df <- cbind(new_df, balance)
#just need new DF
list1 <- c()

for (i in 1:24){
  for (j in 1:nrow(new_df)) {
    list1 <- c(list1, new_df[j,i])
  }
}


list1 = as.vector(list1)
list1
list2 <- c()

for (i in 25:48){
  for (j in 1:nrow(new_df)) {
    list2 <- c(list2, new_df[j,i])
  }
}
list2= as.vector(list2)

plotdf <- as.data.frame(cbind(list1, list2))
plotdf$list1 = unlist(plotdf$list1)
plotdf$list2 = unlist(plotdf$list2)
#check the dup
no_dupes = !duplicated(plotdf$list1)
no_dupes
plotdf_nop =(plotdf[no_dupes,])

plotdf_nop$list2 = as.numeric(plotdf_nop$list2)
#plotdf_nop$list1 = as.factor(plotdf_nop$list1)

randy = strsplit(plotdf_nop$list1, "_")
randy = data.frame(do.call(rbind,randy))

plotdf_nop$list1 = unlist(randy[1])
plotdf_nop$list3 = unlist(randy[2])


plotdf_nop$index = c(1:nrow(plotdf_nop))
colnames(plotdf_nop) = c("cost","balanced_accuracy","gamma","index")
plotdf_nop$cost = as.numeric(plotdf_nop$cost)
plotdf_nop$gamma = as.numeric(plotdf_nop$gamma)

#actually the graph
tenta = plotdf_nop

png(filename = "pixel_graph.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files


jho2 <- ggplot(tenta, aes(x = index, y = balanced_accuracy, fill = gamma)) +
  geom_bar(stat = "identity") + theme(legend.position = "top" )+
  labs(y = "Balanced_Accurarcy") +
  coord_cartesian(ylim = c(0.78, NA)) +
  
  # Move the color aesthetic mapping inside aes()
  geom_line(aes(y = cost, color = "Cost"), size = 0.75) +
  
  scale_y_continuous(name = "Balanced_Accuracy",
                     sec.axis = sec_axis(trans = ~., name = "Cost", labels = c(0, 500, 1000, 1500, 1750, 2000))) + 
  
  # Add a scale_color_manual layer to specify the color and label for the legend
  scale_color_manual(name = "", values = c("Cost" = "red"), labels = c("Cost"))

jho2
dev.off()

################################## PCA PLOT

library(e1071) # required package for SVM
library(plot3D) # required package for data visualization

# Preproc, dividing x = features and y = target for pca in future step
data_PCA <- data_clustering

x <- data_PCA[, -1]
y <- data_PCA$loan_status

# Train an SVM on the transformed data
svm <- svm(loan_status~ ., data = data_PCA, kernel = "radial", cost = 1987.42, gamma = 0.0302)

# Predict SVM output for each data point
pred_labels <- predict(svm, data_PCA)
confusionMatrix(pred_labels, data_PCA$loan_status)
# Create a color vector for the data points based on the predicted labels
colors <- rep(3, length(pred_labels)) # initialize colors vector with 3s
for(i in 1:length(pred_labels)) {
  if(pred_labels[i] == "Charged Off" && y[i] == "Charged Off") {
    colors[i] <- 1
  } else if(pred_labels[i] == "Fully Paid" && y[i] == "Fully Paid") {
    colors[i] <- 2
  } else if (pred_labels[i]=="Charged Off" && y[i] == "Fully Paid" ) {
    
    colors[i] = 4
  }
}
to_comp <- table(colors)
to_comp

# PCA
pca <- prcomp(x)
pca.x <- pca$x[,1:3]
library(plot3D)
library(rgl)
library(plotly)
summary(pca)
library(factoextra)

png(filename = "contributions66.png", width = 10, height = 6, units = 'in', res=900) ## save the plot in files
fviz_eig(pca)
dev.off()

fviz_pca_var(pca,col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)
pca

plot3d(pca$rotation[,1:3], 
       texts=rownames(pca$rotation), 
       col="red", cex=0.8)

text3d(pca$rotation[,1:3], 
       texts=rownames(pca$rotation), 
       col="red", cex=0.8)

coords <- NULL
for (i in 1:nrow(pca$rotation)) {
  coords <- rbind(coords, 
                  rbind(c(0,0,0),
                        pca$rotation[i,1:3]))
}

lines3d(coords, 
        col="red", 
        lwd=1)

dev.off()
var <- get_pca_var(pca)
ind = get_pca_ind(pca)

mean(ind$cos2[,1]) # 0.3439
mean(ind$cos2[,2]) # 0.1778
mean(ind$cos2[,3]) #0.18293
mean(ind$cos2)
ind$coord
var$cos2


library("corrplot")
corrplot(var$cos2, is.corr=FALSE)
fviz_cos2(pca, choice = "var", axes = 1:3)
## Plot for true positive etc...

pca.fin <- as.data.frame(pca.x)
pca.fin <- cbind(pca.fin, colors)
pca.fin$colors <- factor(pca.fin$colors, levels = c("1", "2", "3","4"), labels = c("True Negative", "True Positive", "False Positive","False Negative"))

fig1 <- plot_ly(pca.fin, x = ~PC1, y = ~PC2, z = ~PC3, color = ~colors, colors = c('#BF382A', 'blue', 'green',"yellow"))
fig1 <- fig1 %>% add_markers()
fig1 <- fig1 %>% layout(scene = list(xaxis = list(title = 'PC1'),
                                     yaxis = list(title = 'PC2'),
                                     zaxis = list(title = 'PC3')),
                        legend = list(font = list(size = 30),
                                      itemsizing = "constant",
                                      itemwidth = 50,
                                      itemheight = 20))

fig1


## Plot for support vectors

supportvectors <- svm$index

pca.fin2 <- as.data.frame(pca.x)
colors <- rep(1, length(pca.fin2))
pca.fin2 <- cbind(pca.fin2, colors)
sv_indices <- match(supportvectors, rownames(pca.fin))

pca.fin2$colors[sv_indices] <- 2
pca.fin2$colors <- factor(pca.fin2$colors, levels = c("1", "2"), labels = c("Non sV", "Support Vector"))

fig2 <- plot_ly(pca.fin2, x = ~PC1, y = ~PC2, z = ~PC3, color = ~colors, colors = c('#BF382A', 'blue'))
fig2 <- fig2%>% add_markers()
fig2 <- fig2 %>% layout(scene = list(xaxis = list(title = 'PC1'),
                                     yaxis = list(title = 'PC2'),
                                     zaxis = list(title = 'PC3')),
                        legend = list(font = list(size = 30),
                                      itemsizing = "constant",
                                      itemwidth = 50,
                                      itemheight = 20))
fig2
