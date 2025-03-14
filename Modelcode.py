
library(shiny)
library(ggplot2)
library(tidyverse)
library(caret)
library(randomForest)
library(glmnet)

# Preprocessed dataset
df <- read.csv('/mnt/data/preprocessed_football_data.csv')

# Trainining & Testig
set.seed(42)
trainIndex <- createDataPartition(df$goal_contribution, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Model 1: Random Forest
rf_model <- randomForest(goal_contribution ~ ., data = trainData, ntree=100, importance=TRUE)
y_pred_rf <- predict(rf_model, testData)
rf_r2 <- R2(y_pred_rf, testData$goal_contribution)
rf_rmse <- RMSE(y_pred_rf, testData$goal_contribution)

# Model 2: Ridge Regression
ridge_model <- train(goal_contribution ~ ., data = trainData, method='ridge', trControl=trainControl(method='cv', number=10))
y_pred_ridge <- predict(ridge_model, testData)
ridge_r2 <- R2(y_pred_ridge, testData$goal_contribution)
ridge_rmse <- RMSE(y_pred_ridge, testData$goal_contribution)

# Save Models
saveRDS(rf_model, 'random_forest_model.rds')
saveRDS(ridge_model, 'ridge_regression_model.rds')

# Shiny UI
ui <- fluidPage(
  titlePanel("Football Player Performance Prediction"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model", "Choose Model:", choices = c("Random Forest", "Ridge Regression"))
    ),
    mainPanel(
      plotOutput("predictionPlot"),
      verbatimTextOutput("modelStats")
    )
  )
)

# Shiny Server
server <- function(input, output) {
  output$predictionPlot <- renderPlot({
    if (input$model == "Random Forest") {
      ggplot(data.frame(testData$goal_contribution, y_pred_rf), aes(x = testData$goal_contribution, y = y_pred_rf)) +
        geom_point() +
        geom_smooth(method = "lm", col = "red") +
        labs(title = "Random Forest Predictions", x = "Actual", y = "Predicted")
    } else {
      ggplot(data.frame(testData$goal_contribution, y_pred_ridge), aes(x = testData$goal_contribution, y = y_pred_ridge)) +
        geom_point() +
        geom_smooth(method = "lm", col = "blue") +
        labs(title = "Ridge Regression Predictions", x = "Actual", y = "Predicted")
    }
  })

  output$modelStats <- renderPrint({
    if (input$model == "Random Forest") {
      paste("R2 Score:", rf_r2, "\nRMSE:", rf_rmse)
    } else {
      paste("R2 Score:", ridge_r2, "\nRMSE:", ridge_rmse)
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)
