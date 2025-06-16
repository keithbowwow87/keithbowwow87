
# === Setup ===
library(MASS)
library(spdep)
library(sf)
library(spData)
library(caret)
library(dplyr)
library(keras)
library(ggplot2)

# === Load Data ===
data("Boston")
X <- Boston[, -14]
y <- Boston$medv

# === Load geometry and spatial weights ===
boston_506 <- st_read(system.file("shapes/boston_tracts.shp", package = "spData")[1], quiet = TRUE)
coords <- cbind(boston_506$LON, boston_506$LAT)

nb_q <- poly2nb(boston_506)
lw_q <- nb2listw(nb_q, style = "W")

# === Train/Test Split ===
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# === Train Teacher Model ===
teacher_model <- lagsarlm(medv ~ ., data = Boston, listw = lw_q)
teacher_predictions <- predict(teacher_model)
resid_teacher_test <- y_test - teacher_predictions[-train_indices]

# === Rebuild spatial weights for test set ===
coords_test <- coords[-train_indices, ]
nb_test <- knn2nb(knearneigh(coords_test, k = 5))
lw_test <- nb2listw(nb_test, style = "W")

# === Moran's I for Teacher Residuals ===
moran_teacher <- moran.test(resid_teacher_test, lw_test)
print("Moran’s I (Teacher Residuals):")
print(moran_teacher)

# === Student Model Setup ===
X_train_mat <- as.matrix(X_train)
X_test_mat <- as.matrix(X_test)

scale_model <- preProcess(X_train_mat, method = c("center", "scale"))
X_train_scaled <- as.matrix(predict(scale_model, X_train_mat))
X_test_scaled <- as.matrix(predict(scale_model, X_test_mat))

# Correct Keras Model Definition
build_student_model <- function(input_dim) {
  model <- keras_model_sequential()
  model$add(layer_dense(units = 64, activation = 'relu', input_shape = list(input_dim)))
  model$add(layer_dropout(rate = 0.2))
  model$add(layer_dense(units = 1))

  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(learning_rate = 0.001)
  )

  return(model)
}

student_model <- build_student_model(ncol(X_train_scaled))

student_model %>% fit(
  x = X_train_scaled,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  verbose = 0
)

# === Student Predictions and Residuals ===
student_preds <- student_model %>% predict(x = X_test_scaled)
resid_student_test <- y_test - as.vector(student_preds)

# === Moran's I for Student Residuals ===
moran_student <- moran.test(resid_student_test, lw_test)
print("Moran’s I (Student Residuals):")
print(moran_student)

# === Visualize Residuals Spatially ===
boston_test_sf <- boston_506[-train_indices, ]
boston_test_sf$resid_teacher <- resid_teacher_test
boston_test_sf$resid_student <- resid_student_test

ggplot(boston_test_sf) +
  geom_sf(aes(fill = resid_teacher)) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  labs(title = "Teacher Residuals (Test Set)")

ggplot(boston_test_sf) +
  geom_sf(aes(fill = resid_student)) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
  labs(title = "Student Residuals (Test Set)")
