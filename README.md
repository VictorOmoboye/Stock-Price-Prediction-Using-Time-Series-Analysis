# DEEP LEARNING BASED STOCK PRICE PREDICTION USING TIME SERIES ANALYSIS
## Leveraging LSTM Neural Networks for Enhanced Financial Forecasting
![image](https://github.com/user-attachments/assets/7ce729fe-2174-4675-835f-bc94ea39871f)
# INTRODUCTION
This project explain the development of a time series model using an LSTM architecture to predict **Tesla Stock** market highs. The model employs historical data with using batch normalization and dropout layers to enhance generalization and mitigate overfitting. The model is trained and validated, using metrics such as RMSE to assess its predictive performance. In addition, the model is applied to a different dataset, BTC-USD, to forecast its market highs. Hence by evaluating its accuracy using metrics like MAE, MSE, and RMSE, the model's versatility and effectiveness across diverse data sets are demonstrated. The process includes data preprocessing, feature engineering, model training, and tuning, followed by extensive validation and performance evaluation. This thorough method confirms the model's capacity to precisely forecast market trends within diverse time series datasets, furnishing valuable insights to inform potential investment choices.

# METHODLOGY
## Data Loading and Cleaning

i.	I started the code with the importation of necessary libraries, and subsequently, loading of the datasetfrom CSV files using pandas (pd.read_csv()), followed by setting the date column as the index and converting it to datetime format (pd.to_datetime()).

ii.	It removes unnecessary columns and checks for null values (isnull().sum()) to ensure data quality.

## Exploratory Data Analysis
i.	The dataset was explored through descriptive statistics (describe()) and visualization using matplotlib (plot()) to understand trends, seasonality, and volatility. The data was also seen to be not normally distributed.
![image](https://github.com/user-attachments/assets/b2d3858e-6968-4638-9f9b-23a0de85392b)

ii.	Augmented Dickey-Fuller (ADF) tests (adfuller) are used to check for stationarity in the time series data.

## Data Preprocessing
i.	The code constructs a training dataset by using a sliding window approach to generate input- output pairs (X_train and y_train) for the model.

ii.	The data is split into training and validation sets, and reshaped for compatibility with the model's input format (np.reshape()).

## Model Creation
i.	I made a LSTM model using the Keras library; this consists multiple LSTM layers, batch normalization, dropout layers, and dense layers for feature extraction and regularization.

ii.	The model is compiled with an Adam optimizer and mean squared error loss function.

## Model Training
i.	The model is trained using training data with early stopping and learning rate reduction callbacks to prevent overfitting and optimize training (model.fit()).  

ii.	The model's training process is monitored through loss and validation loss.

## Predictions and Forecasts
i.	Predictions are made on the validation set (model.predict()) and further tested by forecasting the next 30 days.

ii.	Predictions are plotted alongside actual values using matplotlib to visually assess the model's performance.

## Model Evaluation
i.	The model's accuracy is assessed using metrics such as mean absolute error, mean squared error, and root mean squared error (mean_squared_error() and mean_absolute_error()).

ii.	Comparisons between actual and predicted values are visualized, and errors are calculated for performance assessment.

# RESULT

## Initial Code without Min-Max Scaling

### 1.	Good Results with Tesla Stock Dataset

The initial results with the Tesla stock dataset were promising, suggesting that the model could handle this data effectively even without scaling.
![image](https://github.com/user-attachments/assets/d25a3b04-9722-448d-9bb5-d71871cbd0a6)
This could be attributed to the nature of the Tesla stock data being well-suited to the model's assumptions, distribution, or internal parameters.

### 2.	Poor Results with BTC-USD Validation Dataset

In contrast, the BTC-USD dataset produced a flat graph, indicating poor performance and a potential issue with the model.
![image](https://github.com/user-attachments/assets/5f080871-0e8c-4495-b393-2def7fbf7a95)
A flat graph suggests that the model could not capture the trends or patterns in the BTC-USD data, possibly due to the differences in data distribution or feature ranges.

## Min-Max Scaling and Improved Results
### 1. Scaling Both Datasets

Scaling the data using min-max normalization aligns both datasets within a similar rang [0,1]. This process helps eradicate the differences in data distributions and ensures that features are on the same scale. After scaling, both datasets yielded great results, demonstrating the importance of data preprocessing in model training. Through this, the model could better understand and adapt to the patterns and trends in both datasets.

### Tesla Prediction
![image](https://github.com/user-attachments/assets/a4660210-2b40-4374-90bf-cc91c7fd6292)

![image](https://github.com/user-attachments/assets/785792ef-b593-4523-8e53-c86118c905c8)


### BTC-USD Prediction
![image](https://github.com/user-attachments/assets/cc30b575-a999-4676-8e6e-6e79070fe689)

![image](https://github.com/user-attachments/assets/94ca0f50-7ae5-4db9-8526-b1eab121329f)

## Analysis

The findings highlight the importance of scaling in machine learning models to ensure consistent performance across different datasets. Without proper data scaling, a model that excels on one dataset may struggle on another due to varying feature ranges. Scaling helps avoid problems like overfitting or underfitting by providing uniform data input to the model. Although my initial concerns focused on potential domain differences or overfitting, the enhanced results following data scaling indicate that the core issue was likely related to inadequate data preprocessing. Thus, by normalizing the feature ranges, scaling appears to have mitigated discrepancies between datasets, enabling the model to generalize more effectively.

## Conclusion

This project demonstrated the importance of data scaling in achieving consistent and robust model performance across different datasets. Therefore, the insights gained highlights the necessity of ensuring feature ranges are consistent across datasets, which can lead to better generalization and accuracy. Consequently, this project’s success has potential impacts on financial modelling and other domains where data consistency and model performance are critical. In the context of future enhancement, experimenting with other scaling methods such as standardization or more advanced normalization techniques could offer further performance improvements. Additionally, investigating the model's sensitivity to data preprocessing and exploring techniques like regularization or early stopping could further enhance the model’s robustness and adaptability across diverse datasets.

## THANK YOU
For more information, you can contact me
![WhatsApp Image 2024-09-25 at 6 02 53 PM](https://github.com/user-attachments/assets/f750ef0c-4162-4e8b-bf9a-ef696d4e2b03)
