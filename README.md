# TITLE: DEEP LEARNING BASED STOCK PRICE PREDICTION USING TIME SERIES ANALYSIS
## SUBTITLE: Leveraging LSTM Neural Networks for Enhanced Financial Forecasting
This project provides an overview of utilizing deep learning techniques, specifically LSTM neural networks, for stock price prediction through time series analysis. It outlines the significance, proposed solution, challenges, and references, setting the stage for further exploration and implementation in the financial domain.

![image](https://github.com/user-attachments/assets/7ce729fe-2174-4675-835f-bc94ea39871f)
# OVERVIEW

This project emails the development of a time series model using an LSTM architecture to predict Tesla stock market highs. The model employs historical data with using batch normalization and dropout layers to enhance generalization and mitigate overfitting. The model is trained and validated, using metrics such as RMSE to assess its predictive performance. In addition, the model is applied to a different dataset, BTC-USD, to forecast its market highs. Hence by evaluating its accuracy using metrics like MAE, MSE, and RMSE, the model's versatility and effectiveness across diverse data sets are demonstrated. The process includes data preprocessing, feature engineering, model training, and tuning, followed by extensive validation and performance evaluation. This thorough method confirms the model's capacity to precisely forecast market trends within diverse time series datasets, furnishing valuable insights to inform potential investment choices.

# TECHNICAL DETAILS
In recent times, Long Short-Term Memory (LSTM) networks have become a integral approach for handling sequential data such as stock prices, given their unique ability to capture complex temporal dependencies (Chung and Shin, 2018). LSTM networks are a type of recurrent neural network (RNN) known for their capacity to learn long-term dependencies in sequential data (Lipton et al., 2015). Since its introduction by Hochreiter and Schmidhuber in 1997, LSTMs have since been applied to numerous domains, including natural language processing, speech recognition, and time series forecasting (Van Houdt et al., 2020). Basically, what distinguishes LSTMs from other RNNs is their memory cell architecture, which allows them to selectively retain or forget information based on input gates, output gates, and forget gates (Abdullrhman et al., 2021). Consequently, this capability makes LSTMs well-suited for tasks like stock price prediction, where long-range dependencies are common (as used in this study).

Several studies have demonstrated the efficacy of LSTM models in financial forecasting. A study by Ali et al. (2021) showed that LSTMs could outperform traditional financial models such as logistic regression and support vector machines when forecasting stock price movements. They emphasized the significance of data preprocessing and feature engineering in enhancing the efficacy of the model. In a study by Bhandari et al (2020), LSTM models were employed for predicting daily closing prices of major stock indices. Their results indicated that LSTMs performed better than conventional models such as autoregressive integrated moving average (ARIMA) and support vector regression (SVR). They also explored the combination of LSTM with wavelet transform to enhance the model's ability to capture non-linear patterns in the data.
 
Further, an essential aspect of time series forecasting is data preprocessing, in practice, data must often be detrended and normalized to improve model performance and stabilize learning. In light of this, Wibawa et al. (2024) compared different data normalization techniques in LSTM models and found that min-max scaling often yields the best results for time series forecasting. In addition to LSTM models, other deep learning architectures have also been explored in financial forecasting. For instance, convolutional neural networks (CNNs) have been used in combination with LSTMs to enhance feature extraction from time series data (Karim et al., 2017). However, CNNs excel at capturing local patterns and trends, while LSTMs handle long-term dependencies (Wang et al., 2023). Consequently, the hybrid CNN-LSTM approach has been found to improve the accuracy of forecasts in some cases.

Furthermore, studies have investigated the impact of ensemble methods on financial forecasting. Thus, by combining the outputs of multiple models, ensemble approaches can increase robustness and stability. Studies by Wang et al. (2018) and others have shown that ensemble methods can improve forecast accuracy by aggregating the strengths of individual models. While deep learning models like LSTMs have shown great promise in time series forecasting, challenges like the interpretability of these models are imminent (Su et al., 2024). Since deep neural networks function as black boxes, understanding how the model arrives at a specific prediction can be difficult.
However, data scientists are exploring techniques such as attention mechanisms and model- agnostic methods to address this concern. In addition, time series data often exhibit characteristics like seasonality, trends, and outliers, which require careful handling during preprocessing and modelling (Alexandropoulos et al., 2019). Consequently, advanced techniques such as seasonal decomposition of time series data and rolling window analysis are utilized to account for these factors.

Finally, it is essential to acknowledge the importance of evaluating model performance comprehensively. Frequently employed measures encompass mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE). These metrics offer understanding into the accuracy, precision, and bias of the model.

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

## References
Alexandropoulos, S.A.N., Kotsiantis, S.B. and Vrahatis, M.N., (2019). Data preprocessing in predictive data mining. The Knowledge Engineering Review, 34, p.e1.

Ali, M., Khan, D.M., Aamir, M., Ali, A. and Ahmad, Z., (2021). Predicting the direction movement of financial time series using artificial neural network and support vector machine. Complexity, 2021, pp.1-13.

Bhandari, H.N., Rimal, B., Pokhrel, N.R., Rimal, R., Dahal, K.R. and Khatri, R.K., (2022). Predicting stock market index using LSTM. Machine Learning with Applications, 9, p.100320.

Chung, H. and Shin, K.S., (2018). Genetic algorithm-optimized long short-term memory network for stock market prediction. Sustainability, 10(10), p.3765.

Karim, F., Majumdar, S., Darabi, H. and Chen, S., (2017). LSTM fully convolutional networks for time series classification. IEEE access, 6, pp.1662-1669.

Lipton, Z.C., Kale, D.C., Elkan, C. and Wetzel, R., (2015). Learning to diagnose with LSTM recurrent neural networks. arXiv preprint arXiv:1511.03677.

Su, J., Jiang, C., Jin, X., Qiao, Y., Xiao, T., Ma, H., Wei, R., Jing, Z., Xu, J. and Lin, J., (2024). Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review. arXiv preprint arXiv:2402.10350.

Van Houdt, G., Mosquera, C. and Nápoles, G., (2020). A review on the long short-term memory model. Artificial Intelligence Review, 53(8), pp.5929-5955.

Wang, X., Liu, H., Du, J., Dong, X. and Yang, Z., (2023). A long-term multivariate time series forecasting network combining series decomposition and convolutional neural networks. Applied Soft Computing, 139, p.110214.

Wang, Y., Chen, Q., Sun, M., Kang, C. and Xia, Q., 2018. An ensemble forecasting method for the aggregated load with subprofiles. IEEE Transactions on Smart Grid, 9(4), pp.3906-3908.
