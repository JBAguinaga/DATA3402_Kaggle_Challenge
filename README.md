![UTA-DataScience-Logo](https://user-images.githubusercontent.com/98781538/226424191-8cadd40f-3610-4ed5-93b1-9e9072098975.png)

# Homesite Quote Conversion

* This repository is my implementation of a Random Forest and Linear Regression algorithm to predict whether or not customers will purchase a quoted insurance plan.

## Overview

* Using an anonymized database of information on customer and sales activity, including property and coverage information, Homesite is challenging you to predict which customers will purchase a given quote. Accurately predicting conversion would help Homesite better understand the impact of proposed pricing changes and maintain an ideal portfolio of customer segments. 
* The approach in this repository is for a regression task, using two common algorithms and comparing the results. Ultimately, the Random Forest model outperformed the Linear Regression model.

## Summary

### Data

  * Type: CSV file, table format
    * Input: Different anonymized fields that correspond to a certain kind of metric (IE. GeographicField1, GeographicField2, PersonalField4, PropertyField5, etc.)
    * Output: Determining whether the quote will "convert" and will sale.
  * Size: 330 MB
  * Instances Train, Test: 260,753 patients for training, 173,836 for testing.

#### Preprocessing / Clean up

  * Pandas was utilized to remove NaN values from some columns.
  * Some columns that had information that was redudant/irrelevant.

#### Data Visualization
  * Although all fields are anonymized, some plots still provide some kind of information on how the values are distributed:
  
<img src="https://user-images.githubusercontent.com/98781538/236433171-e6032775-537f-4ef5-9fba-04c7c40d33a8.png" width="325" height="325"/><img src="https://user-images.githubusercontent.com/98781538/236433818-fe342195-45d5-4fb0-a770-c36cd6c036c3.png" width="325" height="325"/><img src="https://user-images.githubusercontent.com/98781538/236434494-8efdd1ef-5e0e-4014-9735-ce1dc044a80d.png" width="325" height="325"/> 
* Figure 1,2,3: We can see that for *Salesfield2B*, the distribution has a subtle skew to the left whereas with *PropertyField16A* the values are heavily skewed to the right. Some anonymized features were also encoded in a binary format, such as *PersonalField32*, indicating the potential Buyer has or does not have some attribute.

### Problem Formulation

* Define:
  * Input: Anonymized values for different fields pertaining to property and potential buyer.
  * Output: "QuoteConversion_Flag" value of whether or not the customer will purchase the quote.
  * Models:
    * *Linear Regression*: Linear regression estimates the relationship between a dependent variable and an independent variable. It fits linear model to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
    * *Random Forest*: Ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For regression tasks, the mean or average prediction of the individual trees is returned

### Training

* Both models were trained exclusively on Google Colaboratory. The training did not take more than 5 minutes for each model and the results were plotted. 

### Performance Comparison

* The performance of our algorithms were measured by:
*  A) *Mean Squared Error*, B) *Mean Absolute Error* and  C) *R<sup>2</sup>*. 

* The Random Forest and Linear Regression gave us the following values:
       <img src="https://user-images.githubusercontent.com/98781538/236441867-22d16def-5d6e-4608-8b77-a537a43e0df4.png" width="900" height="450"/>
         
  
### Conclusions
* Metrics for our models showed extremely high performance, more than likely made an error somewhere during the preprocessing/encoding, so it's important to be able to distuingish between a correct score and a rational one.
* Despite the flawed setup, Random Forest still scored better than Linear Regression on all metrics.

### Future Work

* In future, be more meticulous with pre-processing and cleaning the data. Messing something up and producing a good output is potentially even worse than the inverse.
* Understand how to compare the different algorithms in more depth. How can I go back and modify the datasets in such a way that one could potentially beat the other or adjust certain parameters when instantiating the models that might have one algorithm perform better than the other. Understanding these differences could help me understand different use cases for real-world scenarios.

## How to reproduce results

* Download dataset
* Since the majority of this attempt was done on Google Colaboraty, it's recommended for future replications that want to follow this attempt.
* Import the necessary Libraries:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn
```
* Begin with cleaning the dataset by figuring out which columns have missing/NaN values and adjust the data accordingly. 
* After cleaning, decide how best to approach the encoding. This approach in particular used One-hot encoding.
* Instantiate the models and run them on the data. Verify that your results make sense and determine how well one did versus the other.

### Overview of files in repository
* HSQ Preproccessing: 
  * Downloads the CSV file and cleans the columns. Also removes redudant values from the dataframe. 
* HSQ Encoding and Visualizaiton: 
  * Visualizes and creates plots of the attributes along with encoding the categorical variables to prepare them for algorithms.
* HSQ Models: 
  * Instantiates the Random Forest and Linear Regression models, trains the models on the cleaned data and produces metrics.

### Software Setup
* Standard Python "Stack":
  * NumPy
  * Pandas
  * Matplotlib
  * Sklearn 
* Google Colaboratory (recommended)
### Data

*  Dataset from Kaggle: https://www.kaggle.com/competitions/homesite-quote-conversion/data
*  Attention to detail is crucial when preprocessing. Might be useful to visualize first before attempting to clean. 

## Citations

  * https://www.analyticsvidhya.com/blog/2021/06/linear-regression-in-machine-learning/
  * https://www.mygreatlearning.com/blog/mean-square-error-explained/
  * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
  * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

