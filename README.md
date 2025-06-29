
# Life Expectancy Analysis 🌍

![image](https://github.com/user-attachments/assets/30144a4d-a0ed-413c-a3fe-744ea125f5e0)

![image](https://github.com/user-attachments/assets/cd8f7e94-aaa4-4fc4-8acf-abe47277490a)


The chart shows a relatively even distribution of life expectancy across all age groups in Papua New Guinea. This indicates that the proportion of the population reaching old age is roughly the same as the proportion living to younger ages. The average life expectancy in Papua New Guinea is 57 years.

Question :
Significant Differences:
Significant difference in life expectancy between Papua New Guinea and Japan. What are the main factors contributing to this striking difference?

Age Distribution:
In Japan, the largest percentage of the population falls within the 61-80 age group. Why is the proportion of elderly people in Japan much larger than in Papua New Guinea?

Challenges in Papua New Guinea:
This chart indicates that the life expectancy distribution in Papua New Guinea is relatively even across all age groups. What challenges does Papua New Guinea face in increasing the life expectancy of its population?


## Project Overview 🚀

What for :
The Life Expectancy Analysis project investigates the factors associated with life expectancy in East Asia and the Pacific region.
This project intends to use various data science techniques to identify key drivers of life expectancy and forecast future trends based on multiple factors.

## Problem Statement 📑

How does Life expectancy vary significantly across different countries?
How to understanding the factors that contribute to this variation is crucial for improving public health policies and fostering better healthcare systems?

---

## Objective 📚

* To explore the dataset to understand the variables and their relationships.
* To Identify potential factors that may influence life expectancy.
* Determine the statistical significance of these relationships.
* Develop predictive models to estimate life expectancy based on the identified factors.

Intended Audience or Users:
* Researchers: Academics and researchers in the fields of public health, epidemiology, and demography. Data analitics
* Healthcare Professionals: Doctors, nurses, and public health experts.
* General Public: Individuals interested in global health and well-being.

Hypotheses:

1. Economic Factors: Countries with higher GDP per capita tend to have higher life expectancies.
2. Healthcare Access: Countries with better access to healthcare services, including immunization and maternal health care, have higher life expectancies.
3. Education: Higher levels of education are associated with increased life expectancy.
4. Environmental Factors: Exposure to pollution and other environmental hazards can negatively impact life expectancy.
5. Social Factors: Factors like gender equality, social security, and reduced poverty can positively influence life expectancy.

---

## Solution 💡

The key solution to this project was building a predictive model that can estimate life expectancy based on multiple factors such as income, healthcare, education, and environmental conditions.

### Key Insights from the Solution:
- **Economic Development**: Countries with higher GDP per capita tend to have higher life expectancy, though this relationship can vary in lower-income countries.
- **Healthcare Access**: There is a strong correlation between life expectancy and access to healthcare services. Countries with a higher number of healthcare providers (doctors per capita) and better health infrastructure generally exhibit higher life expectancy.
- **Environmental Impact**: Poor air quality and lack of access to clean water are associated with lower life expectancy, particularly in developing countries.
- **Education and Lifestyle**: Higher education levels and healthier lifestyle choices (e.g., lower smoking rates, better diet) correlate with longer life expectancy.

The models developed as part of this analysis can also be used to predict life expectancy based on existing or projected data, offering a forecasting tool for future trends in public health.

---

## Results & Output 📊

The analysis and modeling produced several valuable insights and outcomes:

1. **Key Predictive Factors**: The model identified the most important predictors of life expectancy, such as healthcare access, GDP per capita, and air quality.
2. **Predictive Model Performance**:
   - **Ridge Regression**: Achieved an R² score of 0.75, indicating a strong fit for predicting life expectancy.
   - **Random Forest Regression**: Achieved an R² score of 0.85, providing better accuracy by capturing non-linear relationships.
3. **Global Trends**: Through visualizations, the analysis revealed global life expectancy trends and the disparity between developed and developing countries.
4. **Visualization Tools**: Interactive charts and dashboards were created to present the findings, enabling users to explore trends by region, year, and country.

These insights are crucial for healthcare organizations, governments, and policymakers to make data-informed decisions to improve the quality of life and longevity of populations.

---


## Methodology & Strategy 🔍

To conduct a comprehensive life expectancy analysis, the following methodology was employed:

### 1. **Data Collection & Preprocessing**
   - **Source**: The dataset used for this analysis is sourced from (https://www.kaggle.com/datasets/kiranshahi/life-expectancy-dataset).
   - **ERD Diagram**: Provide a clear visual representation of how data is organized and connected.

![image](https://github.com/user-attachments/assets/4aae9f69-d374-4a8d-9cf3-e45f9ac4c713)


   - **Cleaning**: The data was cleaned to remove missing values, handle outliers, and ensure consistency across all features. Categorical data was encoded appropriately, and numerical features were scaled for modeling.

![image](https://github.com/user-attachments/assets/7036d0ff-185d-4de5-b00c-11c0e5a2085e)


### 2. **Exploratory Data Analysis (EDA)**

   - The first step involved exploring the dataset visually and statistically to identify trends, patterns, and potential correlations between life expectancy and various variable.
   - Key visualizations such as heatmaps, scatter plots, and correlation matrices were used to uncover the relationships between features.



### 3. **Feature Engineering**
   - Derived new features based on domain knowledge (e.g., average healthcare spending per capita, urbanization rates).
   - Applied transformation techniques such as log transformation on skewed data (e.g., GDP per capita) to improve model performance.

### 4. **Modeling & Predictive Analysis**
   - Several machine learning models were trained to predict life expectancy:
     - **Linear Regression**: To model the relationship between life expectancy and predictor variables.
     - **Random Forest Regression**: To capture non-linear relationships and feature interactions.
     - **Gradient Boosting (XGBoost)**: To improve predictive accuracy and handle complex relationships in the data.
   - Hyperparameter tuning was performed using grid search to optimize the models.

---

## Technologies Used ⚙️

- **Python** 🐍
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn`, `plotly` for data visualization
  - `scikit-learn`, `xgboost` for machine learning
- **Google Colabolator Notebooks** for interactive analysis
- **Tableau** for creating advanced visualizations (optional)

---

## Contributing 🤝

We welcome contributions! If you'd like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push your changes (`git push origin feature-name`).
5. Submit a pull request.

Please make sure to write clear, descriptive commit messages and follow the existing coding conventions.

---

## Contributing 🤝

Feel free to contribute by opening issues or submitting pull requests. Contributions are welcome for improving the analysis, adding new features, or cleaning up the code.

---

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments 🙏

- Special thanks to the contributors of the datasets used in this project.
- Thanks to the open-source community for providing essential libraries.
```
