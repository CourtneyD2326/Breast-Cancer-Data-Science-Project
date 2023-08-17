# Identification and Analysis of Key Predictive Features for Breast Cancer Outcomes: A Data-Driven Approach to Personalized Patient Care

![pink-ribbon-for-breast-cancer-awareness](https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/ede2c68c-31f9-4475-bbe6-04a15dcbd6cb)

Breast cancer remains one of the most diagnosed and fatal cancers globally, presenting a complex and multifaceted challenge for clinicians and researchers alike. The successful diagnosis, prognosis, and treatment of this disease necessitate a nuanced understanding of a diverse array of factors that may contribute to its onset and progression. Leveraging the comprehensive SEER dataset for breast cancer, this project seeks to elucidate the most critical features that influence a patient's outcome, specifically their status as alive or dead post-diagnosis.

The analysis will encompass both numerical attributes (including Age at diagnosis, Tumor Size, Regional Node Examined, Regional Node Positive, and Survival Months) and categorical attributes (including Race, Marital Status, T Stage, N Stage, 6th Stage, Grade, A Stage, Estrogen Status, and Progesterone Status). By identifying and quantifying the relative importance of these features, the project aims to contribute to a more targeted and personalized approach to breast cancer care.

Through robust statistical methods and machine learning techniques, the objective is to create a predictive model that not only enhances our understanding of the disease's underlying mechanisms but also facilitates more informed clinical decision-making. This analysis will provide healthcare practitioners with insights into the most influential determinants of patient outcomes, thereby enabling more effective intervention strategies, tailored treatment plans, and improved patient counseling. Ultimately, the project aspires to translate these data-driven insights into tangible benefits for patient care, contributing to the broader efforts to combat and manage breast cancer.

## EDA : Exploratory Data Analysis

### The Dataset
The dataset contains information concerning breast cancer patients from the SEER Program of the NCI (2017). The data involved female patients with infiltrating duct and lobular carcinoma breast cancer (SEER primary cites recode NOS histology codes 8522/3) diagnosed in 2006-2010. The dataset author excluded samples whose survival months were less than one month or had unknown values regarding the tumor data (size, examined regional LNs, positive regional LNs). In total, there are 4024 samples, with 85% pertaining to the Alive class. It was downloaded from Kaggle. 

### The dataset includes the following independent features:
Numerical Attribute:
 *	Age: Age of patient at diagnosis
 * Tumor Size: Size of tumour at diagnosis
 * Regional Node Examined
 * Reginol Node Positive
 * Survival Months
   
Categorical Attribute:
 * Race: Race of patient (white, other (American Indian/ AK Native, Asian/ Pacific Islander)
 * Marital Status: Married (including common law), Single (never married), other)
 * T Stage: Tumor Size (T2, T1, other)
 * N Stage: Number of lymph nodes involved (N1, N2, other)
 * 6th Stage: Overall stage of cancer (IIA, IIB, other)
 * Grade: Grade of cancer (Moderately differentiated; Grade II, Poorly differentiated; Grade III, other)
 * A Stage: Distant metastasis status (Regional, Distant)
 * Estrogen Status
 * Progesterone Status
   
Target Variable:
 * Status: Alive or Dead

### Analysis of the Target Variable : 'Status'

<img width="489" alt="Screenshot 2023-08-17 at 20 35 11" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/42cffdb6-8e0e-42b0-b543-ab3db5f5d5aa">

The problem is classified as a classification task, as the target variable is discrete. There are more alive patients than dead patients (imbalanced dataset). 3408 patients are alive, while 616 patients are dead.

### Anaylsing the Independent Attributes
### Age
The age range of the patients spans from 30 to 69 years. The 25th percentile of the patient age is 47, meaning 25% are younger than this age and 75% are older. The median age is 54, indicating that half of the patients are younger and half are older than this age. At the 75th percentile, the age is 61, with 75% of the patients being younger and 25% being older than this age.

<img width="1033" alt="Screenshot 2023-08-17 at 20 39 24" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/cec5b885-374b-4f2c-bd0e-5592c7f6f4ec">

### Tumor Size
The tumor sizes range from 1 cm to 140 cm. At the 25th percentile, 25% of the tumors are smaller than 16 cm, and 75% are larger. The median size is 25 cm, meaning that half of the tumors are smaller and half are larger than this size. The 75th percentile is at 38 cm, with 75% of the tumors being smaller and 25% being larger than this size. The distribution is skewed to the right.
<img width="1030" alt="Screenshot 2023-08-17 at 20 40 43" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/efd9ef3b-be60-4c88-bb26-8a636b70e0b9">

### Regional Node Examined

<img width="1042" alt="Screenshot 2023-08-17 at 20 43 35" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/69e1447d-f483-488f-9506-1f38cc610093">

## Bivariate Analysis
The heat map suggests correlation greater than 0 between regional node positive and regional node examined (0.41) and tumor size and regional node positive (0.24).

<img width="711" alt="Screenshot 2023-08-17 at 20 45 06" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/04ec3eec-92d8-4954-8d9d-6e054c37adb6">

## Data Preparation
### Encode Ordinal Categorical Attributes and All Other Categorical Attributes
First, we need to encode the tumor grade. The grade of a tumor reflects the aggressiveness of its cells; a higher grade indicates a more aggressive form. This suggests a natural hierarchy among tumor grades, classifying it as an ordinal problem. The grades have an intrinsic order, with Grade I being less than Grade II, and Grade II being less than Grade III. Therefore, we can encode these grades numerically, replacing 'Grade 1' with 1, 'Grade 2' with 2, and so on.

<img width="798" alt="Screenshot 2023-08-17 at 20 47 53" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/510836fb-389b-46ba-a638-2231b39bb624">


## Rescaling Features
Many machine learning algorithms operate under the assumption that numerical features share a consistent scale. The scikit-learn library in Python provides two common techniques for normalization:

MinMaxScaler: Adjusts a column's values to fit within the range [0,1]. StandardScaler: Modifies a numerical column so that it possesses a mean of 0 and a standard deviation of 1. However, for our dataset, considering the presence of outliers in columns like 'Tumor Size', 'Regional Node Examined', 'Regional Node Positive', and 'Survival Months', we'll employ a third approach named RobustScaler. This method is more resilient to outliers. It achieves normalization by subtracting the median from each value in a column and subsequently dividing by the interquartile range.

<img width="742" alt="Screenshot 2023-08-17 at 20 51 27" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/0eda1798-4a44-4702-888e-c37cdbbb1e49">


## Treatmeant of Outliers

<img width="987" alt="Screenshot 2023-08-17 at 20 54 29" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/066c78c3-56f8-43f1-8c41-40beffa0bc1e">


The histograms show a leftward skew, which is suboptimal. Numerous machine learning algorithms perform optimally when the features are not skewed in either direction. Before we proceed with the treatment of the outliers, let's first identify the minimum values in these columns.

<img width="984" alt="Screenshot 2023-08-17 at 20 56 38" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/8e058f1f-1213-49c7-8c83-fdb8d80b3a14">


Rather than disregarding the data rows containing extreme values, we can modify them to lessen the adverse effect of outliers on machine learning models. One common method to achieve this is by employing a log transformation. However, since all the columns contain negative values after rescaling, we must first add a constant to all the values, shifting them into a positive range, before applying the log transformation. In this case, we have opted to shift the values into the interval [1, +∞], laying the groundwork for the subsequent application of a log transformation.

<img width="1011" alt="Screenshot 2023-08-17 at 20 57 52" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/df73de3d-6085-4f28-9228-191ef240e408">


The histograms show that the transformed features are less skewed and have distributions closer to normal than the original features.


# Feature Importance and Machine Learning Models

## Logisitc Regression

### Feature Importance

<img width="898" alt="Screenshot 2023-08-17 at 21 02 12" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/1cf409eb-066b-45ca-9ae2-d87096fa92e4">

From our Feature Importance of Logistic Regession, confusion matrix and GridSearch Model, we get 682 observations that were classified as alive, 657 were classified correctly, and of the 123 observations that were classified as dead, 52 were correctly classified.

## DecisionTree Classifier

### Feature Importance 

<img width="901" alt="Screenshot 2023-08-17 at 21 05 32" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/c19a1c16-8b8f-4f81-a9db-b7331077ea95">

It looks like that Survival Months was the main driver in this classification problem.
DecisionTree Classifier has much lower accuracy on test data than on training data.

## SVC Classifier

Support Vector Classifier has almost the same accuracy on test data and on training data. From the analysis of the confusion matrix, it's evident that out of 682 observations identified as alive, 673 were accurately classified. Similarly, among the 123 observations categorized as dead, 32 were classified correctly. 

## Random Forest Classifier

### Feature Importance

<img width="883" alt="Screenshot 2023-08-17 at 21 10 34" src="https://github.com/CourtneyD2326/Breast-Cancer-Data-Science-Project/assets/85933265/2f17ce3c-78be-45b5-a9df-7c7a0fc66fd8">

With the above Feature Importance barplot shows Survival months, Age, Tumor size, and the number of Regional nodes examined are the most important features means the following:

Survival Months: The number of months a patient has survived may have a strong relationship with their current status (alive or dead). This could reflect the overall severity or progression of the disease.

Age: Age may be a significant factor, possibly reflecting the general health and resilience of the patient, or perhaps indicating that the disease behaves differently in people of different ages.

Tumor Size: The size of the tumor is often directly correlated with the stage and aggressiveness of the cancer. Larger tumors might indicate a more advanced stage of cancer, influencing the prognosis.

Regional Nodes Examined: The number of regional lymph nodes examined could relate to how far the cancer has spread. Lymph nodes are often examined to determine whether cancer has begun to spread to other parts of the body. The presence of cancer in lymph nodes often indicates a more advanced stage of the disease.

## In Summary
These features provide key insights into the characteristics that are most predictive of a patient's status in the context of breast cancer. They may help clinicians understand which factors are most crucial in determining prognosis and could potentially guide treatment decisions.




















