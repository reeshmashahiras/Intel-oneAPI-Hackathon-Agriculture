# Intel-oneAPI-Hackathon-Agriculture
# PROBLEM STATEMENT
Farmers face challenges in making informed decisions about crop selection and disease management. Traditional methods are time-consuming, subjective, and lead to suboptimal outcomes. Limited access to modern technology and expertise exacerbates these issues, leaving farmers vulnerable to economic losses and reduced productivity.

“Our solution aims to revolutionize this process by crop recommendation and the plant disease identification using Intel oneAPI toolkits”.
# SOLUTION
Our innovative web application leverages Intel's optimized toolkits and libraries, along with machine learning algorithms, to analyze soil data and recommend crops based on crucial factors such as nitrogen and phosphorus levels. For crop recommendation, we employ the Random Forest algorithm, utilizing its robustness and ability to handle complex datasets effectively.  We utilize K-Means clustering for data segmentation and a Convolutional Neural Network (CNN) model trained on image data to accurately identify specific plant diseases from uploaded images.Additionally, our platform seamlessly integrates Streamlit for a user-friendly interface and incorporates image recognition technology for plant disease prediction. By combining these technologies, our application provides farmers with actionable insights for crop selection and disease management.
# HOW IT WORKS
Data Collection:
     CROP RECOMMENDATION - Soil data, including nitrogen, phosphorus levels, and other relevant factors, is collected from farms using manual input.
     PLANT DISEASE DETECTION - Users can also upload images of diseased plants to the system for disease diagnosis.

Data Preprocessing:
The collected soil data is preprocessed to handle missing values, and normalize the features.
Images of diseased plants are preprocessed to enhance image quality.

Crop Recommendation:
When users provide manual input for nitrogen, phosphorus levels, and other pertinent factors, the system employs advanced data analytics to generate meticulously curated recommendations for optimal crop selection tailored to the specific agricultural context.

Disease Diagnosis:
Upon user plant image upload, the system employs advanced image recognition to accurately predict plant diseases, facilitating timely and informed disease management decisions.

# FEATURES:
1. A user-friendly interface accessible to farmers for inputting relevant data.
2. Personalized crop recommendations and disease predictions displayed based on the provided inputs.
3. Seamless integration with Intel oneAPI tools for efficient backend processing and real-time updates.
   
![Screenshot 2024-04-06 103600](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/5dc9f486-1101-4d47-a6e0-c8cb4a71ac46)

![Screenshot 2024-04-06 103704](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/813cac99-61ac-41c9-a4ea-4a35bcc49512)

![Screenshot 2024-04-06 104015](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/424404d4-bcce-406a-ae92-afeba82048ca)

![Screenshot 2024-04-06 103439](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/329cfb6d-730a-4879-8da0-32d74d6b636f)

# TECHNOLOGY STACK :
MACHINE LEARNING ALGORITHMS :

Crop Recommendation:
     1.KMeans (for clustering)
     2.StandardScaler (for feature scaling) 
Plant Disease Detection : 
LogisticRegression,KNeighborsClassifier,RandomForestClassifier,DecisionTreeClassifier,SVC,LinearSVC,NuSVC,XGBClassifier,LinearDiscriminantAnalysis,GaussianNB,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,CatBoostClassifier.

LIBRARIES :

   1.numpy: For numerical computing.
   2.pandas: For data manipulation and analysis.
   3.matplotlib: For data visualization.
   4.seaborn: For statistical data visualization.
   5.plotly: For interactive plots.
   6.missingno: For visualizing missing data patterns.
   7.modin: For parallelizing Pandas operations.


# INTEL INTEGRATIONS :
1. Intel Developer Cloud
2. Intel Distribution or python
3. Intel scikit-learn

![Screenshot 2024-04-06 011504](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/a8574741-6987-4999-a056-9269ca3e26c9)


# ADVANTAGES OF MIGRATING TO ONEAPI:
1. Availability of high computing services.
2. Good developer support.

![2](https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/dee77ddf-6c29-41cf-a6df-2d96dda8320a)

# FINAL OUTPUT OF EASY FARMING WEBAPP



https://github.com/reeshmashahiras/Intel-oneAPI-Hackathon-Agriculture/assets/100523261/5c807933-a4d8-440d-b24a-7ed83bcb46df










