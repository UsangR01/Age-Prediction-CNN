# Age-Prediction-CNN
Age Prediction Model

Project Title: Age and Gender Classification with Convolutional Neural Networks
Table of Contents

1.    Introduction
        Project Overview
        Objective
        Dataset Description

2.    Data Preprocessing
        Data Loading
        Data Augmentation
        Data Splitting

3.    Model Architecture
        Model 1: CNN for Age Classification
        Model 2: CNN for Gender Classification
        Model 3: Combined Age and Gender Classification

4.    Training
        Model Training Setup
        Training Procedure
        Hyperparameter Tuning

5.    Evaluation
        Model Evaluation Metrics
        Results and Performance

6.    Conclusion
        Summary
        Challenges Faced
        Future Improvements

8.    References
        External Libraries and Frameworks

**1. INTRODUCTION**

Project Overview
----------------
This project aims to develop Convolutional Neural Network (CNN) models for age classification using facial image data. The project leverages deep learning techniques to predict the age of individuals based on input image data.

Objective
----------
The primary objectives of this project are:

    Age Classification: Predicting the age group of individuals (e.g., ages between: 1-2, 3-9, 10-20, 21-27, 28-35, 36-65 and >=66).

Dataset Description
-------------------
The project utilizes a dataset of facial pixel data, including age, gender and ethnicity labels for each image. The dataset consists of grayscale images, and each image is preprocessed to be of size 48x48 pixels.

**2. DATA PREPROCESSING**

Data Loading
------------
The dataset is loaded into the project using custom data loading functions. It includes image data and corresponding labels for age.

Data Augmentation
------------------
Data augmentation techniques are applied to increase the diversity of the training dataset. Augmentation includes random rotations, shifts, flips, and zooms to improve model generalization.

Mapping Age to Class
--------------------
To facilitate age classification, we map age labels to age classes. This mapping function categorizes ages into predefined classes (mentioned earlier in the objectives section). The function is applied to the age labels in the dataset.

Data Splitting
--------------
The dataset is split into training, validation, and test sets to facilitate model training and evaluation.

**3. MODEL ARCHITECTURE**

CNN architecture for age classification:
----------------------------------------
    Data Augmentation:
        To enhance the training dataset and improve the model's robustness, data augmentation techniques are applied. These include random rotations, width and height shifts, shearing, zooming, and horizontal flipping.
        
Convolutional Neural Network (CNN):
        The core of the model is a Convolutional Neural Network (CNN), a powerful architecture for image classification tasks.

Input Layer: 
        The network accepts grayscale images with dimensions of image_height x image_width x num_channels, where num_channels is set to 1 for grayscale images.

Convolutional Layers: 
        Three convolutional layers are employed, each followed by a Rectified Linear Unit (ReLU) activation function. These layers detect features at different levels of abstraction.
        The first convolutional layer consists of 32 filters with a kernel size of (3, 3).
        The second convolutional layer contains 64 filters with a (3, 3) kernel.
        The third convolutional layer uses 128 filters with a (3, 3) kernel.

Pooling Layers: 
        After each convolutional layer, a max-pooling layer is applied to downsample the feature maps. Max-pooling reduces the spatial dimensions of the features.

Batch Normalization: 
        Batch normalization is utilized after each convolutional layer. This technique helps stabilize training by normalizing the activations.

Flatten Layer: 
        Before transitioning to fully connected layers, the feature maps are flattened into a one-dimensional vector.

Dense Layers: 
        Two dense (fully connected) layers are incorporated for classification purposes. These layers have 256 units each and utilize the ReLU activation function. A dropout layer with a rate of 0.5 is applied after the first dense layer to prevent overfitting.

Output Layer: 
        The final layer consists of num_classes units, where num_classes is the number of age classes to be predicted. It employs the softmax activation function to produce class probabilities.
        
**4. TRAINING**

Model Training Setup
--------------------
    Model compilation with loss functions and optimizers and Learning rate scheduling.

    The model is compiled with the Adam optimizer, which adapts the learning rate during training. 
    A learning rate scheduler is employed, which reduces the learning rate exponentially to fine-tune the training process.

Training Procedure
-------------------
    The model is trained for a specified number of epochs (50 in this case) using the training data augmented with the techniques mentioned earlier. 
    The batch size is set to 32. During training, both training and validation data are used to monitor the model's performance and prevent overfitting.
    Early stopping and model checkpointing for model saving.


**5. EVALUATION**

Model Evaluation Metrics
-------------------------
        The trained model is evaluated on a separate test dataset to assess its performance. The evaluation metrics include loss and accuracy. The final test accuracy is reported as a percentage.

Results and Performance
-----------------------

Here's a personalized analysis of the output from our most improved model:

**Confusion Matrix:**
        Looking at the confusion matrix, I can see how my model is performing for each class. It's a handy table that helps me understand where my model is making correct predictions and where it's stumbling. 
        The rows represent the actual classes, while the columns show what my model predicted.

        For Class 0 (ages 1-2), it's doing quite well with a high number of true positives (215 out of 236) and very few false predictions.
        Class 1 (ages 3-9) shows decent performance but with some room for improvement. It correctly predicts 139 samples, but there are also 40 false-positive predictions.
        Class 2 (ages 10-20) is a bit tricky for my model, with 90 false predictions out of 281 samples.
        Class 3 (ages 21-27) has a significant number of samples (819), and my model correctly predicts about 60% of them.
        Class 4 (ages 28-35), the largest class with 1147 samples, has an okay performance with a 61% recall rate.
        Class 5 (ages 36-65) and Class 6 (ages 66 and above) also have room for improvement with precision and recall scores.

**Classification Report:**
        The classification report provides more detailed metrics for each class and an overall assessment.

        Precision tells me how accurate my positive predictions are. For Class 0, it's quite high at 0.87, but for Class 2, it's lower at 0.54.
        Recall (or sensitivity) indicates my model's ability to find all relevant instances. It's around 0.60 for Class 3, showing that my model is catching a good portion of those samples.
        The F1-score balances precision and recall. The macro average F1-score is 0.65, indicating moderate overall performance.
        The weighted average F1-score is 0.62, reflecting that class imbalances might be affecting my model's performance.

**Analysis:**
    I can see that my model performs well for Class 0 and reasonably well for Class 1. However, it faces challenges with Classes 2, 3, 4, 5, and 6.
    The overall performance is okay, but there's room for improvement, especially for those classes where precision and recall are lower.

**6. CONCLUSION**

**Summary:**
        In this project, our primary objective was to develop and fine-tune a Convolutional Neural Network (CNN) model for the classification of facial attributes, including age, gender, and ethnicity, using grayscale images. This project aimed to leverage deep learning techniques to achieve accurate and reliable predictions.

**Achievements and Outcomes**
        Throughout the course of this project, we have accomplished several key milestones and outcomes:
        
        Data Preprocessing: 
                We started by carefully preprocessing our dataset, which involved handling missing values, normalizing pixel values, and reshaping images to ensure they are suitable for training.
        
        Data Augmentation: 
                To enhance the robustness of our model and mitigate overfitting, we implemented data augmentation techniques. 
                This involved creating multiple variations of our images through processes like rotation, shifting, zooming, and flipping.

        Model Architecture: 
                We designed, implemented, and evaluated multiple CNN architectures. 
                These models were tailored to our specific problem, incorporating layers for convolution, pooling, dropout, and batch normalization to optimize learning.

        Training and Evaluation: 
                Extensive training was performed on the models using the augmented dataset. 
                We monitored key performance metrics during training, including accuracy, loss, and validation scores, to evaluate our model's progress.

        Hyperparameter Tuning: 
                We experimented with various hyperparameters, such as learning rates, batch sizes, and network architectures, to fine-tune our models and improve their predictive capabilities.

        Mapping Age to Class: 
                As a preprocessing step, we mapped the continuous age labels to discrete age classes, simplifying the multi-class classification problem.

        Performance Analysis: 
                We conducted a comprehensive analysis of our model's performance using evaluation metrics such as the confusion matrix, precision, recall, and F1-score. 
                This helped us gain insights into which classes our model excels at and where improvements are needed.

**Challenges Faced:**

        1. Resource Constraints:
                Limited computational resources, such as GPU availability and memory, posed constraints on the training and experimentation process. 
                This sometimes resulted in longer training times and limited the scope of hyperparameter tuning.

        2. Data Imbalance:
                One of the primary challenges was dealing with a significant class imbalance in the dataset. Some classes had a much larger number of samples than others. 
                This imbalance can lead to biased model predictions, where the model may perform well on the majority class but poorly on the minority classes.
                Again, I faced limitations in resources to handle the amount of data generated from oversampling. Often my sessions crashed both on my local system, colab and aws instance.

        3. Model Complexity:
                Designing an optimal neural network architecture posed a challenge. 
                Determining the number of layers, the type of layers (e.g., convolutional, recurrent), and the size of the layers required careful consideration. 
                Overly complex models could lead to overfitting, while overly simple models might not capture the dataset's complexity.

**Future Improvements:**
        While this project has yielded valuable insights and promising results, there are several avenues for future improvement and expansion:

                1. Fine-Tuning and Optimization: 
                        We will continue to fine-tune our models and explore advanced optimization techniques to enhance overall accuracy and convergence speed.

                2. Class Imbalance: 
                        Addressing class imbalances is crucial. 
                        We will experiment with techniques like oversampling, undersampling, or synthetic data generation to improve predictions for underrepresented classes.


**7. References**

External Libraries and Frameworks:

        This project relies on several external libraries and frameworks to facilitate various tasks, including data preprocessing, model building, and evaluation. Here are the key tools used:

                TensorFlow: 
                        TensorFlow is the core framework for developing and training deep learning models. 
                        It provides a versatile environment for building neural networks and offers various modules for data manipulation, image preprocessing, and model evaluation.

                Keras: 
                        Keras, integrated into TensorFlow, simplifies the process of building and training neural networks. 
                        It offers a user-friendly API for defining and configuring deep learning models.

                NumPy: 
                        NumPy is a fundamental library for numerical computing in Python. 
                        It is used for efficient data handling and manipulation, particularly when working with multi-dimensional arrays and matrices.

                Pandas: 
                        Pandas is an essential library for data analysis and manipulation. It's particularly useful for handling tabular data structures like DataFrames.

                Matplotlib and Seaborn: 
                        These visualization libraries enable the creation of informative plots and graphs for data exploration, model performance visualization, and presentation purposes.

                Scikit-Learn: 
                        Scikit-Learn provides a comprehensive suite of tools for machine learning and statistical modeling. 
                        It is used for tasks such as data preprocessing, model selection, and performance evaluation.

                Augmentor: 
                        Augmentor is a Python library used for image data augmentation. 
                        It enhances the dataset by applying various transformations to the images, leading to better model generalization.

                Imgaug: 
                        Imgaug is another image augmentation library that offers a wide range of augmentation techniques, including geometric and color-based transformations.

These external libraries and frameworks collectively streamline the development process, enhance model capabilities, and contribute to the overall success of the age classification project.
