# Email Spam Classification

## Overview
The Email Spam Classification project aims to develop a machine learning model that can effectively classify emails as either spam or non-spam. In this project, there are three different algorithms that are explored for email spam classification: Naïve Bayes, K-Nearest Neighbors (KNN), and Logistic Regression. The goal is to compare the performance of these algorithms and determine which is the most effective.

## Dataset
The dataset used for the project is the "Spambase" dataset, which contains a collection of emails labeled as spam or non-spam (ham). The dataset contains various features representing the characteristics of each email, such as word frequencies, character frequencies, and capital letter usage.

## Structured Approach
1. Data Loading and Preprocessing:
   - The dataset is loaded from a CSV file using the Pandas library.
   - The features and labels are extracted from the dataset.
   - The dataset is split into training and test sets using the train-test split method.

2. Algorithm Implementations:
   - Naïve Bayes: Based on the Bayes' theorem and assumes independence among features. Calculates the prior probabilities and likelihood probabilities of each class and makes predictions based on the highest probability.
   - K-Nearest Neighbors (KNN): Classifies emails based on the majority class among the k nearest neighbors in the feature space. Cosine similarity metric is used to measure similarity between emails.
   - Logistic Regression: Models the probability of an email belonging to a particular class using a linear combination of the features. Uses gradient descent optimization to learn the model parameters.

3. Model Evaluation:
   - Performance of each algorithm is evaluated using 5-fold cross-validation on the training set.
   - Trained models are then evaluated on the test set to assess their generalization ability.
   - Evaluation metrics include accuracy, false positive rate (FPR), true positive rate (TPR), and area under the ROC curve (AUC).

4. Results and Analysis:
   - Performance metrics for each algorithm are collected and analyzed.
   - Accuracy, FPR, TPR, and AUC are used to determine the best performance.

## Algorithm Implementations

### Naïve Bayes Algorithm
- Assumes independence among features and calculates the probability of each class given the feature values.
- Last four attributes are removed from the feature set since they are not related to words.
- Prior probabilities P(c) of each class (spam or non-spam) are calculated based on the frequency of each class in the training data.
- Likelihood probabilities P(a|c) of each attribute given each class are calculated. If a word exists (frequency > 0), its probability is calculated for each class.
- Laplace smoothing is used to handle unseen words and avoid zero probabilities.
- During prediction, the algorithm calculates the product of the prior probability and the likelihood probabilities for each class and selects the class with the highest probability.

### KNN Algorithm
- Classifies a new instance based on the majority class among its k nearest neighbors in the feature space.
- Each email is represented as a feature vector, excluding the last column (class label).
- Cosine similarity between the test instance and each training instance is calculated using the `cosine_similarity` function.
- K nearest neighbors are determined based on calculated similarities.
- Majority class among the k nearest neighbors is assigned as the predicted class for the test instance.
- Parallel processing speeds up the similarity calculations by distributing the computations across multiple cores.

### Logistic Regression Algorithm
- Linear classification algorithm that models the probability of an instance belonging to a particular class.
- Each email is represented as a feature vector, excluding the last column (class label).
- Feature matrix is augmented with a column of ones to account for the bias term.
- Weights are initialized randomly.
- Uses gradient descent optimization to update weights iteratively.
- Sigmoid function is used to calculate the predicted probabilities.
- Cross-entropy loss is calculated, and the gradients are computed based on the difference between the predicted probabilities and the true labels.
- Weights are updated using the calculated gradients and the learning rate.
- Process is repeated for a specified number of epochs.

## Performance Analysis
1. Naïve Bayes (Test Set)
   - Accuracy: 0.89
   - FPR: 0.18
   - TPR: 0.98
   - AUC: 0.90

2. KNN (Test Set)
   - Accuracy: 0.85
   - FPR: 0.18
   - TPR: 0.88
   - AUC: 0.85

3. Logistic Regression (Test Set)
   - Accuracy: 0.51
   - FPR: 0.85
   - TPR: 0.99
   - AUC: 0.57

## Conclusion
Based on the performance analysis, the Naïve Bayes algorithm performs the best among the three algorithms. It achieves the highest accuracy of 0.89 on the test set, indicating that it correctly classifies 89% of the emails. This algorithm also has a relatively low FPR of 0.18, meaning that it misclassifies only 18% of the non-spam emails as spam. The TPR is high at 0.98, indicating that it correctly identifies 98% of the spam emails. The AUC of 0.90 suggests this algorithm has a good overall performance in distinguishing between spam and non-spam emails.

The KNN algorithm performs relatively well, with an accuracy of 0.85 and similar FPR and TPR values to Naïve Bayes. However, its AUC is slightly lower at 0.85.

The Logistic Regression algorithm performs poorly compared to the other two algorithms. The accuracy has a low value of 0.51, the FPR has a high value of 0.85, and the AUC has a low value of 0.57.

In conclusion, the Naïve Bayes algorithm is the best-performing algorithm for the Email Spam Classification project based on the given dataset and evaluation metrics. It achieves a high accuracy, low false positive rate, high true positive rate, and a good AUC score.
