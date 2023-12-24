```python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
import gym

# Define constants
RANDOM_STATE = 42  # For reproducibility

# Define your dataset
# For the purpose of this example, we will use random data
# In a real-world scenario, you would load your dataset here
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(2, size=1000)  # Binary target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Supervised Learning
# Logistic Regression
lr = LogisticRegression(random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions)}')

# Random Forest
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}')

# Support Vector Machine
svc = SVC(random_state=RANDOM_STATE)
svc.fit(X_train, y_train)
svc_predictions = svc.predict(X_test)
print(f'Support Vector Machine Accuracy: {accuracy_score(y_test, svc_predictions)}')

# Unsupervised Learning
# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
kmeans.fit(X_train)
kmeans_predictions = kmeans.predict(X_test)
print(f'KMeans Silhouette Score: {silhouette_score(X_test, kmeans_predictions)}')

# Reinforcement Learning
# For the purpose of this example, we will use the CartPole-v1 environment from OpenAI Gym
# In a real-world scenario, you would define your environment and agent here
env = gym.make('CartPole-v1')
for episode in range(5):  # Run 5 episodes
    observation = env.reset()
    for t in range(100):  # Limit each episode to 100 timesteps
        env.render()
        action = env.action_space.sample()  # Take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
env.close()
```
