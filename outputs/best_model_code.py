import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
file_path = '/Users/tekla/homework2_llm/outputs/engineered_data.csv'
data = pd.read_csv(file_path)

# Split data into features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier with specified hyperparameters
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print metrics
print(f'METRICS: accuracy={accuracy:.2f} f1={f1:.2f} recall={recall:.2f}')