import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import math
from collections import Counter

def criar_dataset_sintetico():
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Glucose': np.random.normal(120, 30, n_samples),
        'BMI': np.random.normal(32, 6, n_samples),
        'Age': np.random.exponential(30, n_samples) + 20,
        'BloodPressure': np.random.normal(72, 12, n_samples),
        'Pregnancies': np.random.poisson(3, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['Glucose'] = np.clip(df['Glucose'], 44, 200)
    df['BMI'] = np.clip(df['BMI'], 18, 67)
    df['Age'] = np.clip(df['Age'], 21, 81).astype(int)
    df['BloodPressure'] = np.clip(df['BloodPressure'], 24, 122)
    df['Pregnancies'] = np.clip(df['Pregnancies'], 0, 17)
    
    diabetes_prob = (
        0.15 * (df['Glucose'] > 126) +
        0.10 * (df['BMI'] > 30) +
        0.08 * (df['Age'] > 45) +
        0.05 * (df['BloodPressure'] > 90) +
        0.03 * (df['Pregnancies'] > 4)
    )
    
    df['Outcome'] = (diabetes_prob > 0.2).astype(int)
    return df

class ID3:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None
        
    def entropy(self, y):
        if len(y) == 0: return 0
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * math.log2(count/total) for count in counts.values() if count > 0)
    
    def information_gain(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
            
        n = len(y)
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask)/n) * left_entropy + (np.sum(right_mask)/n) * right_entropy
        return parent_entropy - weighted_entropy
    
    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for threshold in [np.percentile(values, p) for p in [25, 50, 75]]:
                gain = self.information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 10:
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self.build_tree(X[~left_mask], y[~left_mask], depth+1)
        }
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        return self.predict_one(x, node['left'] if x[node['feature']] <= node['threshold'] else node['right'])
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        return np.array([self.predict_one(x, self.tree) for x in X])

class C45:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None
        
    def entropy(self, y):
        if len(y) == 0: return 0
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * math.log2(count/total) for count in counts.values() if count > 0)
    
    def gain_ratio(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
            
        n = len(y)
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        
        weighted_entropy = (np.sum(left_mask)/n) * left_entropy + (np.sum(right_mask)/n) * right_entropy
        information_gain = parent_entropy - weighted_entropy
        
        p_left = np.sum(left_mask) / n
        p_right = np.sum(right_mask) / n
        split_info = 0
        if p_left > 0: split_info -= p_left * math.log2(p_left)
        if p_right > 0: split_info -= p_right * math.log2(p_right)
        
        return information_gain / split_info if split_info != 0 else 0
    
    def best_split(self, X, y):
        best_ratio = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for threshold in [np.percentile(values, p) for p in [25, 50, 75]]:
                ratio = self.gain_ratio(X, y, feature, threshold)
                if ratio > best_ratio:
                    best_ratio, best_feature, best_threshold = ratio, feature, threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 10:
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self.build_tree(X[~left_mask], y[~left_mask], depth+1)
        }
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        return self.predict_one(x, node['left'] if x[node['feature']] <= node['threshold'] else node['right'])
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        return np.array([self.predict_one(x, self.tree) for x in X])

def print_tree(tree, feature_names, depth=0, condition=""):
    if not isinstance(tree, dict):
        print("  " * depth + f"└─ PRED: {'Diabético' if tree == 1 else 'Não Diabético'}")
        return
    
    feature_name = feature_names[tree['feature']]
    threshold = tree['threshold']
    
    if depth == 0:
        print(f"ROOT")
    else:
        print("  " * (depth-1) + f"├─ {condition}")
    
    print_tree(tree['left'], feature_names, depth+1, f"{feature_name} ≤ {threshold:.1f}")
    print_tree(tree['right'], feature_names, depth+1, f"{feature_name} > {threshold:.1f}")

def extract_rules(tree, feature_names):
    rules = []
    
    def get_paths(node, path=[]):
        if not isinstance(node, dict):
          
            if path:
                rule = "SE " + " E ".join(path) + f" ENTÃO {'Diabético' if node == 1 else 'Não Diabético'}"
                rules.append(rule)
        else:
            feature_name = feature_names[node['feature']]
            threshold = node['threshold']
            
            get_paths(node['left'], path + [f"{feature_name} ≤ {threshold:.1f}"])
            
            get_paths(node['right'], path + [f"{feature_name} > {threshold:.1f}"])
    
    get_paths(tree)
    return rules

def extract_rules_cart(tree, feature_names):
    """Extrai regras específicas do sklearn DecisionTreeClassifier"""
    tree_structure = tree.tree_
    rules = []
    
    def get_paths(node, path=[]):
        if tree_structure.children_left[node] != tree_structure.children_right[node]:
         
            feature = feature_names[tree_structure.feature[node]]
            threshold = tree_structure.threshold[node]
           
            get_paths(tree_structure.children_left[node], 
                     path + [f"{feature} ≤ {threshold:.1f}"])
            
            get_paths(tree_structure.children_right[node], 
                     path + [f"{feature} > {threshold:.1f}"])
        else:
            samples = tree_structure.n_node_samples[node]
            values = tree_structure.value[node][0]
            predicted_class = 'Diabético' if values[1] > values[0] else 'Não Diabético'
            
            if samples >= 5:  
                if path:
                    rule = "SE " + " E ".join(path) + f" ENTÃO {predicted_class}"
                    rules.append(rule)
    
    get_paths(0)
    return rules

def main():
    print("ANÁLISE DE DIABETES: ID3, C4.5 e CART\n")

    try:
        df = pd.read_csv("diabetes.csv")
        print("Dataset carregado: diabetes.csv")
    except:
        print("Dataset não encontrado. Usando dataset sintético.")
        df = criar_dataset_sintetico()
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Dados: {len(df)} amostras, {len(feature_names)} features")
    print(f"Treinamento: {len(X_train)} | Teste: {len(X_test)}\n")
    
    print("=== ALGORITMO ID3 ===")
    id3 = ID3(max_depth=3)
    id3.fit(X_train, y_train)
    id3_pred = id3.predict(X_test)
    print("Árvore:")
    print_tree(id3.tree, feature_names)
    print("\nRegras:")
    id3_rules = extract_rules(id3.tree, feature_names)
    for i, rule in enumerate(id3_rules, 1):
        print(f"{i}. {rule}")
    print(f"Acurácia: {accuracy_score(y_test, id3_pred):.3f}\n")
    
    print("=== ALGORITMO C4.5 ===")
    c45 = C45(max_depth=3)
    c45.fit(X_train, y_train)
    c45_pred = c45.predict(X_test)
    print("Árvore:")
    print_tree(c45.tree, feature_names)
    print("\nRegras:")
    c45_rules = extract_rules(c45.tree, feature_names)
    for i, rule in enumerate(c45_rules, 1):
        print(f"{i}. {rule}")
    print(f"Acurácia: {accuracy_score(y_test, c45_pred):.3f}\n")
    
    # CART
    print("=== ALGORITMO CART ===")
    cart = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=10, random_state=42)
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test)
    
    print("Árvore:")
    tree_text = export_text(cart, feature_names=feature_names, 
                           class_names=['Não Diabético', 'Diabético'])
    print(tree_text)
    
    print("Regras:")
    cart_rules = extract_rules_cart(cart, feature_names)
    for i, rule in enumerate(cart_rules, 1):
        print(f"{i}. {rule}")
    print(f"Acurácia: {accuracy_score(y_test, cart_pred):.3f}\n")
    
    print("=== COMPARAÇÃO DOS ALGORITMOS ===")
    print(f"ID3:  {accuracy_score(y_test, id3_pred):.3f}")
    print(f"C4.5: {accuracy_score(y_test, c45_pred):.3f}")
    print(f"CART: {accuracy_score(y_test, cart_pred):.3f}")
    

if __name__ == "__main__":
    main()