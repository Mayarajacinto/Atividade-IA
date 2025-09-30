import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PRISMClassifier:
    def __init__(self):
        self.rules = []
        self.feature_names = None
        
    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        X_disc = self._discretize(X.values)
        
        self.rules = []
        for target_class in [0, 1]:
            rules = self._generate_rules(X_disc, y.values, target_class)
            self.rules.extend(rules)
        
        return self
    
    def _discretize(self, X):
        X_disc = np.zeros_like(X, dtype=int)
        self.thresholds = {}
        
        for i in range(X.shape[1]):
            values = X[:, i]
            low, high = np.percentile(values, [33, 67])
            X_disc[:, i] = np.digitize(values, [low, high])
            self.thresholds[i] = [low, high]
            
        return X_disc
    
    def _generate_rules(self, X, y, target_class):
        rules = []
        indices = np.where(y == target_class)[0]
        
        for _ in range(3):  # Máximo 3 regras por classe
            if len(indices) < 15:
                break
                
            rule = self._create_rule(X, y, target_class, indices)
            if rule:
                rules.append(rule)
                # Remove exemplos cobertos
                covered = self._get_covered(X, rule['conditions'], indices)
                indices = np.setdiff1d(indices, covered)
                
        return rules
    
    def _create_rule(self, X, y, target_class, indices):
        conditions = []
        current_indices = indices.copy()
        
        for _ in range(2): 
            best_acc = 0
            best_condition = None
            best_indices = None
            
            for feature in range(X.shape[1]):
                for value in [0, 1, 2]: 
                    test_conditions = conditions + [(feature, value)]
                    satisfied = self._get_covered(X, test_conditions, current_indices)
                    
                    if len(satisfied) < 10:
                        continue
                        
                    acc = np.sum(y[satisfied] == target_class) / len(satisfied)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_condition = (feature, value)
                        best_indices = satisfied
            
            if best_condition is None or best_acc > 0.8:
                break
                
            conditions.append(best_condition)
            current_indices = best_indices
        
        if len(current_indices) >= 10:
            correct = np.sum(y[current_indices] == target_class)
            return {
                'conditions': conditions,
                'target_class': target_class,
                'accuracy': correct / len(current_indices),
                'support': len(current_indices),
                'correct': correct
            }
        return None
    
    def _get_covered(self, X, conditions, indices):
        result = indices.copy()
        for feature, value in conditions:
            mask = X[result, feature] == value
            result = result[mask]
        return result
    
    def predict(self, X):
        X_disc = self._discretize_test(X.values)
        predictions = []
        
        for i in range(X.shape[0]):
            votes = [0, 0]
            
            for rule in self.rules:
                if self._satisfies_rule(X_disc[i], rule['conditions']):
                    votes[rule['target_class']] += rule['accuracy']
            
            predictions.append(1 if votes[1] > votes[0] else 0)
            
        return np.array(predictions)
    
    def _discretize_test(self, X):
        X_disc = np.zeros_like(X, dtype=int)
        for i in range(X.shape[1]):
            X_disc[:, i] = np.digitize(X[:, i], self.thresholds[i])
        return X_disc
    
    def _satisfies_rule(self, x, conditions):
        for feature, value in conditions:
            if x[feature] != value:
                return False
        return True
    
    def print_rules(self):
        print("BASE DE REGRAS: ALGORITMO PRISM")
        
        labels = ['Baixo', 'Médio', 'Alto']
        class_names = ['Não Diabético', 'Diabético']
        
        for i, rule in enumerate(self.rules, 1):
            print(f"\nREGRA {i}:")
            
            if rule['conditions']:
                conditions_str = []
                for feature_idx, value in rule['conditions']:
                    feature_name = self.feature_names[feature_idx]
                    value_label = labels[value]
                    conditions_str.append(f"{feature_name} é {value_label}")
                print(f"  SE: {' E '.join(conditions_str)}")
            else:
                print(f"  SE: (sempre)")
                
            class_name = class_names[rule['target_class']]
            print(f"  ENTÃO: {class_name}")
            print(f"  Confiança: {rule['accuracy']:.1%} ({rule['correct']}/{rule['support']} amostras)")

def criar_dataset():
    np.random.seed(42)
    n = 500
    
    data = {
        'Glucose': np.clip(np.random.normal(120, 30, n), 50, 200),
        'BMI': np.clip(np.random.normal(30, 8, n), 15, 50),
        'Age': np.clip(np.random.normal(45, 15, n), 20, 80).astype(int),
        'BloodPressure': np.clip(np.random.normal(75, 15, n), 50, 120)
    }
    
    df = pd.DataFrame(data)
    
    score = (
        0.3 * (df['Glucose'] > 126) +
        0.2 * (df['BMI'] > 30) +
        0.2 * (df['Age'] > 50) +
        0.1 * (df['BloodPressure'] > 90)
    )
    
    df['Outcome'] = (score > 0.4).astype(int)
    return df

def main():
    
    # Carregar dados
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        df = criar_dataset()
        print("Dataset sintético criado!")
    
    print(f"Amostras: {len(df)}")
    
    # Preparar dados
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    prism = PRISMClassifier()
    prism.fit(X_train, y_train)
    y_pred = prism.predict(X_test)
    prism.print_rules()
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n")
    print("MÉTRICAS DE DESEMPENHO")
    print(f"Acurácia: {accuracy:.3f} ({accuracy:.1%})")
    
    if accuracy >= 0.75:
        print("Boa performance")
    elif accuracy >= 0.65:
        print("Performance aceitável")
    else:
        print("Performance baixa")

if __name__ == "__main__":
    main()