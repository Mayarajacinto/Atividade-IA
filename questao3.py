import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

def carregar_dados():
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        print("Usando dataset sintético")
        df = criar_dataset_sintetico()
    
    return df

def desenhar_arvore(tree, feature_names, class_names, max_depth=4):
    print("ÁRVORE DE DECISÃO")
    
    tree_structure = tree.tree_
    
    def draw_node(node, depth=0, prefix="", is_last=True, condition="ROOT"):
        if depth > max_depth:
            return
            
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        samples = tree_structure.n_node_samples[node]
        
        if tree_structure.children_left[node] != tree_structure.children_right[node]:
            feature_idx = tree_structure.feature[node]
            feature_name = feature_names[feature_idx]
            threshold = tree_structure.threshold[node]
            
            print(f"{prefix}{connector}{condition} (n={samples})")
            
            left_condition = f"{feature_name} <= {threshold:.1f}"
            draw_node(
                tree_structure.children_left[node], 
                depth + 1, 
                prefix + extension, 
                False, 
                left_condition
            )
            right_condition = f"{feature_name} > {threshold:.1f}"
            draw_node(
                tree_structure.children_right[node], 
                depth + 1, 
                prefix + extension, 
                True, 
                right_condition
            )
        else:
            values = tree_structure.value[node][0]
            predicted_class = np.argmax(values)
            confidence = values[predicted_class] / samples
            class_name = class_names[predicted_class]
            
            print(f"{prefix}{connector}PREDIÇÃO: {class_name} ({confidence:.1%}, n={samples})")
    
    draw_node(0)

def extrair_regras(tree, feature_names, class_names):
    print("REGRAS")
    
    tree_structure = tree.tree_
    regras = []
    
    def extrair_caminho(node, caminho=[]):
        if tree_structure.children_left[node] != tree_structure.children_right[node]:
            feature_idx = tree_structure.feature[node]
            feature_name = feature_names[feature_idx]
            threshold = tree_structure.threshold[node]
            
            caminho_esquerdo = caminho + [f"{feature_name} <= {threshold:.1f}"]
            extrair_caminho(tree_structure.children_left[node], caminho_esquerdo)
            
            caminho_direito = caminho + [f"{feature_name} > {threshold:.1f}"]
            extrair_caminho(tree_structure.children_right[node], caminho_direito)
        else:
            samples = tree_structure.n_node_samples[node]
            values = tree_structure.value[node][0]
            predicted_class = np.argmax(values)
            confidence = values[predicted_class] / samples
            class_name = class_names[predicted_class]
            
            if samples >= 10: 
                regras.append({
                    'condicoes': caminho.copy(),
                    'predicao': class_name,
                    'confianca': confidence,
                    'amostras': samples
                })
    
    extrair_caminho(0)
    
    for i, regra in enumerate(regras, 1):
        print(f"\nREGRA {i}:")
        if regra['condicoes']:
            print(f"  SE: {' E '.join(regra['condicoes'])}")
        else:
            print(f"  SE: (sempre)")
        print(f"  ENTÃO: {regra['predicao']}")
        print(f"  Confiança: {regra['confianca']:.1%} ({regra['amostras']} amostras)")
    
    return regras

def main():
    print("ANÁLISE DE DIABETES: ÁRVORE DE DECISÃO")
    
    df = carregar_dados()
    print(f"Amostras: {len(df)} | Features: {len(df.columns)-1}")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    arvore = DecisionTreeClassifier(
        random_state=42,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10
    )
    
    arvore.fit(X_train, y_train)
    
    y_pred = arvore.predict(X_test)
    
    desenhar_arvore(arvore, X.columns.tolist(), ['Não Diabético', 'Diabético'])
    
    regras = extrair_regras(arvore, X.columns.tolist(), ['Não Diabético', 'Diabético'])
    
    accuracy = accuracy_score(y_test, y_pred)
   
    print("\n")
    print(f"Acurácia: {accuracy:.4f} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("Boa performance")
    elif accuracy >= 0.7:
        print("Performance aceitável")
    else:
        print("Precisa de melhorias")

if __name__ == "__main__":
    main()