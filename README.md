# 🔎 Framework de Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-818cf8?style=for-the-badge)
![LIME](https://img.shields.io/badge/LIME-Local_Explanations-34d399?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-fbbf24?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-34d399?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-60a5fa?style=for-the-badge)

> **4 XAI techniques unified in one framework + interactive Streamlit dashboard | SHAP · LIME · Counterfactuals | Applied to Fintech credit scoring | Python**

---

## 📌 Visão Geral

Modelos de ML em produção precisam ser explicáveis — por exigência regulatória (LGPD, GDPR), por confiança dos stakeholders e pela responsabilidade no impacto das decisões. Este framework unifica quatro técnicas complementares de XAI em uma classe única, reutilizável e documentada, com foco em decisões de crédito Fintech — e inclui um **dashboard interativo em Streamlit** para demonstração ao vivo.

### Técnicas Implementadas

| Técnica | Escopo | Velocidade | Interpretabilidade |
|:--------|:-------|:----------:|:-----------------:|
| **SHAP** | Global + Local | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **LIME** | Local | ⚡⚡ | ⭐⭐⭐⭐ |
| **Permutation Importance** | Global | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **Counterfactual** | Local | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 🎯 Problema de Negócio

**Contexto:** Uma Fintech precisa explicar suas decisões de crédito automatizadas — para o time técnico, para os stakeholders e, principalmente, para os clientes que tiveram crédito negado.

**Desafios endereçados:**
- LGPD (Art. 20) e GDPR (Art. 22) exigem explicação de decisões automatizadas
- Times de negócio precisam confiar nos modelos antes de adotá-los
- Clientes têm direito a saber o que precisa mudar para serem aprovados
- Auditorias precisam de relatórios documentados e reproduzíveis

---

## 🖥️ Dashboard Interativo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xaidashboard-879fp9cpmsmnjbalhee2yr.streamlit.app/#framework-de-explainable-ai)

🔗 **[Acessar o Dashboard ao vivo](https://xaidashboard-879fp9cpmsmnjbalhee2yr.streamlit.app/#framework-de-explainable-ai)**

O framework inclui um **dashboard Streamlit** com 5 páginas:

| Página | Conteúdo |
|:-------|:---------|
| 🏠 **Visão Geral** | Métricas do modelo e distribuição de risco |
| 🔴 **Explicar Cliente** | Sliders interativos + SHAP, LIME e Counterfactual em tempo real |
| 📊 **SHAP Global** | Ranking de importância e summary plot |
| 🔀 **Permutation Importance** | Queda de ROC-AUC por feature com intervalo de confiança |
| 📋 **Relatório Completo** | Relatório de auditoria gerado automaticamente com download |

### Rodar localmente
```bash
pip install streamlit shap lime xgboost scikit-learn pandas numpy matplotlib
streamlit run xai_dashboard.py
```

### Deploy gratuito no Streamlit Cloud
```bash
# 1. Subir xai_dashboard.py para o GitHub
git add xai_dashboard.py requirements.txt
git commit -m "feat: add streamlit dashboard"
git push origin main

# 2. Acessar share.streamlit.io → conectar repositório → Deploy
# Link público gerado em ~2 minutos
```

---

## 🗂️ Estrutura do Projeto

```
xai-framework/
│
├── 📓 notebooks/
│   └── xai_framework.ipynb           # Notebook principal completo
│
├── 📁 src/
│   ├── shap_explainer.py             # Módulo SHAP
│   ├── lime_explainer.py             # Módulo LIME
│   ├── permutation_explainer.py      # Módulo Permutation Importance
│   ├── counterfactual_explainer.py   # Módulo Counterfactual
│   └── explainer_framework.py        # Classe unificada ExplainerFramework
│
├── xai_dashboard.py                  # Dashboard Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Guia Rápido de Uso

```python
from src.explainer_framework import ExplainerFramework

# 1. Inicializar com qualquer modelo sklearn/xgboost
xai = ExplainerFramework(
    model               = meu_modelo,
    X_train             = X_train,
    X_test              = X_test,
    y_test              = y_test,
    feature_names       = feature_names,
    model_type          = 'tree',            # 'tree' ou 'kernel'
    immutable_features  = ['age', 'gender']  # features não alteráveis no counterfactual
)

# 2. Ajustar todos os módulos de uma vez
xai.fit_all()

# 3. Explicar uma instância (SHAP + LIME + Counterfactual)
xai.explain_instance(idx=42)

# 4. Visão global de importância
xai.global_summary()

# 5. Comparar SHAP vs Permutation Importance
xai.compare_importance_methods()
```

---

## 🔬 Técnicas Implementadas

### Módulo 1 — SHAP
- `SHAPExplainer` com `TreeExplainer` (tree-based) e `KernelExplainer` (qualquer modelo)
- Summary plot (dot e bar), Waterfall individual, Dependence plot
- Ranking global por `mean(|SHAP value|)`

### Módulo 2 — LIME
- `LIMEExplainer` com `LimeTabularExplainer`
- Explicações locais com modelo linear aproximado
- Comparação lado a lado de múltiplas instâncias

### Módulo 3 — Permutation Importance
- `PermutationImportanceExplainer` com 20 repetições por default
- Bar chart com intervalo de confiança + boxplot de variabilidade
- Comparação entre múltiplos modelos simultaneamente

### Módulo 4 — Counterfactual Explanations
- `CounterfactualExplainer` com busca greedy por perturbação mínima
- Respeita features imutáveis (ex: idade, gênero)
- Relatório em linguagem natural para comunicar a decisão ao cliente

---

## 💡 Quando Usar Cada Técnica

| Situação | Técnica Recomendada |
|:---------|:--------------------|
| Entender quais features o modelo usa globalmente | SHAP (bar plot) |
| Explicar decisão para o time técnico | SHAP (waterfall) |
| Explicar decisão para stakeholders não técnicos | LIME |
| Validar que o modelo não usa features espúrias | Permutation Importance |
| Comunicar negativa de crédito ao cliente | Counterfactual |
| Auditoria regulatória (LGPD/GDPR) | Counterfactual + SHAP |

---

## ⚙️ Como Executar o Notebook

### 1. Clone o repositório
```bash
git clone https://github.com/GabrielAlessi/xai-framework.git
cd xai-framework
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o notebook
```bash
jupyter notebook notebooks/xai_framework.ipynb
```

> **💡 Kaggle:** Disponível publicamente com todos os outputs renderizados. A última célula do notebook salva automaticamente o `xai_dashboard.py` para download.

---

## 📦 Dependências

```
shap>=0.42.0
lime>=0.2.0
xgboost>=1.7.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
streamlit>=1.28.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 🚀 Próximos Passos

- [ ] **API REST com FastAPI** — endpoint `/explain` retornando JSON com SHAP + counterfactual
- [ ] **Análise de Fairness** — verificar viés por faixa etária, renda e gênero
- [ ] **Monitoring de Concept Drift** — detectar quando a importância das features muda em produção
- [ ] **Suporte a NLP** — estender o framework para classificadores de texto (SHAP para BERT)
- [ ] **Testes automatizados** — cobertura com pytest para todos os módulos

---

## 👨‍💻 Autor

**Gabriel Alessi Naumann**  
Cientista de Dados Sênior

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--alessi--naumann-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/gabriel-alessi-naumann/)
[![GitHub](https://img.shields.io/badge/GitHub-GabrielAlessi-181717?style=flat&logo=github)](https://github.com/GabrielAlessi)
[![Kaggle](https://img.shields.io/badge/Kaggle-gabrielalessinaumann-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/gabrielalessinaumann)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

*⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório!*
