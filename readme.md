# Fraud Detection Pipeline (MLOps)

Sistema de detecção de fraudes em pagamentos digitais com pipeline MLOps completo: treinamento reproduzível via DVC, rastreamento de experimentos com MLflow e API de predição em tempo real via FastAPI — tudo orquestrado com Docker.

---

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Machine Learning | XGBoost, Random Forest (Scikit-Learn) |
| Pipeline | DVC |
| Experimentos | MLflow |
| API | FastAPI + Uvicorn |
| Infraestrutura | Docker + Docker Compose |
| Linguagem | Python 3.12+ |

---

## Passo a Passo para Executar

### 1. Instalar dependências (primeira vez)

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

### 2. Subir o servidor MLflow (Docker)

```bash
docker-compose up mlflow -d
```

Acesse a interface em: **http://localhost:5000**

---

### 3. Treinar os modelos

```bash
dvc repro
```

O que acontece por baixo:
- Etapa 1 — **Pré-processamento**: limpa os dados, cria features e salva as matrizes de treino/teste
- Etapa 2 — **Treino**: treina XGBoost e Random Forest, loga métricas e artefatos no MLflow, registra o modelo campeão no Model Registry com o alias `@champion`

Para ver as métricas no terminal após o treino:

```bash
dvc metrics show
```

---

### 4. Subir a API de predição (Docker)

```bash
docker-compose up api --build
```

Na inicialização, a API conecta ao MLflow, carrega o modelo campeão e o preprocessor diretamente do registry — sem usar arquivos locais.

---

### 5. Testar a API

Acesse a interface interativa (Swagger): **http://localhost:8000/docs**

No Swagger:
1. Clique em `POST /predict`
2. Clique em **"Try it out"**
3. Os campos já vêm preenchidos com um exemplo suspeito
4. Modifique os valores que quiser
5. Clique em **"Execute"**

Resposta esperada:
```json
{
  "fraude_detectada": true,
  "probabilidade": 0.9123,
  "threshold": 0.1
}
```

Verificar se a API está de pé:
```bash
curl http://localhost:8000/health
```

---

## Campos do /predict

| Campo | O que é | Valores aceitos |
|---|---|---|
| `valor_transacao` | Valor da transação em reais | 50 a 50000 |
| `tipo_transacao` | Tipo da operação | `Pagamento` / `Transferência` / `Saque` |
| `meio_pagamento` | Meio de pagamento | `Cartão` / `Banco Online` / `PIX` / `Carteira Digital` |
| `tipo_dispositivo` | Dispositivo utilizado | `Android` / `iOS` / `Navegador` |
| `localizacao` | Cidade da transação | `São Paulo` / `Rio de Janeiro` / `Brasília` / `Belo Horizonte` / `Salvador` |
| `idade_conta_dias` | Dias desde a criação da conta | 10 a 1999 |
| `hora_transacao` | Hora do dia | 0 a 23 |
| `tentativas_falhas` | Tentativas de pagamento falhas anteriores | 0 a 4 |
| `valor_medio_transacoes` | Média histórica de transações do usuário | 100 a 30000 |
| `transacao_internacional` | É internacional? | `0` = Não / `1` = Sim |
| `risco_ip` | Score de risco do IP | 0.0 (seguro) a 1.0 (suspeito) |
| `tentativas_login_24h` | Logins nas últimas 24h | 1 a 9 |

### Exemplos de perfis para teste

**Perfil suspeito** → tende a retornar `fraude_detectada: true`
```json
{
  "valor_transacao": 45000, "valor_medio_transacoes": 200,
  "hora_transacao": 2, "risco_ip": 0.95,
  "transacao_internacional": 1, "idade_conta_dias": 12,
  "tentativas_falhas": 4, "tentativas_login_24h": 9,
  "tipo_transacao": "Transferência", "meio_pagamento": "PIX",
  "tipo_dispositivo": "Navegador", "localizacao": "São Paulo"
}
```

**Perfil normal** → tende a retornar `fraude_detectada: false`
```json
{
  "valor_transacao": 350, "valor_medio_transacoes": 500,
  "hora_transacao": 14, "risco_ip": 0.05,
  "transacao_internacional": 0, "idade_conta_dias": 900,
  "tentativas_falhas": 0, "tentativas_login_24h": 2,
  "tipo_transacao": "Pagamento", "meio_pagamento": "Cartão",
  "tipo_dispositivo": "Android", "localizacao": "Rio de Janeiro"
}
```

---

## Estrutura do Projeto

```
├── api/                  # API FastAPI
│   ├── app.py            # Endpoints /health e /predict
│   └── Dockerfile
├── data/raw/             # Dataset original
├── pipeline/             # Scripts das etapas DVC
│   ├── preprocess.py
│   └── train.py
├── src/                  # Lógica central
│   ├── preprocessor.py   # Limpeza e feature engineering
│   ├── model_trainer.py  # Treino + registro no MLflow
│   └── config.py
├── docker-compose.yml    # MLflow + API
├── dvc.yaml              # Definição do pipeline
└── params.yaml           # Hiperparâmetros
```

---

## Estratégia do Modelo

O threshold de decisão é **0.10** — intencionalmente baixo para maximizar a captura de fraudes (alto recall), aceitando mais falsos positivos. Em detecção de fraude financeira, deixar uma fraude passar é mais custoso do que bloquear uma transação legítima.
