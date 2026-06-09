"""
Gera dataset sintético com padrões reais de fraude.
Fraudes e transações normais têm distribuições diferentes em cada feature.
Substitui o CSV em data/raw/.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_TOTAL = 7500
N_FRAUD  = 600   # ~8%
N_NORMAL = N_TOTAL - N_FRAUD

def gen(n, fraud: bool):
    if fraud:
        transaction_type  = np.random.choice(['Payment', 'Transfer', 'Withdrawal'], n, p=[0.45, 0.40, 0.15])
        payment_mode      = np.random.choice(['Card', 'NetBanking', 'UPI', 'Wallet'], n, p=[0.40, 0.20, 0.25, 0.15])
        device_type       = np.random.choice(['Android', 'iOS', 'Web'], n, p=[0.40, 0.30, 0.30])
        device_location   = np.random.choice(['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai'], n)
        avg_amt           = np.random.uniform(100, 20000, n)
        # Valor bem acima da média histórica
        multiplier        = np.random.lognormal(1.2, 0.6, n)
        transaction_amt   = np.clip(avg_amt * multiplier, 50, 50000)
        _age = np.concatenate([np.random.randint(10, 60, int(n*0.5)),
                               np.random.randint(60, 300, int(n*0.3)),
                               np.random.randint(300, 2000, n - int(n*0.5) - int(n*0.3))])
        np.random.shuffle(_age)
        account_age = _age
        _hr = np.concatenate([np.random.randint(0, 5, int(n*0.45)),   # madrugada
                              np.random.randint(5, 24, n - int(n*0.45))])
        np.random.shuffle(_hr)
        hour = _hr
        failed            = np.random.choice([0,1,2,3,4], n, p=[0.05, 0.10, 0.20, 0.35, 0.30])
        ip_risk           = np.random.beta(6, 2, n)                   # alto risco (0.6–1.0)
        is_intl           = np.random.binomial(1, 0.35, n)
        logins            = np.random.choice(range(1, 10), n, p=[0.02,0.03,0.05,0.10,0.15,0.20,0.20,0.15,0.10])
    else:
        transaction_type  = np.random.choice(['Payment', 'Transfer', 'Withdrawal'], n, p=[0.52, 0.33, 0.15])
        payment_mode      = np.random.choice(['Card', 'NetBanking', 'UPI', 'Wallet'], n, p=[0.33, 0.27, 0.25, 0.15])
        device_type       = np.random.choice(['Android', 'iOS', 'Web'], n, p=[0.46, 0.36, 0.18])
        device_location   = np.random.choice(['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai'], n)
        avg_amt           = np.random.uniform(100, 30000, n)
        # Valor próximo da média histórica
        multiplier        = np.random.lognormal(0.0, 0.25, n)
        transaction_amt   = np.clip(avg_amt * multiplier, 50, 50000)
        account_age       = np.random.randint(180, 2000, n)           # contas estabelecidas
        hour              = np.random.randint(7, 22, n)               # horário comercial/noite cedo
        failed            = np.random.choice([0,1,2,3,4], n, p=[0.55, 0.30, 0.10, 0.04, 0.01])
        ip_risk           = np.random.beta(2, 7, n)                   # baixo risco (0.05–0.35)
        is_intl           = np.random.binomial(1, 0.08, n)
        logins            = np.random.choice(range(1, 10), n, p=[0.20,0.25,0.20,0.15,0.10,0.05,0.03,0.01,0.01])

    return pd.DataFrame({
        'transaction_amount':       transaction_amt.round(2),
        'transaction_type':         transaction_type,
        'payment_mode':             payment_mode,
        'device_type':              device_type,
        'device_location':          device_location,
        'account_age_days':         account_age,
        'transaction_hour':         hour,
        'previous_failed_attempts': failed,
        'avg_transaction_amount':   avg_amt.round(2),
        'is_international':         is_intl,
        'ip_risk_score':            ip_risk.round(3),
        'login_attempts_last_24h':  logins,
        'fraud_label':              int(fraud),
    })

df_fraud  = gen(N_FRAUD,  fraud=True)
df_normal = gen(N_NORMAL, fraud=False)

df = pd.concat([df_fraud, df_normal], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, 'transaction_id', [f"T{i+1}" for i in range(N_TOTAL)])
df.insert(1, 'user_id',        [f"U{np.random.randint(1000,9999)}" for _ in range(N_TOTAL)])

out = Path("data/raw/Digital_Payment_Fraud_Detection_Dataset.csv")
df.to_csv(out, index=False)

# Relatório
fraud  = df[df.fraud_label == 1]
nfraud = df[df.fraud_label == 0]
print(f"Dataset gerado: {N_TOTAL} transacoes | {len(fraud)} fraudes ({len(fraud)/N_TOTAL*100:.1f}%)")
print()
print(f"{'Feature':<32} {'Fraude':>10} {'Normal':>10}")
print("-" * 54)
print(f"{'IP risk score':<32} {fraud.ip_risk_score.mean():>10.3f} {nfraud.ip_risk_score.mean():>10.3f}")
print(f"{'Idade conta (dias)':<32} {fraud.account_age_days.mean():>10.0f} {nfraud.account_age_days.mean():>10.0f}")
print(f"{'Valor / media historica':<32} {(fraud.transaction_amount/fraud.avg_transaction_amount).mean():>10.2f} {(nfraud.transaction_amount/nfraud.avg_transaction_amount).mean():>10.2f}")
print(f"{'Hora media':<32} {fraud.transaction_hour.mean():>10.1f} {nfraud.transaction_hour.mean():>10.1f}")
print(f"{'Tentativas falhas (media)':<32} {fraud.previous_failed_attempts.mean():>10.2f} {nfraud.previous_failed_attempts.mean():>10.2f}")
print(f"{'Internacional (%)':<32} {fraud.is_international.mean()*100:>10.1f} {nfraud.is_international.mean()*100:>10.1f}")
print(f"{'Tentativas login 24h':<32} {fraud.login_attempts_last_24h.mean():>10.2f} {nfraud.login_attempts_last_24h.mean():>10.2f}")
print()
print("Salvo em:", out)
