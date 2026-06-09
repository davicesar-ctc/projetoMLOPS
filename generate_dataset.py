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
        payment_mode      = np.random.choice(['Card', 'NetBanking', 'UPI', 'Wallet'], n, p=[0.38, 0.22, 0.25, 0.15])
        device_type       = np.random.choice(['Android', 'iOS', 'Web'], n, p=[0.42, 0.32, 0.26])
        device_location   = np.random.choice(['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai'], n)
        avg_amt           = np.random.uniform(100, 25000, n)
        # Valor acima da média histórica, mas com bastante sobreposição
        multiplier        = np.random.lognormal(0.30, 0.50, n)
        transaction_amt   = np.clip(avg_amt * multiplier, 50, 50000)
        # Contas tendem a ser mais novas, mas há fraudes em contas antigas
        _age = np.concatenate([np.random.randint(10, 120, int(n*0.28)),
                               np.random.randint(120, 600, int(n*0.30)),
                               np.random.randint(600, 2000, n - int(n*0.28) - int(n*0.30))])
        np.random.shuffle(_age)
        account_age = _age
        # ~25% na madrugada (sobrepõe com normais)
        _hr = np.concatenate([np.random.randint(0, 6, int(n*0.25)),
                              np.random.randint(6, 24, n - int(n*0.25))])
        np.random.shuffle(_hr)
        hour = _hr
        failed            = np.random.choice([0,1,2,3,4], n, p=[0.14, 0.22, 0.28, 0.21, 0.15])
        ip_risk           = np.random.beta(3.3, 3.0, n)              # risco moderado-alto, sobrepõe
        is_intl           = np.random.binomial(1, 0.22, n)
        logins            = np.random.choice(range(1, 10), n, p=[0.10,0.14,0.16,0.16,0.14,0.12,0.08,0.06,0.04])
    else:
        transaction_type  = np.random.choice(['Payment', 'Transfer', 'Withdrawal'], n, p=[0.52, 0.33, 0.15])
        payment_mode      = np.random.choice(['Card', 'NetBanking', 'UPI', 'Wallet'], n, p=[0.33, 0.27, 0.25, 0.15])
        device_type       = np.random.choice(['Android', 'iOS', 'Web'], n, p=[0.45, 0.35, 0.20])
        device_location   = np.random.choice(['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Mumbai'], n)
        avg_amt           = np.random.uniform(100, 30000, n)
        # Valor próximo da média histórica, com variação
        multiplier        = np.random.lognormal(0.10, 0.42, n)
        transaction_amt   = np.clip(avg_amt * multiplier, 50, 50000)
        # Contas em geral estabelecidas, mas com vários usuários novos legítimos
        _age = np.concatenate([np.random.randint(15, 300, int(n*0.30)),
                               np.random.randint(300, 2000, n - int(n*0.30))])
        np.random.shuffle(_age)
        account_age = _age
        # Maioria de dia, mas ~10% de madrugada (compras noturnas legítimas)
        _hr = np.concatenate([np.random.randint(0, 6, int(n*0.10)),
                              np.random.randint(6, 24, n - int(n*0.10))])
        np.random.shuffle(_hr)
        hour = _hr
        failed            = np.random.choice([0,1,2,3,4], n, p=[0.38, 0.30, 0.18, 0.09, 0.05])
        ip_risk           = np.random.beta(2.6, 4.2, n)             # risco baixo-moderado, sobrepõe
        is_intl           = np.random.binomial(1, 0.10, n)
        logins            = np.random.choice(range(1, 10), n, p=[0.18,0.22,0.19,0.14,0.10,0.07,0.05,0.03,0.02])

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
