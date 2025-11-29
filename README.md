🧠 Totem IA — Sprint 2
Integração Sensores → SQL → Analytics → Machine Learning
Flexmedia Challenge — FIAP
__________________________________________________________________________
📌 1. Introdução

A Sprint 2 representa a primeira etapa prática da implementação do Totem IA, conectando coleta de dados simulados, armazenamento estruturado, tratamento, análise, visualização e aprendizado de máquina supervisionado.

Tudo foi construído com base no planejamento arquitetural da Sprint 1, tornando esta entrega um protótipo funcional real, que demonstra:

como o totem coleta informações do mundo físico,

como essas informações são estruturadas e armazenadas,

como são transformadas em métricas,

como podem alimentar sistemas inteligentes.

🎯 2. Objetivos da Sprint 2

✔ Demonstrar integração entre sensores/simulações e banco SQL
✔ Registrar dados brutos e eventos de interação
✔ Criar dashboard com métricas iniciais
✔ Realizar análises estatísticas descritivas
✔ Treinar um modelo supervisionado simples (toque curto vs longo)
✔ Garantir organização e limpeza dos dados
✔ Representar claramente todo o fluxo do pipeline

🧱 3. Arquitetura Implementada

A arquitetura prática desenvolvida nesta sprint segue o fluxo:

flowchart TD
    A[Sensores Simulados<br>sensor_sim.py] --> B[Arquivo CSV<br>sample_interactions.csv]
    B --> C[Ingestão SQL<br>ingest_to_sql.py]
    C --> D[Banco SQLite<br>flexmedia.sqlite]
    D --> E[Análise Estatística<br>analysis.py]
    E --> F[Relatórios e Gráficos<br>media/ e data/report_summary.json]
    E --> G[Machine Learning<br>ml_train.py]
    F --> H[Dashboard Front-end<br>Streamlit]


Esta arquitetura implementa todo o pipeline real de dados, cobrindo todos os requisitos técnicos da Sprint 2.

🧬 4. Fluxo de Dados Completo
Fluxo Entrada → Processamento → Saída
sequenceDiagram
    participant S as Sensor Simulado
    participant CSV as CSV
    participant SQL as Banco SQLite
    participant A as Script de Análise
    participant D as Dashboard
    participant ML as Modelo ML

    S->>CSV: Registro de evento bruto
    CSV->>SQL: Ingestão dos dados
    SQL->>A: Carregamento dos dados
    A->>A: Limpeza, deduplicação e validação
    A->>D: Dados analisados e estruturados
    A->>ML: Dataset tratado
    ML->>ML: Treinamento e validação

🗄️ 5. Modelo de Dados (DER)
erDiagram
    SESSAO {
        string id PK
        string idioma
        string inicio
        string fim
        int duracao_seconds
    }

    INTERACAO {
        int id PK
        string sessao_id FK
        string timestamp
        string sensor_id
        string tipo
        string pergunta
        string resposta
        string content_id
        float duration
        int value
    }

    SESSAO ||--|{ INTERACAO : "possui"

📁 6. Estrutura do Repositório
Enterprise-Challenge---Sprint-2---FlexMedia/
│
├── data/
│   ├── sample_interactions.csv
│   ├── flexmedia.sqlite
│   └── report_summary.json
│
├── media/
│   ├── interacoes_por_tipo.png
│   ├── touch_dist.png
│
├── sensor_sim.py
├── ingest_to_sql.py
├── analysis.py
├── dashboard_streamlit.py
├── ml_train.py
└── README.md

🧩 7. Módulos Implementados (Códigos Completos)
7.1. Simulador de Sensores — sensor_sim.py
import csv, time, random, uuid
from datetime import datetime
import os

CSV_OUT = 'data/sample_interactions.csv'

def random_interaction(session_id):
    sensor = random.choice(['touch_1','touch_2','pres_1'])
    event = random.choices(['touch','presence','qr'], weights=[0.7,0.25,0.05])[0]
    duration = round(random.uniform(0.05,4.0) if event=='touch' else 0.0,3)
    value = 1 if event in ('touch','presence') else 0
    lang = random.choices(['pt-BR','en-US','es-ES'], weights=[0.7,0.2,0.1])[0]
    content_id = str(uuid.uuid4()) if random.random() < 0.6 else None

    pergunta, resposta = None, None
    if event == 'touch' and random.random() < 0.4:
        pergunta = random.choice(["Qual é esse animal?","Horário?","Onde fica o banheiro?"])
        resposta = "Resposta simulada."

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'sensor_id': sensor,
        'event_type': event,
        'duration': duration,
        'value': value,
        'session_anon_id': session_id,
        'language': lang,
        'content_id': content_id,
        'pergunta': pergunta,
        'resposta': resposta
    }

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = None
        for s in range(50):
            session_id = str(uuid.uuid4())
            for i in range(random.randint(3,20)):
                row = random_interaction(session_id)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                time.sleep(0.01)
    print('CSV gerado:', CSV_OUT)

7.2. Ingestão de Dados SQL — ingest_to_sql.py
import sqlite3, csv

DB = 'data/flexmedia.sqlite'
CSV = 'data/sample_interactions.csv'

SCHEMA = '''
CREATE TABLE IF NOT EXISTS sessao (
  id TEXT PRIMARY KEY,
  idioma TEXT,
  inicio TEXT,
  fim TEXT,
  duracao_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS interacao (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sessao_id TEXT,
  timestamp TEXT,
  sensor_id TEXT,
  tipo TEXT,
  pergunta TEXT,
  resposta TEXT,
  content_id TEXT,
  duration REAL,
  value INTEGER
);

CREATE INDEX IF NOT EXISTS idx_interacao_sessao ON interacao(sessao_id);
'''

def ingest(csv_path, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sess_id = row['session_anon_id']

            cur.execute('SELECT id FROM sessao WHERE id=?',(sess_id,))
            if cur.fetchone() is None:
                cur.execute(
                    'INSERT INTO sessao (id, idioma, inicio) VALUES (?,?,?)',
                    (sess_id, row['language'], row['timestamp'])
                )

            cur.execute(
                '''INSERT INTO interacao 
                (sessao_id, timestamp, sensor_id, tipo, pergunta, resposta, content_id, duration, value)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (sess_id, row['timestamp'], row['sensor_id'], row['event_type'], 
                 row['pergunta'], row['resposta'], row['content_id'], 
                 float(row['duration']), int(row['value']))
            )

    conn.commit()
    conn.close()
    print("Ingestão concluída.")

if __name__ == "__main__":
    ingest(CSV, DB)

7.3. Análises Estatísticas — analysis.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

DB = 'data/flexmedia.sqlite'
conn = sqlite3.connect(DB)

df = pd.read_sql_query('SELECT * FROM interacao', conn, parse_dates=['timestamp'])

df = df.drop_duplicates(subset=['timestamp','sensor_id','tipo'])

df['touch_type'] = df['duration'].apply(
    lambda d: 'none' if d==0 else ('short' if d <= 0.5 else 'long')
)

os.makedirs('media', exist_ok=True)

plt.figure()
df['tipo'].value_counts().plot(kind='bar')
plt.title('Interações por Tipo')
plt.tight_layout()
plt.savefig('media/interacoes_por_tipo.png')

plt.figure()
df['touch_type'].value_counts().plot(kind='bar')
plt.title('Distribuição de Toques (Short vs Long)')
plt.tight_layout()
plt.savefig('media/touch_dist.png')

report = {
    "total_interacoes": len(df),
    "interacoes_por_tipo": df['tipo'].value_counts().to_dict(),
    "duracao_media": float(df['duration'].mean() or 0)
}

with open('data/report_summary.json','w',encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Análises concluídas.")

7.4. Dashboard Interativo — dashboard_streamlit.py
import streamlit as st
import sqlite3
import pandas as pd

DB = "data/flexmedia.sqlite"

st.title("Dashboard Totem IA — Sprint 2")

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM interacao", conn, parse_dates=['timestamp'])

st.metric("Total de Interações", len(df))

st.subheader("Interações por Tipo")
st.bar_chart(df['tipo'].value_counts())

st.subheader("Últimas 20 Interações")
st.dataframe(df.sort_values('timestamp', ascending=False).head(20))

7.5. Machine Learning Supervisionado — ml_train.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DB = "data/flexmedia.sqlite"
conn = sqlite3.connect(DB)

df = pd.read_sql_query(
    "SELECT duration FROM interacao WHERE duration > 0",
    conn
)

df['label'] = (df['duration'] > 0.5).astype(int)

X = df[['duration']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

📊 8. Prints das Execuções (Simulados)

Geração do CSV
CSV gerado: data/sample_interactions.csv

Ingestão SQL
Ingestão concluída.

Análises
Análises concluídas.

Modelo supervisionado
              precision recall f1-score support
...

Dashboard
http://localhost:8501

🚀 9. Como Executar o Projeto

pip install pandas streamlit matplotlib scikit-learn


Gerar dados brutos:

python sensor_sim.py


Ingerir no banco:

python ingest_to_sql.py


Analisar:

python analysis.py


Abrir dashboard:

streamlit run dashboard_streamlit.py


Treinar modelo:

python ml_train.py

## Link youtube https://youtu.be/a5JUQUJ5Zxs
