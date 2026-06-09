# Perguntas e Respostas — Avaliação MLOps

---

## Fundamentos

**1. O que diferencia um modelo de ML na produção de um modelo em notebook?**

Um notebook é um ambiente exploratório: os dados são fixos, a execução é manual, não há tratamento de erros, e o resultado só existe na máquina de quem rodou. Um modelo em produção precisa responder requisições em tempo real, com latência previsível, tratamento de falhas, logs estruturados, versionamento e capacidade de rollback. A diferença fundamental é que o notebook serve para descoberta — a produção serve para confiabilidade. Neste projeto, a separação é explícita: o pipeline (DVC + scripts Python) treina e versiona o modelo; a API FastAPI expõe o modelo treinado como serviço, com validação de entrada via Pydantic e carregamento a partir do MLflow Registry — nunca diretamente de um notebook.

---

**2. O que é model decay (decaimento de modelo) e como você detectaria cada tipo no seu projeto?**

Model decay é a degradação gradual da performance de um modelo em produção. Existem dois tipos principais:

- **Data drift**: a distribuição das features muda ao longo do tempo (ex: novos padrões de fraude surgem com PIX que não existiam no treino). Detecta-se monitorando a distribuição estatística das features de entrada comparada com a distribuição do treino — testes como KS (Kolmogorov-Smirnov) ou PSI (Population Stability Index) são comuns.
- **Concept drift**: a relação entre as features e o target muda (ex: uma transação às 3h deixa de ser suspeita porque virou padrão de delivery noturno). Detecta-se monitorando as métricas do modelo em produção ao longo do tempo com dados rotulados.

Neste projeto, o ponto de entrada para detectar ambos seria monitorar a distribuição de `risco_ip`, `idade_conta_dias` e `valor_transacao / valor_medio_transacoes` nas requisições ao `/predict` e comparar com a distribuição do dataset de treino.

---

**3. Por que reprodutibilidade é um requisito crítico em sistemas de ML e como você a garante?**

Sem reprodutibilidade não é possível: auditar um modelo em produção, debugar uma queda de performance, atender exigências regulatórias (LGPD, SOX), ou colaborar em equipe sem resultados divergentes entre máquinas. Em ML, qualquer variação — nos dados, no ambiente, nos hiperparâmetros, no seed — pode produzir um modelo diferente.

Neste projeto, a reprodutibilidade é garantida em quatro camadas:
- **Dados**: DVC rastreia o arquivo CSV por hash e versiona as matrizes processadas — rodar `dvc repro` sempre parte do mesmo estado.
- **Hiperparâmetros**: centralizados em `params.yaml`, versionado junto com o código no Git.
- **Ambiente**: `requirements.txt` com versões fixas; Docker garante que o ambiente de execução seja idêntico em qualquer máquina.
- **Aleatoriedade**: `random_state=42` fixo no treino e no split de dados.

---

## Versionamento e Rastreabilidade

**4. Qual a diferença entre versionar código, dados e modelos? Por que cada um exige abordagens diferentes?**

- **Código** é texto — o Git funciona perfeitamente: diffs legíveis, merge, histórico de linha a linha.
- **Dados** são binários grandes (CSVs, parquets) que não pertencem ao repositório Git (tamanho, sensibilidade). O DVC resolve isso armazenando apenas o hash do arquivo no Git e o arquivo em si num armazenamento externo (S3, GCS, local). Assim, `data/raw/dataset.csv.dvc` no Git aponta para a versão exata do dado.
- **Modelos** são artefatos binários com metadados ricos: quem treinou, com quais dados, quais hiperparâmetros, quais métricas. O MLflow Model Registry resolve isso: cada versão do modelo tem um `run_id` que aponta para o experimento completo, e aliases como `@champion` indicam o estado atual em produção.

---

**5. No seu projeto, como você garantiu rastreabilidade entre um modelo em produção e o dado que o gerou?**

A cadeia é completa e rastreável em duas direções:

1. **DVC → dados e pipeline**: o arquivo `dvc.lock` registra o hash exato do dataset e dos artefatos gerados (matrizes, preprocessor) a cada execução. Dado um commit do Git, é possível reconstruir exatamente os dados que geraram aquele modelo.

2. **MLflow → experimento e modelo**: cada run no MLflow tem um `run_id` único que registra os hiperparâmetros (`params.yaml`), as métricas (ROC-AUC, PR-AUC, Accuracy) e os artefatos (modelo + `preprocessor.joblib`). O modelo registrado no Model Registry com alias `@champion` aponta diretamente para esse `run_id`.

Para saber exatamente com quais dados um modelo em produção foi treinado: Model Registry → `run_id` → MLflow run → commit do Git no momento do treino → `dvc.lock` → hash do dataset.

---

**6. Se dois cientistas treinaram o mesmo modelo com hiperparâmetros diferentes, como você garantiria que ambos os experimentos são comparáveis e reproduzíveis?**

Três condições precisam ser cumpridas:

1. **Mesmo dataset**: ambos devem usar o mesmo hash de dados via `dvc pull` — se os dados divergem, a comparação é inválida.
2. **Mesmo código de avaliação**: mesma divisão treino/teste (`random_state=42`, `test_size=0.2`) e mesmas métricas calculadas da mesma forma.
3. **Rastreamento centralizado**: ambos logam no mesmo MLflow Tracking Server com o mesmo `experiment_name`. Cada um cria seu próprio `run_id`, e a interface do MLflow permite comparar todas as runs lado a lado — hiperparâmetros, métricas e artefatos.

Neste projeto, isso é garantido porque `params.yaml` controla os hiperparâmetros (qualquer mudança fica no diff do Git), e `pipeline/train.py` aponta para `http://localhost:5000` como tracking server.

---

## Experimentação

**7. O que você loga num experimento além de métricas e por que isso importa?**

Logar apenas métricas é insuficiente — sem contexto, um ROC-AUC de 0.92 não diz nada acionável. O que deve ser logado:

- **Hiperparâmetros**: para reproduzir o treino e entender o que foi otimizado.
- **Artefatos**: o modelo serializado e o preprocessor — sem eles, a run não tem valor operacional.
- **Versão do dataset**: hash ou referência DVC — garante que se sabe exatamente com quais dados aquela performance foi obtida.
- **Threshold de decisão**: especialmente em classificação com classes desbalanceadas, o threshold muda completamente as métricas de negócio.
- **Tempo de treino**: relevante para decisões de custo e escala.
- **Metadados de ambiente**: versão do Python, das bibliotecas, tipo de hardware — afetam reprojetibilidade.

Neste projeto, o `model_trainer.py` loga hiperparâmetros, métricas, o modelo, e o `preprocessor.joblib` como artefato — permitindo que a API carregue tudo a partir de um único `run_id`.

---

**8. Como você decide qual modelo promover para produção a partir de múltiplos experimentos?**

A decisão deve considerar métricas técnicas e de negócio:

- **Para detecção de fraude**, ROC-AUC e PR-AUC são mais relevantes que accuracy, porque o dataset é desbalanceado — um modelo que classifica tudo como "não-fraude" teria ~92% de accuracy mas seria inútil.
- **PR-AUC** (área sob a curva Precisão-Recall) é especialmente importante quando a classe positiva (fraude) é rara.
- **Comparação sempre no mesmo conjunto de teste** — nunca no treino.
- **Validação de negócio**: às vezes o modelo com ROC-AUC levemente menor tem melhor recall em faixas de threshold operacionalmente relevantes.

Neste projeto, a promoção é automática: `model_trainer.py` compara os dois modelos por `roc_auc`, registra o melhor no MLflow Model Registry como `fraud_detection_champion` e atribui o alias `@champion`. A API sempre carrega `models:/fraud_detection_champion@champion`.

---

## Serving e APIs

**9. Por que expor um modelo via API REST em vez de entregá-lo como biblioteca ou script?**

- **Biblioteca**: acopla a linguagem e a versão de Python do cliente ao modelo. Se o modelo atualiza, todos os consumidores precisam atualizar a dependência.
- **Script**: não serve requisições em tempo real, não tem interface padronizada, não escala.
- **API REST**: agnóstica de linguagem (qualquer sistema que faz HTTP pode consumir), permite versionar o contrato de interface independente do modelo, escala horizontalmente, é monitorável com métricas de latência e disponibilidade, e isola o ambiente do modelo do ambiente do consumidor.

Em contexto de fraude, a API é chamada em tempo real pelo sistema de pagamentos no momento da transação — uma biblioteca ou script nunca atenderia esse requisito.

---

**10. Como você lidaria com erros de validação de entrada numa API de inferência em produção?**

Em três camadas:

1. **Validação automática via Pydantic** (implementado): o FastAPI rejeita automaticamente qualquer campo fora do tipo ou range definido, retornando HTTP 422 com detalhes do campo problemático. Campos categóricos usam `Literal` — o Swagger mostra um dropdown com as únicas opções aceitas.

2. **Logging estruturado**: todo erro de validação deve ser logado com o payload recebido (sem dados sensíveis), timestamp e identificador da requisição — para detectar padrões de uso incorreto.

3. **Resposta padronizada**: a mensagem de erro deve ser clara para o consumidor mas não expor detalhes internos do modelo ou da infraestrutura.

Em produção avançada, adicionaria monitoramento de taxa de erros 422 — um pico pode indicar que o sistema upstream mudou o contrato de dados.

---

**11. Por que containerizar uma aplicação de ML e o que o Docker resolve especificamente para esse contexto?**

O problema clássico em ML é: "funciona na minha máquina". Isso acontece porque ML tem dependências frágeis — versões de scikit-learn, xgboost, numpy frequentemente introduzem breaking changes e os resultados podem ser numericamente diferentes entre versões.

O Docker resolve isso criando um ambiente imutável e portátil: o `Dockerfile` define exatamente o Python, as bibliotecas e as configurações. Qualquer máquina que rodar aquela imagem terá o ambiente idêntico — desenvolvimento, staging, produção. Para ML isso é crítico porque afeta tanto a reprojetibilidade do modelo quanto a previsibilidade do comportamento em produção.

---

**12. Qual a diferença entre uma imagem e um container?**

- **Imagem**: template estático e imutável, criado a partir de um `Dockerfile` via `docker build`. É como uma classe — define a estrutura mas não executa nada.
- **Container**: instância em execução de uma imagem. É como um objeto instanciado da classe — tem estado, usa memória, ocupa porta, pode ser pausado ou destruído.

A mesma imagem pode gerar múltiplos containers simultâneos (escalabilidade horizontal). Neste projeto, `docker-compose up api --build` constrói a imagem e instancia um container chamado `fraud_api`.

---

**13. Qual o papel do docker-compose no projeto e o que ele orquestra?**

O `docker-compose.yml` define e orquestra dois serviços que precisam trabalhar juntos:

- **`mlflow`**: servidor de tracking e Model Registry, com backend file-based persistido em volume (`./mlruns`), exposto na porta 5000 com `--serve-artifacts` para que outros containers possam baixar artefatos via HTTP.
- **`api`**: a FastAPI, buildada a partir da raiz do projeto, configurada com `MLFLOW_TRACKING_URI=http://mlflow:5000` (usando o DNS interno do Docker) e dependente do serviço `mlflow`.

Sem o compose, seria necessário subir cada container manualmente, configurar a rede entre eles, gerenciar volumes e variáveis de ambiente separadamente. O compose faz tudo isso com um único `docker-compose up`.

---

**14. Como você lidaria com variáveis de configuração sensíveis num ambiente containerizado?**

Nunca hardcoded no código ou no `Dockerfile`. As abordagens, em ordem de maturidade:

1. **Arquivo `.env`** (desenvolvimento): define variáveis localmente, nunca versionado no Git (`.gitignore`). O compose e o Python (`python-dotenv`) carregam automaticamente.
2. **Variáveis de ambiente no compose** (desenvolvimento/staging): definidas em `environment:` no `docker-compose.yml` ou passadas via `-e` no `docker run`.
3. **Docker Secrets** (produção): mecanismo nativo do Docker Swarm/K8s para injetar segredos em tempo de execução sem exposição em variáveis de ambiente.
4. **Gerenciadores de segredo** (produção enterprise): HashiCorp Vault, AWS Secrets Manager, Azure Key Vault.

Neste projeto, `MLFLOW_TRACKING_URI` e `FRAUD_THRESHOLD` são passadas como variáveis de ambiente no compose — sensíveis como senhas de banco ou chaves de API seguiriam o mesmo padrão via `.env` (não versionado).

---

## CI/CD para ML

**15 / 18. O que significa CI/CD no contexto de ML?**

- **CI (Continuous Integration)**: a cada push de código, um pipeline automatizado executa testes, valida os dados, re-treina o modelo se necessário e verifica se a performance não regrediu em relação ao baseline. O objetivo é detectar problemas cedo.
- **CD (Continuous Delivery/Deployment)**: se o CI passa, o modelo validado é automaticamente empacotado, registrado no Model Registry e eventualmente promovido para produção — sem intervenção manual.

Em ML, CI/CD vai além do software tradicional porque inclui validação de dados (além de código), testes de performance de modelo (além de testes unitários) e promoção baseada em métricas (além de simples aprovação de build).

---

**16. Quais etapas você incluiria num pipeline de CI para um projeto de ML?**

1. **Linting e testes unitários**: verificar qualidade do código (ruff, pytest).
2. **Validação de dados**: checar schema, distribuição e ausência de corrupção no dataset (Great Expectations ou Pandera).
3. **Reprodução do pipeline**: `dvc repro` com os dados versionados.
4. **Avaliação do modelo**: verificar se as métricas do novo modelo atingem um mínimo aceitável (ex: ROC-AUC > 0.80).
5. **Comparação com baseline**: rejeitar automaticamente se o novo modelo for pior que o campeão atual.
6. **Build da imagem Docker**: garantir que a API builda sem erros.
7. **Teste de integração da API**: subir a API e fazer uma requisição ao `/predict` com um payload válido — verificar resposta 200 e schema correto.

---

**17. O que é continuous training e em que cenários ele faz sentido?**

Continuous training é o re-treino automático do modelo em produção quando novos dados chegam ou quando drift é detectado — sem intervenção manual. Faz sentido quando:

- O padrão do target muda rapidamente (ex: fraude financeira — novos golpes surgem constantemente).
- Há um fluxo contínuo de dados rotulados chegando.
- O custo de um modelo desatualizado é alto (perdas financeiras, falsos negativos custosos).

Não faz sentido em domínios estáveis (ex: reconhecimento de dígitos escritos à mão) onde os dados raramente mudam. Para este projeto de fraude, continuous training seria o próximo passo natural — re-treinar semanalmente com transações recentes rotuladas pelo time de operações.

---

**19. Como você testaria um modelo de ML num pipeline de CI? Que tipos de testes são relevantes?**

- **Testes de dados**: o dataset tem o schema esperado? Há valores nulos inesperados? A proporção de fraudes está dentro do esperado? (Pandera / Great Expectations)
- **Testes de transformação**: o preprocessor produz o número correto de features? Nenhuma coluna é dropada inesperadamente?
- **Testes de performance mínima**: ROC-AUC acima de um threshold aceitável no conjunto de teste fixo.
- **Teste de regressão**: o novo modelo é igual ou melhor que o modelo em produção atual — nunca pior.
- **Testes de contrato da API**: o endpoint `/predict` retorna o schema correto para inputs válidos e HTTP 422 para inputs inválidos.
- **Testes de latência**: a inferência acontece dentro de um SLA aceitável (ex: < 200ms para uma requisição).

---

**20. O que pode dar errado ao fazer deploy automático de um novo modelo sem testes de validação? Como mitigar?**

Riscos:
- **Regressão silenciosa**: o novo modelo tem métricas globais parecidas mas performa pior em subgrupos críticos (ex: fraudes de alto valor).
- **Data leakage não detectado**: modelo com performance artificialmente alta vai para produção e falha com dados reais.
- **Quebra de contrato**: o modelo novo espera features diferentes — a API começa a retornar erros ou predições sem sentido.
- **Instabilidade numérica**: em alguns ambientes, modelos com pesos em float32 produzem resultados levemente diferentes — sem testes, isso passa despercebido.

Mitigações:
- **Shadow mode**: rodar o novo modelo em paralelo com o atual, comparando outputs sem afetar o usuário.
- **Canary deployment**: expor o novo modelo para 5-10% do tráfego real antes do rollout completo.
- **Critérios automáticos de promoção**: só promover se ROC-AUC ≥ modelo atual E recall na faixa de threshold ≥ mínimo definido.
- **Rollback automático**: se as métricas em produção caírem abaixo do baseline nas primeiras horas, reverter automaticamente para a versão anterior.

---

## Monitoramento e Produção

**21. O que você monitora num modelo em produção? Diferencie monitoramento de infraestrutura de monitoramento de modelo.**

**Monitoramento de infraestrutura** (o sistema está de pé?):
- Latência das requisições (p50, p95, p99)
- Taxa de erros HTTP (4xx, 5xx)
- Uso de CPU e memória do container
- Disponibilidade do MLflow e da API

**Monitoramento de modelo** (o modelo ainda faz sentido?):
- **Distribuição das features de entrada**: detecta data drift — ex: `risco_ip` subindo sistematicamente pode indicar novo vetor de fraude ou problema no sistema de origem.
- **Distribuição das predições**: se a taxa de `fraude_detectada: true` dobrar do dia para a noite, algo mudou.
- **Performance com feedback**: quando labels reais chegam (transações confirmadas como fraude ou não), recalcular ROC-AUC e PR-AUC em produção.
- **Taxa de falsos positivos**: em fraude, bloquear transações legítimas gera custo operacional — monitorar a taxa de contestações.

A distinção é crítica: infraestrutura pode estar perfeita (API respondendo em 10ms) enquanto o modelo está completamente degradado.

---

**22. O que é um baseline de modelo e por que ele é indispensável na produção?**

Um baseline é o ponto de referência mínimo contra o qual qualquer modelo novo é comparado. Pode ser:
- Um modelo mais simples (regressão logística vs XGBoost)
- Uma regra heurística (ex: "bloquear toda transação > R$10.000 após meia-noite")
- O modelo atualmente em produção

Sem baseline, não há como saber se um novo modelo é uma melhoria real ou simplesmente diferente. É a diferença entre dizer "o modelo tem ROC-AUC 0.91" e "o modelo tem ROC-AUC 0.91 contra 0.87 do baseline atual — melhoria de 4 pontos percentuais". O segundo é acionável; o primeiro não.

Em produção, o baseline também serve como critério de rollback: se o novo modelo cair abaixo da performance do baseline, deve ser revertido.

---

## Decisões de Design e Trade-offs

**23. Como você garantiria que seu pipeline funciona tanto em desenvolvimento quanto em produção sem modificações manuais?**

Através de configuração por variáveis de ambiente, não por código:

- **`MLFLOW_TRACKING_URI`**: em desenvolvimento, aponta para `http://localhost:5000`; dentro do Docker Compose, aponta para `http://mlflow:5000` (DNS interno). O código nunca muda — só a variável de ambiente.
- **`FRAUD_THRESHOLD`**: ajustável por ambiente sem re-build da imagem.
- **Paths relativos**: `src/config.py` usa `Path(__file__).parent.parent` para calcular caminhos relativos à raiz do projeto — funciona em qualquer máquina.
- **Docker garante paridade de ambiente**: a mesma imagem rodada localmente e em produção elimina a classe de bugs "funciona no meu notebook".

A regra geral é: tudo que muda entre ambientes é configuração (variável de ambiente), não código.

---

**24. Quais são os trade-offs entre usar o MLflow self-hosted versus um serviço gerenciado como DagsHub?**

| Aspecto | MLflow Self-hosted | DagsHub (gerenciado) |
|---|---|---|
| **Controle** | Total — dados nunca saem da infraestrutura própria | Dados ficam em servidor terceiro |
| **Privacidade / Compliance** | Ideal para dados sensíveis (financeiros, saúde) | Requer análise de conformidade (LGPD) |
| **Custo** | Fixo (servidor próprio) — barato em escala | Por uso / por usuário — pode ser caro em escala |
| **Manutenção** | Alta — backup, updates, disponibilidade são responsabilidade própria | Zero — o provedor gerencia tudo |
| **Colaboração** | Requer VPN ou exposição de rede para times distribuídos | Interface web acessível de qualquer lugar |
| **Integração** | Flexível, mas configuração manual | Integrado com GitHub, DVC remoto, Hugging Face |
| **Setup inicial** | Horas (como neste projeto) | Minutos |

Para este projeto acadêmico, o self-hosted faz sentido: demonstra o conhecimento da infraestrutura. Em produção com dados financeiros reais, o self-hosted também seria preferível por compliance. Para um time de pesquisa sem dados sensíveis, DagsHub ou MLflow no Databricks acelerariam muito o desenvolvimento.
