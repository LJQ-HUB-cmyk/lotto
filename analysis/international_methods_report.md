# International Lottery Prediction Research Report

## Top Open-Source Repos
1. zepen/predict_Lottery_ticket (990 stars) - LSTM+CRF for SSQ/DLT
2. yangboz/LotteryPrediction (290 stars) - Transformer+Prophet for SSQ
3. KittenCN/predict_Lottery_ticket (258 stars) - Multi-layer LSTM SSQ/DLT/PL3
4. CorvusCodex/LotteryAi (144 stars) - Generic ML prediction
5. Ahmad-Alam/Lottery-Prediction (11 stars) - 4-layer LSTM Powerball/MegaMillions
6. michaelkupfer97/Lotto-prediction (3 stars) - RF+ARIMA+LSTM ensemble (German)
7. JeffMv/Lofea (40 stars) - Feature engineering library

## Loto-739-Pipeline (Most Sophisticated)
11 stages: Historical -> Frequency -> Decay -> Bayesian Fusion (Dirichlet+chi2) -> K-Means -> Monte Carlo -> Recency/Gap -> Markov (cluster-based) -> Entropy -> DL (classical+quantum fusion) -> Ticket Gen
Cloned to: /root/lotto/analysis/Loto-739-Pipeline/

## NEW Methods for Your System

### HIGH PRIORITY
1. CRF Decoding Layer - Replace independent softmax with CRF
2. Bayesian Fusion - Combine all 5+ strategies via log-space weighted fusion
3. Gap/Recency Analysis - Track draw absence patterns
4. Decay-Weighted Frequency - 0.98^weeks exponential decay

### MEDIUM PRIORITY
5. K-Means Cluster Modulation
6. Cluster-based Markov Chain (reduces state space)
7. ARIMA baseline per position

### EXPERIMENTAL
8. Transformer architecture
9. Quantum feature encoding (Qiskit)
10. Binary group classification

## Key Insights
- Multi-signal fusion > single models
- CRF underutilized for lottery sequence prediction
- Chinese SSQ/DLT repos dominate (990+ stars)
- DL alone barely beats random (AUC ~0.55 vs 0.50)
- Determinism and graceful degradation are key design principles
