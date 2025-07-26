
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import json
import os
from collections import deque
import random
import time

class AutonomousGameEconomyAI:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False
        self.memory = deque(maxlen=10000)
        self.performance_history = []
        self.decision_history = []
        self.factor_weights = {
            'economic_pressure': 1.0,
            'liquidity_ratio': 1.0,
            'market_sentiment': 1.0,
            'player_retention': 1.0,
            'token_health': 1.0,
            'transaction_volume': 1.0 # This one will be excluded from the 25-feature state vector to match 25.
        }
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.min_rate = 0.05
        self.max_rate = 0.5
        self.learning_rate = 0.001
        
        # Initialize models to None
        self.prediction_model = None
        self.decision_model = None
        self.critic_model = None

        # Try to load models. If any fails, rebuild all.
        models_loaded_successfully = False
        try:
            self.prediction_model = tf.keras.models.load_model('prediction_model.h5')
            self.decision_model = tf.keras.models.load_model('decision_model.h5')
            self.critic_model = tf.keras.models.load_model('critic_model.h5')
            # Verify if loaded models are actually Keras Models
            if isinstance(self.prediction_model, tf.keras.Model) and \
               isinstance(self.decision_model, tf.keras.Model) and \
               isinstance(self.critic_model, tf.keras.Model):
                print("Modelos carregados com sucesso!")
                models_loaded_successfully = True
            else:
                raise ValueError("Modelos carregados não são instâncias válidas do Keras.")
        except Exception as e:
            print(f"Não foi possível carregar modelos existentes ({e}). Construindo novos.")
            self.prediction_model = self._build_prediction_model()
            self.decision_model = self._build_decision_model()
            self.critic_model = self._build_critic_model()

        # Ensure models are not None after initialization/loading/building
        if not isinstance(self.prediction_model, tf.keras.Model): self.prediction_model = self._build_prediction_model()
        if not isinstance(self.decision_model, tf.keras.Model): self.decision_model = self._build_decision_model()
        if not isinstance(self.critic_model, tf.keras.Model): self.critic_model = self._build_critic_model()

        self.load_config() # Load config (including memory, history, epsilon, weights)

    def fit_scaler(self):
        """Treina o scaler com dados históricos (se necessário).
        No contexto atual, `calculate_advanced_state` já normaliza os dados,
        então este scaler pode ser para um propósito diferente ou legado.
        Vamos focar em carregar os parâmetros se existirem.
        """
        self.load_scaler_params() # Try to load existing scaler params

    def load_scaler_params(self):
        """Carrega parâmetros salvos do scaler"""
        try:
            with open('scaler_params.json', 'r') as f:
                params = json.load(f)
            self.scaler.min_ = np.array(params['min_'])
            self.scaler.scale_ = np.array(params['scale_'])
            self.scaler.data_min_ = np.array(params['data_min_'])
            self.scaler.data_max_ = np.array(params['data_max_'])
            self.scaler_fitted = True
            print("Parâmetros do scaler carregados")
        except Exception as e:
            print(f"Parâmetros do scaler não encontrados: {e}. O scaler não será usado para normalização.")
            self.scaler_fitted = False # Ensure it's false if loading fails

    def _build_prediction_model(self):
        """Modelo LSTM para prever tendências futuras.
        Atenção: Este modelo espera 17 features, enquanto os modelos de decisão/crítico esperam 25.
        Certifique-se de que a entrada para este modelo seja corretamente fatiada do estado completo.
        """
        inputs = Input(shape=(30, 17))  # 30 dias, 17 features

        lstm1 = LSTM(256, return_sequences=True)(inputs)
        dropout1 = Dropout(0.3)(lstm1)

        lstm2 = LSTM(128, return_sequences=True)(dropout1)

        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(lstm2, lstm2)

        lstm3 = LSTM(64, return_sequences=False)(attention_output)
        dropout2 = Dropout(0.2)(lstm3)

        dense1 = Dense(128, activation='relu')(dropout2)
        dense2 = Dense(64, activation='relu')(dense1)

        volume_pred = Dense(1, name='volume')(dense2)
        price_pred = Dense(1, name='price')(dense2)
        players_pred = Dense(1, name='players')(dense2)
        sentiment_pred = Dense(1, name='sentiment')(dense2)

        model = Model(
            inputs=inputs,
            outputs=[volume_pred, price_pred, players_pred, sentiment_pred]
        )

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            loss_weights=[1.0, 1.0, 1.0, 0.5]
        )

        return model

    def _build_decision_model(self):
        """Modelo de decisão com aprendizado por reforço. Espera 25 features."""
        state_input = Input(shape=(25,))  # Tamanho ajustado: 25 features

        x = Dense(512, activation='relu')(state_input)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        action = Dense(1, activation='sigmoid')(x)  

        model = Model(inputs=state_input, outputs=action)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return model

    def _build_critic_model(self):
        """Modelo crítico para avaliar decisões. Espera 25 features."""
        state_input = Input(shape=(25,))
        action_input = Input(shape=(1,))

        state_h1 = Dense(256, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)

        action_h1 = Dense(128)(action_input)

        concat = tf.keras.layers.Concatenate()([state_h2, action_h1])
        concat_h1 = Dense(256, activation='relu')(concat)
        concat_h2 = Dense(128, activation='relu')(concat_h1)

        q_value = Dense(1)(concat_h2)

        model = Model(inputs=[state_input, action_input], outputs=q_value)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return model

    def _map_sentiment_to_float(self, sentiment_str):
        """Mapeia strings de sentimento ('positive', 'neutral', 'negative') para valores numéricos (-1 a 1)."""
        sentiment_str_lower = str(sentiment_str).lower() # Ensure it's a string
        if "positive" in sentiment_str_lower:
            return 1.0
        elif "negative" in sentiment_str_lower:
            return -1.0
        else: # neutral or any other
            return 0.0

    def calculate_advanced_state(self, data, withdrawal_rate):
        """
        Calcula o vetor de estado avançado do jogo com exatamente 25 características.
        A `withdrawal_rate` é uma entrada crucial para a IA.
        """
        state = []
        
        # Definir divisores para normalização das 10 métricas básicas
        divisors = {
            'transaction_volume': 100000.0,
            'nft_avg_price': 1000.0,
            'daily_active_users': 10000.0,
            'new_players': 1000.0,
            'leaving_players': 1000.0,
            'dex_liquidity': 1000000.0,
            'token_price': 10.0,
            'token_generation_rate': 10000.0,
            'token_burn_rate': 10000.0,
            'circulating_assets': 10000000.0,
        }

        # 1. Adiciona as 10 métricas básicas (índices 0-9)
        for key in ['transaction_volume', 'nft_avg_price', 'daily_active_users', 
                    'new_players', 'leaving_players', 'dex_liquidity', 'token_price', 
                    'token_generation_rate', 'token_burn_rate', 'circulating_assets']:
            value = float(data.get(key, 0.0)) # Ensure float
            state.append(value / divisors.get(key, 1.0)) # Normalize
        
        # 2. Adiciona community_sentiment (índice 10) - Mapeado para float e normalizado para 0-1
        sentiment_value = self._map_sentiment_to_float(data.get('community_sentiment', 'neutral'))
        state.append((sentiment_value + 1) / 2) # Normalize -1:1 to 0:1

        # 3. Calculando indicadores avançados (índices 11-15)
        # Garante que os valores brutos estão disponíveis para cálculos
        raw_dau = data.get('daily_active_users', 0.0)
        raw_circulating_assets = data.get('circulating_assets', 0.0)
        raw_transaction_volume = data.get('transaction_volume', 0.0)
        raw_token_generation_rate = data.get('token_generation_rate', 0.0)
        raw_leaving_players = data.get('leaving_players', 0.0)
        raw_new_players = data.get('new_players', 0.0)
        raw_token_burn_rate = data.get('token_burn_rate', 0.0)
        raw_dex_liquidity = data.get('dex_liquidity', 0.0)

        # 11: Velocidade do mercado
        velocity = raw_transaction_volume / max(raw_circulating_assets, 1e-6)
        state.append(min(velocity / 10.0, 1.0)) # Normalize

        # 12: Taxa de churn
        churn_rate = raw_leaving_players / max(raw_dau, 1e-6)
        state.append(churn_rate) # Já é uma razão, geralmente entre 0 e 1

        # 13: Taxa de crescimento
        growth_rate = raw_new_players / max(raw_dau, 1e-6)
        state.append(growth_rate) # Já é uma razão, geralmente entre 0 e 1

        # 14: Saúde do token (deflacionário vs inflacionário)
        token_health = (raw_token_burn_rate - raw_token_generation_rate) / max(raw_token_generation_rate, 1e-6)
        state.append((token_health + 1) / 2) # Normaliza para 0-1

        # 15: Ratio de liquidez
        liquidity_ratio = raw_dex_liquidity / max(raw_transaction_volume, 1e-6)
        state.append(min(liquidity_ratio / 10.0, 1.0)) # Normaliza

        # 4. Adiciona current_withdrawal_rate (índice 16)
        state.append(float(withdrawal_rate)) # Já deve estar normalizado entre 0.05 e 0.50

        # 5. Métricas de aprendizado (índices 17-19)
        # 17: recent_performance
        if len(self.performance_history) >= 10:
            # Certifica-se de que 'performance_score' existe nos dicionários
            scores = [p['performance_score'] for p in self.performance_history[-10:] if 'performance_score' in p]
            if scores:
                recent_performance = np.mean(scores)
                state.append(recent_performance)
            else:
                state.append(0.5) # Default if no valid scores
        else:
            state.append(0.5) # Default if not enough history

        # 18: Volatilidade das taxas de saque
        if len(self.decision_history) >= 10:
            rates = [d['rate'] for d in self.decision_history[-10:] if 'rate' in d]
            volatility = np.std(rates) if len(rates) > 1 else 0.0
            state.append(volatility * 10.0) # Escala para range de feature
        else:
            state.append(0.0)

        # 19: Pressure index (índice de pressão)
        pressure = (
            churn_rate * 0.3 +
            (1 - liquidity_ratio) * 0.2 +
            (1 - ((sentiment_value + 1) / 2)) * 0.2 + 
            abs(token_health) * 0.3
        )
        state.append(pressure)

        # 6. Métricas de auto-aprendizado (weights) (índices 20-24)
        # Inclui 5 dos 6 factor_weights para totalizar 25 features.
        # 'transaction_volume' é excluído aqui para manter 25 features.
        factor_keys_to_include = [
            'economic_pressure', 'liquidity_ratio', 'market_sentiment', 
            'player_retention', 'token_health'
        ]
        for factor_key in factor_keys_to_include:
            state.append(self.factor_weights.get(factor_key, 1.0)) # Default to 1.0 if not found

        # Verificação final do tamanho do estado
        if len(state) != 25:
            print(f"Erro crítico: O estado final tem {len(state)} features, mas 25 são esperadas!")
            # Isso não deve acontecer com as correções acima, mas é um bom fallback.
            if len(state) < 25:
                state.extend([0.0] * (25 - len(state)))
            elif len(state) > 25:
                state = state[:25]

        return np.array(state, dtype=np.float32) # Garante que é um array numpy float32

    def read_txt_data(self):
        """Lê dados dos arquivos TXT com tratamento de erros."""
        data = {}
        default_values = {
            'transaction_volume': 10000.0,
            'nft_avg_price': 100.0,
            'daily_active_users': 1000.0,
            'new_players': 50.0,
            'leaving_players': 30.0,
            'dex_liquidity': 50000.0,
            'token_price': 1.0,
            'token_generation_rate': 1000.0,
            'token_burn_rate': 800.0,
            'circulating_assets': 1000000.0,
            'community_sentiment': "neutral" # O que bb.py escreve
        }
        
        for key, default in default_values.items():
            filename = f"{key}.txt"
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if key == 'community_sentiment':
                        data[key] = content # Armazena como string, mapeia para float em calculate_advanced_state
                    else:
                        data[key] = float(content)
            except (FileNotFoundError, ValueError):
                print(f"Arquivo '{filename}' não encontrado ou inválido. Usando valor padrão: {default}")
                # Cria o arquivo com o valor padrão para a próxima execução
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(str(default))
                data[key] = default # Usa o valor padrão para esta execução

        # Lê a taxa de saque atual do arquivo (fornecida externamente)
        try:
            with open('current_withdrawal_rate.txt', 'r', encoding='utf-8') as f:
                data['current_withdrawal_rate'] = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            print("Arquivo 'current_withdrawal_rate.txt' não encontrado ou inválido. Usando 0.20.")
            with open('current_withdrawal_rate.txt', 'w', encoding='utf-8') as f:
                f.write("0.20")
            data['current_withdrawal_rate'] = 0.20 # Usa o valor padrão para esta execução

        return data

    def _calculate_performance_score(self, data, rate):
        """Calcula score de performance da economia."""
        score = 0
        
        # Garante que os valores brutos estão disponíveis para cálculos
        raw_dex_liquidity = data.get('dex_liquidity', 0.0)
        raw_transaction_volume = data.get('transaction_volume', 0.0)
        raw_daily_active_users = data.get('daily_active_users', 0.0)
        raw_leaving_players = data.get('leaving_players', 0.0)
        raw_token_burn_rate = data.get('token_burn_rate', 0.0)
        raw_token_generation_rate = data.get('token_generation_rate', 0.0)
        
        # Líquidez (0-25 pontos)
        liquidity_ratio = raw_dex_liquidity / max(raw_transaction_volume, 1e-6)
        score += min(liquidity_ratio * 10, 25)
        
        # Medidores de player (0-25 pontos)
        if raw_daily_active_users > 0:
            retention_rate = 1 - (raw_leaving_players / raw_daily_active_users)
            score += retention_rate * 25
        
        # Sentimento comunitário (0-25 pontos) - Usa o valor numérico mapeado
        sentiment_value = self._map_sentiment_to_float(data.get('community_sentiment', 'neutral'))
        sentiment_ratio = (sentiment_value + 1) / 2 # Normaliza -1:1 para 0:1
        score += sentiment_ratio * 25
        
        # Eficiência de token (0-25 pontos)
        burn_efficiency = raw_token_burn_rate / max(raw_token_generation_rate, 1e-6)
        score += min(burn_efficiency * 25, 25)
        
        return score / 100.0 # Normaliza o score total para 0-1

    def calculate_reward(self, old_state, new_state, action):
        """Calcula recompensa baseada no impacto da decisão."""
        reward = 0
        
        # Estabilização de preço (Índice 6 - token_price normalizado)
        old_price = old_state[6]
        new_price = new_state[6]
        price_stability = 1 - abs(new_price - old_price) / max(old_price, 0.01)
        reward += price_stability * 2
        
        # Retenção de jogadores (Índice 12 - churn_rate)
        old_churn = old_state[12]
        new_churn = new_state[12]
        if new_churn < old_churn: # Churn diminuiu (bom)
            reward += (old_churn - new_churn) * 5
        else: # Churn aumentou ou ficou igual (ruim)
            reward -= (new_churn - old_churn) * 3
        
        # Volume saudável (Índice 0 - transaction_volume normalizado)
        new_volume = new_state[0]
        optimal_volume = 0.5 # Volume normalizado ideal (ex: 50% do max possível)
        volume_diff = abs(new_volume - optimal_volume)
        reward += (1 - volume_diff) * 1.5 # Recompensa maior quanto mais próximo do ideal
        
        # Sentimento positivo (Índice 10 - community_sentiment normalizado)
        sentiment = new_state[10]
        reward += sentiment * 2 # Recompensa maior para sentimento mais positivo
        
        # Penalidade de volatilidade (Índice 18 - volatility)
        volatility = new_state[18]
        reward -= volatility * 3 # Penaliza alta volatilidade
        
        # Liquidez saudável (Índice 15 - liquidity_ratio)
        liquidity_ratio = new_state[15]
        reward += liquidity_ratio * 1.5 # Recompensa maior para maior liquidez
        
        return reward

    def remember(self, state, action, reward, next_state):
        """Armazena experiência na memória de replay."""
        # Garante que os estados são arrays numpy antes de armazenar
        self.memory.append((np.array(state, dtype=np.float32), action, reward, 
                            np.array(next_state, dtype=np.float32) if next_state is not None else None))

    def remember_current(self, data, state, rate, action):
        """Armazena dados atuais na performance_history e decision_history."""
        decision_data = {
            'timestamp': datetime.now().isoformat(),
            'rate': rate,
            'action': action, # Salva a ação aqui
            'state': state.tolist(), # Armazena como lista para serialização JSON
            'data': data, # Armazena dados brutos também
            'performance_score': self._calculate_performance_score(data, rate)
        }
        self.decision_history.append(decision_data)
        self.performance_history.append(decision_data)

    def _generate_analysis(self, data, state, rate):
        """Gera análise detalhada usando predições reais do modelo LSTM."""
        basic_analysis = {
            'decision_factors': [],
            'risks': [],
            'opportunities': [],
            'predictions': {}
        }
        
        # Análise de fatores existente (usando índices do estado de 25 features)
        if state[12] > 0.1:  # Churn rate (índice 12)
            basic_analysis['decision_factors'].append(f"Alta taxa de saída de jogadores ({state[12]*100:.1f}%)")
            basic_analysis['risks'].append("Risco de êxodo em massa de jogadores")
        
        if state[15] < 0.5:  # Liquidity ratio (índice 15)
            basic_analysis['decision_factors'].append(f"Liquidez baixa (ratio: {state[15]:.2f})")
            basic_analysis['risks'].append("Risco de travamento do mercado")
        
        # Usa o valor numérico mapeado do sentimento (índice 10)
        if state[10] < 0.3: # Sentimento negativo (normalizado 0-1, então <0.3 é negativo)
            basic_analysis['decision_factors'].append("Sentimento negativo da comunidade")
            basic_analysis['risks'].append("Risco de dano à reputação")
        
        if state[14] > 0.7:  # Token health (índice 14) - assumindo >0.7 é saudável
            basic_analysis['opportunities'].append("Economia deflacionária saudável")
        
        # PREDIÇÕES USANDO O MODELO LSTM
        # O modelo de predição espera 17 características.
        # Precisamos extrair as primeiras 17 características do estado de 25 features.
        if len(self.decision_history) >= 30: # Precisa de 30 estados históricos para LSTM
            sequence = []
            for i in range(30):
                idx = -(30 - i) # Pega estados do histórico recente
                if len(self.decision_history) > abs(idx) and 'state' in self.decision_history[idx]:
                    # Garante que o estado tem pelo menos 17 features para o modelo de predição
                    sequence.append(self.decision_history[idx]['state'][:17]) 
                else:
                    # Se não houver histórico suficiente ou estado malformado, preenche com zeros
                    print(f"Aviso: Histórico de decisão insuficiente ou malformado para predição LSTM no índice {idx}. Preenchendo com zeros.")
                    sequence.append([0.0] * 17) # Preenche com zeros
            
            if len(sequence) == 30 and all(len(s) == 17 for s in sequence):
                try:
                    sequence = np.array(sequence, dtype=np.float32).reshape(1, 30, 17)
                    predictions = self.prediction_model.predict(sequence, verbose=0)
                    volume_pred, price_pred, players_pred, sentiment_pred = predictions
                                            
                    # Converte predições normalizadas de volta para valores reais para análise
                    # Compara com o estado atual normalizado
                    
                    volume_change = (volume_pred[0][0] - state[0]) * 100 # state[0] é volume normalizado
                    price_change = (price_pred[0][0] - state[6]) * 100 # state[6] é preço do token normalizado
                    players_change = (players_pred[0][0] - state[2]) * 100 # state[2] é DAU normalizado
                    
                    basic_analysis['predictions'] = {
                        'volume_24h': f"{'Aumento' if volume_change > 0 else 'Queda'} de {abs(volume_change):.1f}% no volume previsto",
                        'token_price': f"Preço do token: {'alta' if price_change > 0 else 'baixa'} de {abs(price_change):.1f}% esperada",
                        'player_retention': f"Base de jogadores: {'crescimento' if players_change > 0 else 'redução'} de {abs(players_change):.1f}% projetado",
                        'sentiment_trend': f"Sentimento: tendência {'positiva' if sentiment_pred[0][0] > 0.5 else 'negativa'} ({sentiment_pred[0][0]:.2f})",
                        'liquidity_impact': f"Líquidez: {'melhora' if (sentiment_pred[0][0] - state[10]) > 0 else 'deterioração'}" # Compara sentimento previsto com o atual (normalizado)
                    }
                    
                except Exception as e:
                    print(f"Erro na predição LSTM: {e}")
                    simple_pred = self._generate_simple_predictions(data, state)
                    basic_analysis['predictions'].update(simple_pred)
            else:
                basic_analysis['predictions'] = {
                    'volume_24h': "Dados insuficientes para predição (necessário almeno 30 dias de dados)",
                    'token_price': "Coletando dados históricos...",
                    'player_retention': "Prediçoes disponíveis após acumular dados suficientes"
                }
        else:
            basic_analysis['predictions'] = {
                'volume_24h': "Dados insuficientes para predição (necessário almeno 30 dias de dados)",
                'token_price': "Coletando dados históricos...",
                'player_retention': "Prediçoes disponíveis após acumular dados suficientes"
            }

        # Raciocínio da taxa. Agora que o estado tem 25 variáveis
        if rate > 0.3: 
            basic_analysis['decision_factors'].append(f"淀粉いたな高い税率選択 ({rate:.1%}) 通貨発行を抑制")
        elif rate < 0.15:
            basic_analysis['decision_factors'].append(f"選択された低税率 ({rate:.1%}) 市場活性化を促進")
        else:
            basic_analysis['decision_factors'].append(f"安定的な政策選択 ({rate:.1%}) 長期ビジョンに沿う")
        
        return basic_analysis

    def _generate_simple_predictions(self, data, state):
        """Predições simples baseadas em tendências atuais."""
        simple_analysis = {
            'liquidity_impact': "Análise de estabilidade do mercado requer histórico completo de dados"
        }
        
        # Base simples baseado compartilhamento de tokens
        # Ensure state has enough elements before accessing state[10]
        sentiment_seed = state[10] if isinstance(state, np.ndarray) and len(state) > 10 else 0.0
        simple_analysis['token_distribution'] = f"Compartilhamento de tokens {data.get('token_price', 0.0):.4f} (SEED={sentiment_seed:.2f})" 
        
        return simple_analysis

    def decide_action(self, state):
        """Decide a ação usando exploração vs exploração."""
        # Ensure decision_model is a valid Keras Model instance
        if not isinstance(self.decision_model, tf.keras.Model):
            print("Aviso: decision_model não é uma instância válida do Keras. Tentando reconstruir.")
            self.decision_model = self._build_decision_model()
            if not isinstance(self.decision_model, tf.keras.Model):
                raise RuntimeError("Falha ao construir decision_model. Não é possível decidir a ação.")

        # Verifica se o estado tem o tamanho correto antes de tomar decisão
        if state.shape[0] != 25:
            print(f"Estado tem tamanho errado! {state.shape[0]} ao invés de 25. Ajustando...")
            # Tenta ajustar o estado para 25 features
            if state.shape[0] < 25:
                # Preenche com zeros se for menor
                state = np.pad(state, (0, 25 - state.shape[0]), 'constant', constant_values=0.0)
            else:
                # Trunca se for maior
                state = state[:25]
            print(f"Estado ajustado para tamanho: {state.shape[0]}")
        
        if np.random.random() < self.epsilon:
            action = np.random.uniform(0, 1)
        else:
            state_input = state.reshape(1, -1)
            try:
                action = self.decision_model.predict(state_input, verbose=0)[0][0]
            except Exception as e:
                print(f"Erro na predição do decision_model: {e}. Usando ação padrão.")
                action = 0.5

        rate = self.min_rate + action * (self.max_rate - self.min_rate)
        # Suavização
        if len(self.decision_history) > 0:
            last_rate = self.decision_history[-1]['rate']
            max_change = 0.05
            rate = np.clip(rate, last_rate - max_change, last_rate + max_change)
        
        return rate, action

    def process_decision(self, data):
        """Processa a decisão e gera a análise completa."""
        try:
            # Pega e normaliza o estado
            withdrawal_rate = data.get('current_withdrawal_rate', 0.20)
            state = self.calculate_advanced_state(data, withdrawal_rate)

            # Toma decisão
            rate, action = self.decide_action(state)

            # Calcula métricas de desempenho
            performance_score = self._calculate_performance_score(data, rate)

            # Armazena decisão na performance_history e decision_history
            decision_data = {
                'timestamp': datetime.now().isoformat(),
                'rate': rate,
                'action': action, # Salva a ação aqui
                'state': state.tolist(), # Armazena como lista para serialização JSON
                'data': data, # Armazena dados brutos também
                'performance_score': performance_score
            }
            self.decision_history.append(decision_data)
            self.performance_history.append(decision_data)

            # Gera análise e predictions
            full_analysis = self._generate_analysis(data, state, rate)
            
            return {"rate": rate, "analysis": full_analysis, "state": state, "action": action}
        except Exception as e:
            print(f"Erro durante processamento: {e}")
            # Ensure state is a numpy array of zeros if an error occurs before it's fully calculated
            fallback_state = np.zeros(25, dtype=np.float32)
            # Retorna um valor padrão para 'action' também em caso de erro
            return {"rate": 0.3, "analysis": self._generate_simple_predictions(data, fallback_state), "state": fallback_state, "action": 0.5}

    def _save_decision(self, rate, analysis):
        """Salva decisão em arquivos."""
        # Armazena taxa atual
        with open('current_withdrawal_rate.txt', 'w') as f:
            f.write(f"{rate:.6f}")
        
        # Registra histórico de decisões
        try:
            with open('ai_decisions_log.json', 'a') as f:
                f.write(json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'rate': rate,
                    'analysis': analysis
                }) + '\n')
        except Exception as e:
            print(f"Erro salvando decisão no relatório: {e}")

    def _save_rewards(self):
        """Salva recompensas históricas (memória de replay) para análise de desempenho."""
        if not self.memory: 
            print("Memória de recompensas vazia, nada para salvar neste momento.")
            return
        
        try:
            # Salva a memória completa. Use 'w' para sobrescrever, 'a' para anexar.
            # Se a memória é um deque, ela já contém o histórico completo.
            # Sobrescrever é geralmente mais seguro para evitar duplicação em cada execução.
            with open('ai_rewards_log.json', 'w') as f:
                # Converte np.ndarray para lista para serialização JSON
                json.dump([
                    (s.tolist(), a, r, ns.tolist() if ns is not None else None) 
                    for s, a, r, ns in self.memory
                ], f, indent=4)
            print(f"Memória de recompensas salva com {len(self.memory)} entradas.")
        except Exception as e:
            print(f"Erro salvando recompensas: {e}")

    def _save_state_snapshot(self, data, state, rate, action): # Adicionado 'action' aqui
        """Salva um snapshot completo do estado para análise posterior e treinamento."""
        last_performance_score = 0.0
        last_rate_from_history = 0.0

        if self.performance_history:
            last_entry = self.performance_history[-1]
            # Use .get() with a default value to prevent KeyError if 'performance_score' is missing
            last_performance_score = last_entry.get('performance_score', 0.0)
            last_rate_from_history = last_entry.get('rate', 0.0) # Use .get() for safety

        # Ensure timestamps in historical_data are strings for JSON serialization
        historical_decisions_for_snapshot = []
        for d in self.decision_history[-10:]:
            if 'timestamp' in d and 'rate' in d and 'performance_score' in d:
                # Ensure timestamp is string, convert if it's datetime object
                ts = d['timestamp']
                if isinstance(ts, datetime):
                    ts = ts.isoformat()
                historical_decisions_for_snapshot.append({
                    'timestamp': ts,
                    'rate': d['rate'],
                    'performance_score': d['performance_score']
                })

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'data': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in data.items()},
            'state': state.tolist(), # Salva o vetor de estado de 25 features
            'rate': rate,
            'action': action, # Salva a ação aqui
            'weights': self.factor_weights,
            'performance': {
                'last_score': last_performance_score,
                'last_rate': last_rate_from_history
            },
            'historical_data': {
                'last_10_decisions': historical_decisions_for_snapshot
            }
        }
        
        try:
            with open('state_snapshots.json', 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
        except Exception as e:
            print(f"Erro salvando snapshot: {e}")

    def replay(self, batch_size=32):
        """Treina os modelos com experiências passadas (Replay Buffer)."""
        if len(self.memory) < batch_size:
            print(f"Memória insuficiente para replay. Necessário {batch_size}, disponível {len(self.memory)}.")
            return
            
        print(f"Iniciando replay training com batch_size={batch_size}...")
        for _ in range(10): # Treina múltiplas vezes em batches aleatórios
            batch = random.sample(self.memory, batch_size)
            
            for state, action, reward, next_state in batch:
                try:
                    # Garante que os estados são arrays numpy float32
                    state_np = np.array(state, dtype=np.float32)
                    next_state_np = np.array(next_state, dtype=np.float32) if next_state is not None else None

                    # Calcula valor Q futuro
                    if next_state_np is not None:
                        next_action = self.decision_model.predict(next_state_np.reshape(1, -1), verbose=0)[0][0]
                        next_q = self.critic_model.predict(
                            [next_state_np.reshape(1, -1), np.array([[next_action]], dtype=np.float32)], 
                            verbose=0
                        )[0][0]
                        target = reward + 0.95 * next_q  # Γ = 0.95
                    else:
                        target = reward
                        
                    # Treina critic model
                    self.critic_model.fit(
                        [state_np.reshape(1, -1), np.array([[action]], dtype=np.float32)], 
                        np.array([[target]], dtype=np.float32), 
                        epochs=1, verbose=0, batch_size=1
                    )
                    
                    # Treina decision model
                    self.decision_model.fit(
                        state_np.reshape(1, -1), 
                        np.array([[action]], dtype=np.float32), 
                        epochs=1, verbose=0, batch_size=1
                    )
                    
                except Exception as e:
                    print(f"Erro durante o replay de uma amostra: {e}")
                    continue 

        # Ajusta epsilon regularmente
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon ajustado para: {self.epsilon:.4f}")

    def train_on_historical_data(self):
        """Treina a IA com dados históricos (state_snapshots.json) para popular a memória de replay."""
        historical_data = []
        if os.path.exists('state_snapshots.json'):
            try:
                with open('state_snapshots.json', 'r') as f:
                    for line in f:
                        try:
                            snapshot = json.loads(line)
                            historical_data.append(snapshot)
                        except json.JSONDecodeError as e:
                            print(f"Erro ao decodificar JSON em state_snapshots.json: {e}. Linha: {line.strip()}")
                            continue
            except Exception as e:
                print(f"Erro ao ler state_snapshots.json: {e}")
                
        if len(historical_data) < 2:
            print("Dados históricos insuficientes para treinamento (necessário pelo menos 2 snapshots).")
            return

        print(f"Carregando {len(historical_data)} snapshots históricos para treinamento...")
        
        # Limpa a memória antes de popular a partir de dados históricos para evitar duplicações
        self.memory.clear()

        for i in range(len(historical_data) - 1):
            try:
                current_snapshot = historical_data[i]
                next_snapshot = historical_data[i + 1]
                
                # Garante que os estados são arrays numpy float32
                current_state = np.array(current_snapshot['state'], dtype=np.float32)
                next_state = np.array(next_snapshot['state'], dtype=np.float32)
                
                # A ação que levou do current_state para o next_state é a taxa de saque do current_snapshot
                # Agora 'action' é explicitamente salvo no snapshot
                action = current_snapshot.get('action', 0.5) # Use .get() for robustness
                if not isinstance(action, (float, int, np.floating)): # Ensure action is numeric
                    print(f"Aviso: Ação '{action}' no snapshot não é numérica. Usando 0.5.")
                    action = 0.5

                # Calcula recompensa baseado na transição
                reward = self.calculate_reward(current_state, next_state, action)
                
                # Armazena na memória de replay
                self.remember(current_state, action, reward, next_state)
                
            except Exception as e:
                print(f"Erro processando histórico para replay: {e}")
                continue

        # Realiza o treinamento de replay após popular a memória
        if len(self.memory) > 0:
            print(f"Iniciando replay training com {len(self.memory)} experiências...")
            self.replay()
            print("Treinamento com dados históricos concluído!")
        else:
            print("Nenhuma experiência válida encontrada no histórico para treinamento.")

    def save_models(self):
        """Salva modelos treinados e o estado de aprendizado."""
        try:
            self.prediction_model.save('prediction_model.h5')
            self.decision_model.save('decision_model.h5')
            self.critic_model.save('critic_model.h5')
            print("Modelos salvos com sucesso!")
        except Exception as e:
            print(f"Erro salvando modelos: {e}")
        
        # Salva configurações 
        ai_config = {
            'epsilon': self.epsilon,
            'factor_weights': self.factor_weights,
            # Converte deque para lista para serialização JSON
            'current_memory': list(self.memory), 
            # Converte objetos datetime para strings ISO format
            'decision_history': [
                {k: v.isoformat() if isinstance(v, datetime) else v for k, v in d.items()} 
                for d in self.decision_history
            ],
            'performance_history': [
                {k: v.isoformat() if isinstance(v, datetime) else v for k, v in d.items()} 
                for d in self.performance_history
            ],
            'default_values': self.get_default_values()
        }

        try:
            with open('ai_config.json', 'w') as f:
                json.dump(ai_config, f, default=str, indent=4) 
            print("My AI Config Stored Successfully!")
        except Exception as e:
            print(f"Erro salvando ai_config.json: {e}")

    def get_default_values(self):
        """Retorna valores padrão usados para inicialização."""
        return {
            'transaction_volume': 10000.0,
            'nft_avg_price': 100.0,
            'daily_active_users': 1000.0,
            'new_players': 50.0,
            'leaving_players': 30.0,
            'dex_liquidity': 50000.0,
            'token_price': 1.0,
            'token_generation_rate': 1000.0,
            'token_burn_rate': 800.0,
            'circulating_assets': 1000000.0,
            'community_sentiment': "neutral" # O que bb.py escreve
        }

    def load_config(self):
        """Carrega configurações salvas."""
        try:
            with open('ai_config.json', 'r') as f:
                config = json.load(f)
            
            self.epsilon = config.get('epsilon', 1.0)
            self.factor_weights = config.get('factor_weights', self.factor_weights)
            
            # Recria a memória, convertendo de volta para np.array
            self.memory = deque(maxlen=10000)
            for s, a, r, ns in config.get('current_memory', []):
                # Ensure 'a' (action) is a float before storing
                action_val = float(a) if isinstance(a, (int, float, np.floating)) else 0.5 # Default if not numeric
                self.memory.append((np.array(s, dtype=np.float32), action_val, r, np.array(ns, dtype=np.float32) if ns is not None else None))
            
            # Recria histórico de decisões, convertendo timestamps de volta para datetime
            self.decision_history = []
            for entry in config.get('decision_history', []):
                new_entry = entry.copy()
                if 'timestamp' in new_entry and isinstance(new_entry['timestamp'], str):
                    try:
                        new_entry['timestamp'] = datetime.fromisoformat(new_entry['timestamp'])
                    except ValueError:
                        pass # Mantém como string se a análise falhar
                self.decision_history.append(new_entry)
            
            self.performance_history = []
            for entry in config.get('performance_history', []):
                new_entry = entry.copy()
                if 'timestamp' in new_entry and isinstance(new_entry['timestamp'], str):
                    try:
                        new_entry['timestamp'] = datetime.fromisoformat(new_entry['timestamp'])
                    except ValueError:
                        pass # Mantém como string se a análise falhar
                self.performance_history.append(new_entry)
            
            print("Configurações da IA carregadas com sucesso!")
            return True
        except FileNotFoundError:
            print("Arquivos de configuração não existem. Iniciando de zero.")
            # Garante que os arquivos padrão existam se a configuração não for encontrada
            default_values = self.get_default_values()
            for key, default_val in default_values.items():
                filename = f"{key}.txt"
                if not os.path.exists(filename):
                    try:
                        with open(filename, 'w') as f:
                            f.write(str(default_val))
                    except Exception as e:
                        print(f"Erro ao criar arquivo padrão {filename}: {e}")
            # Garante que current_withdrawal_rate.txt exista
            if not os.path.exists('current_withdrawal_rate.txt'):
                with open('current_withdrawal_rate.txt', 'w') as f:
                    f.write("0.20")
            return False
        except Exception as e:
            print(f"Load config error: {e}")
            return False

def create_sample_files():
    """Cria arquivos TXT de exemplo para teste."""
    basic_data = {
        'transaction_volume.txt': '15000.0',
        'nft_avg_price.txt': '250.5',
        'daily_active_users.txt': '1200.0',
        'new_players.txt': '80.0',
        'leaving_players.txt': '45.0',
        'dex_liquidity.txt': '50000.0',
        'token_price.txt': '0.85',
        'token_generation_rate.txt': '1000.0',
        'token_burn_rate.txt': '800.0',
        'circulating_assets.txt': '1000000.0',
        'community_sentiment.txt': 'neutral', # Consistente com a saída do bb.py
        'current_withdrawal_rate.txt': '0.25' # Garante que este arquivo exista
    }
    
    for filename, content in basic_data.items():
        try:
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write(content)
                print(f"Sample file created: {filename}")
            else:
                # Se o arquivo já existe, apenas verifica se o conteúdo é float ou string
                # e tenta garantir que seja válido para evitar erros na leitura
                with open(filename, 'r', encoding='utf-8') as f:
                    current_content = f.read().strip()
                if filename == 'community_sentiment.txt':
                    if current_content not in ["positive", "neutral", "negative"]:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write("neutral") # Default to neutral if invalid
                else:
                    try:
                        float(current_content)
                    except ValueError:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content) # Overwrite with default if invalid float
        except Exception as e:
            print(f"Erro criando/verificando arquivo de exemplo {filename}: {e}")

# Instância global da IA (necessária para as funções de monitoramento e atualização de pesos)
ai = AutonomousGameEconomyAI()

def execute_phase1():
    """Executa a fase 1 da análise: lê o estado, decide a taxa e salva."""
    data = ai.read_txt_data()
    result = ai.process_decision(data)
    rate, analysis, current_state_vector, action = result['rate'], result['analysis'], result['state'], result['action'] # Get action here
    print(f"\nwithdrawal rate chosen: {rate:.3%}")
    print(json.dumps(analysis, indent=2, ensure_ascii=False)) # ensure_ascii=False para exibir japonês corretamente
    
    ai._save_decision(rate, analysis)
    ai._save_state_snapshot(data, current_state_vector, rate, action) # Pass action to snapshot

def execute_phase2():
    """Fase de aprimoramento: treina a IA usando o histórico de estados."""
    ai.train_on_historical_data() # Popula self.memory e executa replay
    ai._save_rewards() # Salva a memória atualizada em arquivo

def load_reward_history():
    """Carrega histórico de recompensas para análise (agora lê a memória completa)."""
    try:
        if os.path.exists('ai_rewards_log.json'):
            with open('ai_rewards_log.json', 'r') as f:
                return json.load(f) 
        return []
    except Exception as e:
        print(f"Erro carregando histórico de rewards: {e}")
        return []

def update_weights(reward_data):
    """Atualiza pesos dos fatores (chamada para adapt_weights)."""
    print("Atualizando pesos dos fatores (se houver dados suficientes)...")
    ai.adapt_weights(ai.performance_history) # Pass performance history to it

def get_metric_correlation(rewards, metric):
    """Calcula correlação entre métricas e recompensas da IA."""
    if not hasattr(ai, 'performance_history') or not ai.performance_history:
        return None
        
    metrics = []
    r_values = []
    
    for idx, entry in enumerate(rewards): # entry é (state, action, reward, next_state)
        if len(ai.performance_history) > idx and 'data' in ai.performance_history[idx]:
            metric_value = ai.performance_history[idx]['data'].get(metric, 0)
            
            if metric == 'community_sentiment':
                metric_value = ai._map_sentiment_to_float(metric_value)

            metrics.append(metric_value)
            r_values.append(entry[2]) # Reward é o 3º elemento no tuple (índice 2)
            
    if len(metrics) < 3 or np.std(metrics) == 0 or np.std(r_values) == 0:
        return None
        
    metrics = np.array(metrics)
    r_values = np.array(r_values)
    
    metrics = (metrics - np.mean(metrics)) / (np.std(metrics) + 1e-8) # Adicionado 1e-8 para evitar divisão por zero
    r_values = (r_values - np.mean(r_values)) / (np.std(r_values) + 1e-8) # Adicionado 1e-8 para evitar divisão por zero
    
    try:
        correlation = np.corrcoef(metrics, r_values)[0, 1]
        return correlation
    except Exception as e:
        print(f"Erro calculando correlação para {metric}: {e}")
        return None

def adjust_weight_based_on_correlation(correlation):
    """Ajusta pesos com base em correlações descobertas."""
    try:
        if correlation is None or np.isnan(correlation): # Handle NaN from correlation
            return None
        if correlation > 0.5:
            return min(2.0, correlation * 2)
        elif correlation < -0.5:
            return max(0.1, abs(correlation) * 2) 
        else:
            return None
    except Exception as e:
        print(f"Erro ajustando peso baseado em correlação: {e}")
        return None

def ai_performance_monitoring():
    """Monitora desempenho da IA com base histórica."""
    if not os.path.exists('ai_decisions_log.json'):
        print("Monitoramento de desempenho necessita de histórico de decisões.")
        return

    try:
        with open('ai_decisions_log.json', 'r') as f:
            decisions = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        decisions = []
        print(f"Erro ao carregar histórico de decisões: {e}. Iniciando análise limpa.")
    
    current_time = datetime.now().strftime('%Y-%m-%d')
    print(f"Monitoria de desempenho IA iniciado: {current_time}")

    if len(decisions) >= 10:
        recent_decisions = decisions[-10:]
        
        rate_changes = [d['rate'] for d in recent_decisions if 'rate' in d]
        if len(rate_changes) < 2: # Need at least 2 rates to check consistency
            print("Não há dados de taxa suficientes para verificar consistência.")
            update_trust_confidence(50)
            return

        consistent_count = 0
        for i in range(1, len(rate_changes)):
            if abs(rate_changes[i] - rate_changes[i-1]) < 0.05:
                consistent_count += 1
        
        if consistent_count < len(rate_changes) - 1:
            print(f"A pegada atual indica instabilidades nas últimas {len(rate_changes) - 1 - consistent_count} decisões.")
            create_stability_report(recent_decisions)
        else:
            print(f"Taxas de saque são compatíveis para últimas {consistent_count} decisões.")
            update_trust_confidence(95)
    else:
        print(f"Estatísticas necessitam de no mínimo 10 decisões para IA robusta.")
        update_trust_confidence(50)

def update_trust_confidence(confidence_level):
    """Atualiza confiança pública na IA."""
    trust_data = { 
        'timestamp': datetime.now().isoformat(),
        'current_confidence': confidence_level
    }
    try:
        with open('trust.log', 'a') as f:
            f.write(json.dumps(trust_data) + '\n')
        print(f"IA está {confidence_level}% operação autônoma")
    except Exception as e:
        print(f"Erro atualizando confiança: {e}")

def create_stability_report(decisions):
    """Cria relatórios de estabilidade acíclicos."""
    stability_data = {
        'filename': "ai_stability_issues.log",
        'header': ["Relatório de Estabilidade:",
                   "Valores acíclicos", 
                   "Habilitando...", "latest uncertain decisions?",
                   "Proposed changes for long-term vision improvement:"
        ],
        'decisions': decisions[-5:]
    }
    
    try:
        with open(stability_data['filename'], 'a') as f:
            f.write(json.dumps(stability_data, indent=2, ensure_ascii=False) + '\n')
        print(f"Relatório de problemas de instabilidade adicionado.")
    except Exception as e:
        print(f"Erro ao criar relatório de estabilidade: {e}")

if __name__ == "__main__":
    create_sample_files() # Garante que todos os arquivos necessários existam antes de iniciar

    # Execute a primeira fase da IA (lê, decide, salva a taxa e o snapshot do estado)
    execute_phase1()
    
    # Execute a fase de aprimoramento (treinamento da IA com o histórico de estados)
    execute_phase2()
    
    # Monitora e melhora a confiança da IA
    ai_performance_monitoring()

    # Salva modelos e configuração após todas as operações
    ai.save_models()
