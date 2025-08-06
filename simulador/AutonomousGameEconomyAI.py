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

# Custom object for Keras serialization if 'mse' is truly a custom function
# If mse is just a string 'mse', this is not needed.
# If you have a custom 'mse' function defined elsewhere, it needs this decorator.
# @tf.keras.saving.register_keras_serializable()
# def mse_custom(y_true, y_pred):
#     return tf.keras.losses.mean_squared_error(y_true, y_pred)

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
        self.epsilon = 0.01 # Definir epsilon baixo para pouca exploração (quase só explotação)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.min_rate = 0.05
        self.max_rate = 0.5
        self.learning_rate = 0.001
        
        # Initialize models to None
        self.prediction_model = None
        self.decision_model = None
        self.critic_model = None

        # Tenta carregar modelos existentes, caso contrário, constrói novos.
        # Para um simulador "não-treinável", é crucial que os modelos existam ou sejam construídos
        # com pesos padrão (se você não tiver .h5s).
        self.load_models() # Chamada para carregar ou construir modelos

        # Carrega configurações (histórico, pesos)
        self.load_config() 

    def load_models(self):
        """Tenta carregar modelos existentes ou constrói novos se não encontrados."""
        models_loaded_successfully = False
        try:
            self.prediction_model = tf.keras.models.load_model('prediction_model.h5')
            self.decision_model = tf.keras.models.load_model('decision_model.h5')
            self.critic_model = tf.keras.models.load_model('critic_model.h5')
            
            if isinstance(self.prediction_model, tf.keras.Model) and \
               isinstance(self.decision_model, tf.keras.Model) and \
               isinstance(self.critic_model, tf.keras.Model):
                print("Modelos carregados com sucesso!")
                models_loaded_successfully = True
            else:
                raise ValueError("Modelos carregados não são instâncias válidas do Keras.")
        except Exception as e:
            print(f"Não foi possível carregar modelos existentes ({e}). Construindo novos para a simulação.")
            # Se você não quer treinamento, esses modelos seriam "vazios" ou com pesos aleatórios
            # A menos que você tenha um conjunto inicial de modelos .h5 para o simulador.
            self.prediction_model = self._build_prediction_model()
            self.decision_model = self._build_decision_model()
            self.critic_model = self._build_critic_model()

        # Garante que os modelos são instâncias válidas após o carregamento/construção
        if not isinstance(self.prediction_model, tf.keras.Model): self.prediction_model = self._build_prediction_model()
        if not isinstance(self.decision_model, tf.keras.Model): self.decision_model = self._build_decision_model()
        if not isinstance(self.critic_model, tf.keras.Model): self.critic_model = self._build_critic_model()


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
            with open('scaler_params.json', 'r', encoding='utf-8') as f:
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
            'transaction_volume': 1000000.0, # Aumentado para lidar com valores maiores do simulador
            'nft_avg_price': 1000.0,
            'daily_active_users': 50000.0,  # Aumentado para lidar com valores maiores do simulador
            'new_players': 1000.0,
            'leaving_players': 1000.0,
            'dex_liquidity': 10000000.0, # Aumentado para lidar com valores maiores do simulador
            'token_price': 10.0,
            'token_generation_rate': 10000.0,
            'token_burn_rate': 10000.0,
            'circulating_assets': 10000000.0,
        }

        # 1. Adiciona as 10 métricas básicas (índices 0-9)
        for key in ['transaction_volume', 'nft_avg_price', 'daily_active_users', 
                    'new_players', 'leaving_players', 'dex_liquidity', 'token_price', 
                    'token_generation_rate', 'token_burn_rate', 'circulating_assets']:
            value = float(data.get(key, self.get_default_values().get(key, 0.0))) # Pega do data ou default
            state.append(value / divisors.get(key, 1.0)) # Normalize
        
        # 2. Adiciona community_sentiment (índice 10) - Mapeado para float e normalizado para 0-1
        sentiment_value = self._map_sentiment_to_float(data.get('community_sentiment', 'neutral'))
        state.append((sentiment_value + 1) / 2) # Normalize -1:1 to 0:1

        # 3. Calculando indicadores avançados (índices 11-15)
        # Garante que os valores brutos estão disponíveis para cálculos
        raw_dau = data.get('daily_active_users', self.get_default_values().get('daily_active_users', 0.0))
        raw_circulating_assets = data.get('circulating_assets', self.get_default_values().get('circulating_assets', 0.0))
        raw_transaction_volume = data.get('transaction_volume', self.get_default_values().get('transaction_volume', 0.0))
        raw_token_generation_rate = data.get('token_generation_rate', self.get_default_values().get('token_generation_rate', 0.0))
        raw_leaving_players = data.get('leaving_players', self.get_default_values().get('leaving_players', 0.0))
        raw_new_players = data.get('new_players', self.get_default_values().get('new_players', 0.0))
        raw_token_burn_rate = data.get('token_burn_rate', self.get_default_values().get('token_burn_rate', 0.0))
        raw_dex_liquidity = data.get('dex_liquidity', self.get_default_values().get('dex_liquidity', 0.0))

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
        # Para a IA de simulador que não treina, esses valores podem ser fixos ou baseados
        # em um histórico limitado. Manteremos a lógica existente, mas o 'replay' não será chamado.
        # recent_performance (média dos últimos scores)
        if len(self.performance_history) >= 10:
            scores = [p['performance_score'] for p in self.performance_history[-10:] if 'performance_score' in p]
            if scores:
                recent_performance = np.mean(scores)
                state.append(recent_performance)
            else:
                state.append(0.5) # Default if no valid scores
        else:
            state.append(0.5) # Default if not enough history

        # volatility_rates (desvio padrão das últimas taxas)
        if len(self.decision_history) >= 10:
            rates = [d['rate'] for d in self.decision_history[-10:] if 'rate' in d]
            volatility = np.std(rates) if len(rates) > 1 else 0.0
            state.append(volatility * 10.0) # Escala para range de feature
        else:
            state.append(0.0)

        # pressure_index
        pressure = (
            churn_rate * 0.3 +
            (1 - liquidity_ratio) * 0.2 +
            (1 - ((sentiment_value + 1) / 2)) * 0.2 + 
            abs(token_health) * 0.3
        )
        state.append(pressure)

        # 6. Métricas de auto-aprendizado (weights) (índices 20-24)
        factor_keys_to_include = [
            'economic_pressure', 'liquidity_ratio', 'market_sentiment', 
            'player_retention', 'token_health'
        ]
        for factor_key in factor_keys_to_include:
            state.append(self.factor_weights.get(factor_key, 1.0)) # Default to 1.0 if not found

        # Verificação final do tamanho do estado
        if len(state) != 25:
            print(f"Erro crítico: O estado final tem {len(state)} features, mas 25 são esperadas! Ajustando...")
            if len(state) < 25:
                state.extend([0.0] * (25 - len(state)))
            elif len(state) > 25:
                state = state[:25]
            print(f"Estado ajustado para {len(state)} features.")

        return np.array(state, dtype=np.float32)

    # REMOVER OU COMENTAR FUNÇÕES DE TREINAMENTO:
    # A IA para o simulador não deve treinar
    def calculate_reward(self, old_state, new_state, action):
        # Esta função é usada pelo replay, que será desabilitado.
        # Mas vamos mantê-la caso você queira usá-la para fins de cálculo de performance
        # ou se decidir reabilitar um treinamento manual no futuro.
        reward = 0
        # Seu cálculo de recompensa...
        old_price = old_state[6]
        new_price = new_state[6]
        price_stability = 1 - abs(new_price - old_price) / max(old_price, 0.01)
        reward += price_stability * 2
        
        old_churn = old_state[12]
        new_churn = new_state[12]
        if new_churn < old_churn: # Churn diminuiu (bom)
            reward += (old_churn - new_churn) * 5
        else: # Churn aumentou ou ficou igual (ruim)
            reward -= (new_churn - old_churn) * 3
        
        new_volume = new_state[0]
        optimal_volume = 0.5 # Volume normalizado ideal (ex: 50% do max possível)
        volume_diff = abs(new_volume - optimal_volume)
        reward += (1 - volume_diff) * 1.5
        
        sentiment = new_state[10]
        reward += sentiment * 2
        
        volatility = new_state[18]
        reward -= volatility * 3
        
        liquidity_ratio = new_state[15]
        reward += liquidity_ratio * 1.5
        
        return reward

    def remember(self, state, action, reward, next_state):
        # Esta função ainda pode ser útil para acumular um histórico de estados/decisões,
        # mas o replay que a usa para treinamento não será mais chamado.
        self.memory.append((np.array(state, dtype=np.float32), float(action), float(reward), 
                            np.array(next_state, dtype=np.float32) if next_state is not None else None))

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
        
        # Em um simulador, queremos quase sempre explotação (usar o que o modelo prevê)
        # Como self.epsilon já está baixo (0.01), a exploração será mínima.
        if np.random.random() < self.epsilon:
            action = np.random.uniform(0, 1) # Ação aleatória (exploração)
        else:
            state_input = state.reshape(1, -1) # Formata o estado para entrada do modelo
            try:
                action = self.decision_model.predict(state_input, verbose=0)[0][0] # Ação do modelo (explotação)
            except Exception as e:
                print(f"Erro na predição do decision_model: {e}. Usando ação padrão.")
                action = 0.5 # Fallback para ação padrão

        rate = self.min_rate + action * (self.max_rate - self.min_rate)
        # Suavização
        if len(self.decision_history) > 0:
            last_rate = self.decision_history[-1]['rate']
            max_change = 0.05 # Limita a mudança para suavizar
            rate = np.clip(rate, last_rate - max_change, last_rate + max_change)
        
        return float(rate), float(action) # Garante que os retornos são floats nativos

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
            basic_analysis['decision_factors'].append(f"高い税率選択 ({rate:.1%}) 通貨発行を抑制")
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

    def process_incoming_data(self, data_from_frontend, current_withdrawal_rate_frontend):
        """
        Recebe dados diretamente do frontend (Flask) e processa a decisão da IA.
        Não lê mais de arquivos .txt aqui, apenas usa os dados fornecidos.
        """
        # A IA não vai treinar, então epsilon é fixo baixo para exploração mínima.
        self.epsilon = self.epsilon_min 

        # Pega e normaliza o estado com base nos dados do frontend e na taxa atual
        # A taxa de saque atual agora virá do frontend (o valor que o app exibiu da última decisão)
        state = self.calculate_advanced_state(data_from_frontend, current_withdrawal_rate_frontend)

        # Toma decisão (a IA apenas "inferir" com base nos modelos carregados)
        rate, action = self.decide_action(state) 

        # Calcula métricas de desempenho (para exibição no frontend)
        performance_score = self._calculate_performance_score(data_from_frontend, rate)

        # Armazena decisão na performance_history e decision_history
        # É importante que o histórico seja mantido para cálculos como 'recent_performance'
        decision_data = {
            'timestamp': datetime.now().isoformat(),
            'rate': float(rate),
            'action': float(action), 
            'state': state.tolist(), # Armazena como lista para serialização JSON
            'data': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in data_from_frontend.items()},
            'performance_score': float(performance_score)
        }
        self.decision_history.append(decision_data)
        self.performance_history.append(decision_data)

        # Gera análise e predictions (usando os modelos para inferência)
        full_analysis = self._generate_analysis(data_from_frontend, state, rate)
        
        return {"rate": float(rate), "analysis": full_analysis, "state": state.tolist(), "action": float(action)}

    # A função `read_txt_data` NÃO será usada mais pelo Flask diretamente, 
    # pois os dados virão no POST request.
    # Você pode mantê-la como um método interno para outros propósitos
    # ou se precisar dela para inicialização, mas o app.py não a chamará.
    # Removi essa função para evitar confusão no fluxo de dados do simulador.
    # Se você ainda precisa dela para outros fins, pode adicionar de volta.

    # Desativar métodos de salvamento de log/modelos automáticos se não forem necessários para o simulador
    # _save_decision e _save_state_snapshot podem ser úteis para debugar o simulador
    # mas o save_models e _save_rewards não seriam chamados após cada simulação se não há treinamento.

    def _save_decision(self, rate, analysis):
        """Salva decisão em arquivos."""
        # Adicionado encoding para evitar erros em sistemas diferentes
        with open('current_withdrawal_rate.txt', 'w', encoding='utf-8') as f:
            f.write(f"{float(rate):.6f}")
        
        try:
            with open('ai_decisions_log.json', 'a', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'rate': float(rate),
                    'analysis': analysis
                }, ensure_ascii=False)
                f.write('\n') # Adiciona uma nova linha após cada JSON
        except Exception as e:
            print(f"Erro salvando decisão no relatório: {e}")

    def _save_rewards(self):
        """Salva recompensas históricas (memória de replay)."""
        # Para o simulador "não-treinável", este método não precisa ser chamado
        # a menos que você queira manter o log da 'memória' por algum motivo.
        if not self.memory: 
            print("Memória de recompensas vazia, nada para salvar neste momento.")
            return
        
        try:
            with open('ai_rewards_log.json', 'w', encoding='utf-8') as f:
                json.dump([
                    (s.tolist(), float(a), float(r), ns.tolist() if ns is not None else None) 
                    for s, a, r, ns in self.memory
                ], f, indent=4, ensure_ascii=False)
            print(f"Memória de recompensas salva com {len(self.memory)} entradas.")
        except Exception as e:
            print(f"Erro salvando recompensas: {e}")

    def _save_state_snapshot(self, data, state, rate, action):
        """Salva um snapshot completo do estado para análise posterior."""
        # Mantenha este para registrar os estados da simulação
        last_performance_score = 0.0
        last_rate_from_history = 0.0

        if self.performance_history:
            last_entry = self.performance_history[-1]
            last_performance_score = float(last_entry.get('performance_score', 0.0))
            last_rate_from_history = float(last_entry.get('rate', 0.0))

        historical_decisions_for_snapshot = []
        # Certifique-se de que estamos pegando apenas as informações serializáveis
        for d in self.decision_history[-10:]: # Pegar as 10 últimas decisões
            ts = d.get('timestamp')
            if isinstance(ts, datetime): # Se for um objeto datetime, converte para string
                ts = ts.isoformat()
            
            historical_decisions_for_snapshot.append({
                'timestamp': ts,
                'rate': float(d.get('rate', 0.0)),
                'performance_score': float(d.get('performance_score', 0.0))
            })

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'data': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in data.items()},
            'state': state.tolist(),
            'rate': float(rate),
            'action': float(action),
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
            with open('state_snapshots.json', 'a', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False)
                f.write('\n') # Adiciona uma nova linha após cada JSON
        except Exception as e:
            print(f"Erro salvando snapshot: {e}")

    # Desativar train_on_historical_data e replay para o simulador
    def replay(self, batch_size=32):
        print("Replay buffer desativado para o modo simulador.")
        return

    def train_on_historical_data(self):
        print("Treinamento de dados históricos desativado para o modo simulador.")
        return

    def save_models(self):
        """Salva modelos (mas para o simulador, não serão treinados novamente)."""
        # Apenas salva se eles existirem e forem válidos (como se fossem modelos pré-treinados)
        try:
            if isinstance(self.prediction_model, tf.keras.Model):
                self.prediction_model.save('prediction_model.h5')
            if isinstance(self.decision_model, tf.keras.Model):
                self.decision_model.save('decision_model.h5')
            if isinstance(self.critic_model, tf.keras.Model):
                self.critic_model.save('critic_model.h5')
            print("Modelos salvos (se existirem e forem válidos).")
        except Exception as e:
            print(f"Erro salvando modelos: {e}")
        
        ai_config = {
            'epsilon': self.epsilon,
            'factor_weights': self.factor_weights,
            'current_memory': [ # Converte para lista de listas para JSON, float para numpy
                [s.tolist(), float(a), float(r), ns.tolist() if ns is not None else None] 
                for s, a, r, ns in self.memory
            ],
            'decision_history': [ # Converte datetimes para strings
                {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()} 
                for d in self.decision_history
            ],
            'performance_history': [ # Converte datetimes para strings
                {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()} 
                for d in self.performance_history
            ],
            'default_values': self.get_default_values()
        }

        try:
            with open('ai_config.json', 'w', encoding='utf-8') as f:
                json.dump(ai_config, f, indent=4, ensure_ascii=False)
            print("My AI Config Stored Successfully!")
        except Exception as e:
            print(f"Erro salvando ai_config.json: {e}")

    def get_default_values(self):
        """Retorna valores padrão usados para inicialização e para preencher lacunas."""
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
            'community_sentiment': "neutral"
        }

    def load_config(self):
        """Carrega configurações salvas."""
        try:
            with open('ai_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.epsilon = config.get('epsilon', self.epsilon_min) # Definir como epsilon_min padrão
            self.factor_weights = config.get('factor_weights', self.factor_weights)
            
            # Recria a memória, convertendo de volta para np.array e float nativos
            self.memory = deque(maxlen=10000)
            for item in config.get('current_memory', []):
                s, a, r, ns = item
                action_val = float(a)
                reward_val = float(r)
                state_np = np.array(s, dtype=np.float32)
                next_state_np = np.array(ns, dtype=np.float32) if ns is not None else None
                self.memory.append((state_np, action_val, reward_val, next_state_np))
            
            # Recria histórico de decisões, convertendo timestamps de volta para datetime
            self.decision_history = []
            for entry in config.get('decision_history', []):
                new_entry = entry.copy()
                if 'timestamp' in new_entry and isinstance(new_entry['timestamp'], str):
                    try:
                        new_entry['timestamp'] = datetime.fromisoformat(new_entry['timestamp'])
                    except ValueError:
                        pass 
                if 'rate' in new_entry: new_entry['rate'] = float(new_entry['rate'])
                if 'action' in new_entry: new_entry['action'] = float(new_entry['action'])
                if 'performance_score' in new_entry: new_entry['performance_score'] = float(new_entry['performance_score'])
                if 'data' in new_entry and isinstance(new_entry['data'], dict):
                    for k, v in new_entry['data'].items():
                        if isinstance(v, (int, float)):
                            new_entry['data'][k] = float(v)
                self.decision_history.append(new_entry)
            
            self.performance_history = []
            for entry in config.get('performance_history', []):
                new_entry = entry.copy()
                if 'timestamp' in new_entry and isinstance(new_entry['timestamp'], str):
                    try:
                        new_entry['timestamp'] = datetime.fromisoformat(new_entry['timestamp'])
                    except ValueError:
                        pass 
                if 'rate' in new_entry: new_entry['rate'] = float(new_entry['rate'])
                if 'action' in new_entry: new_entry['action'] = float(new_entry['action'])
                if 'performance_score' in new_entry: new_entry['performance_score'] = float(new_entry['performance_score'])
                if 'data' in new_entry and isinstance(new_entry['data'], dict):
                    for k, v in new_entry['data'].items():
                        if isinstance(v, (int, float)):
                            new_entry['data'][k] = float(v)
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
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(str(default_val))
                    except Exception as e:
                        print(f"Erro ao criar arquivo padrão {filename}: {e}")
            if not os.path.exists('current_withdrawal_rate.txt'):
                with open('current_withdrawal_rate.txt', 'w', encoding='utf-8') as f:
                    f.write("0.20")
            return False
        except Exception as e:
            print(f"Load config error: {e}")
            return False

# Remova ou comente o bloco __main__ original para que a IA não execute nada por si só ao ser importada
# if __name__ == "__main__":
#     create_sample_files()
#     execute_phase1()
#     execute_phase2()
#     ai_performance_monitoring()
#     ai.save_models()

# Funções auxiliares (que antes estavam no __main__ ou globais)
# Elas não precisam ser parte da classe AI se só são chamadas externamente
def create_sample_files_if_not_exist():
    """Cria arquivos TXT de exemplo se não existirem."""
    # Usado para garantir que os arquivos .txt existam para a AI ler na inicialização do Flask
    # e para o frontend ter valores iniciais.
    basic_data = {
        'transaction_volume.txt': '50000.0', # Valores mais realistas para simulador
        'nft_avg_price.txt': '250.5',
        'daily_active_users.txt': '10000.0', # Mais players
        'new_players.txt': '800.0',
        'leaving_players.txt': '450.0',
        'dex_liquidity.txt': '500000.0', # Mais liquidez
        'token_price.txt': '1.00',
        'token_generation_rate.txt': '10000.0',
        'token_burn_rate': '8000.0',
        'circulating_assets.txt': '10000000.0',
        'community_sentiment.txt': 'neutral',
        'current_withdrawal_rate.txt': '0.20' # Taxa inicial
    }
    
    for filename, content in basic_data.items():
        try:
            if not os.path.exists(filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Sample file created: {filename}")
            else:
                # Apenas atualiza se o conteúdo for inválido, não sobrescreve sempre
                with open(filename, 'r', encoding='utf-8') as f:
                    current_content = f.read().strip()
                if filename == 'community_sentiment.txt':
                    if current_content not in ["positive", "neutral", "negative"]:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write("neutral")
                else:
                    try:
                        float(current_content)
                    except ValueError:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
        except Exception as e:
            print(f"Erro criando/verificando arquivo de exemplo {filename}: {e}")