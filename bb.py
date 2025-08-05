import json
import os
import requests
import numpy as np
from collections import deque
import random
import math
from datetime import datetime, timedelta
import copy

class PerformanceTracker:
    """Rastreia e avalia a performance das decisões da IA"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.performance_file = 'ai_performance.json'
        self.load_performance_data()
    
    def load_performance_data(self):
        """Carrega dados de performance salvos"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics_history = deque(data.get('metrics_history', []), maxlen=100)
            except Exception as e:
                print(f"Erro ao carregar performance: {e}")
    
    def save_performance_data(self):
        """Salva dados de performance"""
        try:
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics_history': list(self.metrics_history),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar performance: {e}")
    
    def calculate_economic_health_score(self, state):
        """Calcula um score de saúde econômica (0-100)"""
        score = 50  # Base neutra
        
        # Fatores positivos
        if state['daily_active_users'] > 100: score += 15
        elif state['daily_active_users'] > 50: score += 10
        elif state['daily_active_users'] > 10: score += 5
        
        if state['token_price'] > 0.5: score += 15
        elif state['token_price'] > 0.2: score += 10
        elif state['token_price'] > 0.1: score += 5
        
        if state['new_players'] > state['leaving_players']: score += 10
        
        if state['community_sentiment'] == 'positive': score += 15
        elif state['community_sentiment'] == 'neutral': score += 5
        
        # Fatores negativos
        if state['daily_active_users'] < 5: score -= 20
        if state['token_price'] < 0.05: score -= 25
        if state['leaving_players'] > state['new_players'] * 2: score -= 15
        if state['community_sentiment'] == 'negative': score -= 10
        
        return max(0, min(100, score))
    
    def evaluate_decision(self, previous_state, current_state, withdrawal_rate):
        """Avalia a qualidade da decisão tomada"""
        prev_score = self.calculate_economic_health_score(previous_state)
        curr_score = self.calculate_economic_health_score(current_state)
        
        improvement = curr_score - prev_score
        
        # Penaliza decisões que levam à estagnação
        stagnation_penalty = self.calculate_stagnation_penalty(current_state)
        
        # Recompensa baseada na melhoria da economia
        reward = improvement - stagnation_penalty
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'prev_score': prev_score,
            'curr_score': curr_score,
            'improvement': improvement,
            'stagnation_penalty': stagnation_penalty,
            'final_reward': reward,
            'withdrawal_rate': withdrawal_rate,
            'key_metrics': {
                'dau': current_state['daily_active_users'],
                'token_price': current_state['token_price'],
                'sentiment': current_state['community_sentiment']
            }
        }
        
        self.metrics_history.append(metric)
        self.save_performance_data()
        
        return reward, metric
    
    def calculate_stagnation_penalty(self, current_state):
        """Calcula penalidade por estagnação"""
        if len(self.metrics_history) < 3:
            return 0
        
        recent_states = [m['key_metrics'] for m in list(self.metrics_history)[-3:]]
        
        # Verifica variação nos últimos 3 estados
        dau_variance = np.var([s['dau'] for s in recent_states])
        price_variance = np.var([s['token_price'] for s in recent_states])
        
        # Se variação muito baixa, aplica penalidade
        penalty = 0
        if dau_variance < 0.1: penalty += 5
        if price_variance < 0.0001: penalty += 5
        
        # Penalidade extra se todos os sentimentos são iguais
        sentiments = [s['sentiment'] for s in recent_states]
        if len(set(sentiments)) == 1: penalty += 3
        
        return penalty
    
    def get_performance_summary(self):
        """Retorna resumo da performance"""
        if not self.metrics_history:
            return "Sem dados de performance disponíveis"
        
        recent_rewards = [m['final_reward'] for m in list(self.metrics_history)[-10:]]
        avg_reward = np.mean(recent_rewards)
        
        return {
            'avg_recent_reward': avg_reward,
            'total_decisions': len(self.metrics_history),
            'trend': 'improving' if avg_reward > 0 else 'declining' if avg_reward < -2 else 'stable'
        }

class AdvancedPromptGenerator:
    """Gera prompts adaptativos baseados no contexto e performance"""
    
    def __init__(self, performance_tracker):
        self.performance_tracker = performance_tracker
        self.prompt_templates = self.load_prompt_templates()
    
    def load_prompt_templates(self):
        """Templates de prompt para diferentes situações"""
        return {
            'crisis': """
SITUAÇÃO CRÍTICA DETECTADA! 
A economia está em colapso. Você precisa tomar decisões DRÁSTICAS para reverter:
- DAU muito baixo: {daily_active_users}
- Preço do token crítico: ${token_price}
- Jogadores saindo em massa: {leaving_players}

SEJA AGRESSIVO nas mudanças. Pequenos ajustes NÃO funcionarão!
""",
            'stagnation': """
ALERTA: ESTAGNAÇÃO ECONÔMICA DETECTADA!
Os últimos {stagnation_cycles} ciclos mostraram mudanças mínimas.
Você DEVE implementar mudanças significativas para quebrar este padrão.

Performance recente: {recent_performance}

FORÇA uma variação de pelo menos 10% em métricas chave!
""",
            'growth': """
ECONOMIA EM CRESCIMENTO - MANTENHA O MOMENTUM!
Tendência positiva detectada. Continue as estratégias que estão funcionando,
mas ajuste para sustentabilidade a longo prazo.

Não seja conservador demais - capitalize no crescimento!
""",
            'volatile': """
ECONOMIA INSTÁVEL - ESTABILIZE!
Detectada alta volatilidade. Priorize estabilização:
- Reduza flutuações bruscas
- Foque em métricas de longo prazo
- Mantenha crescimento sustentável
"""
        }
    
    def analyze_economic_context(self, current_data, history):
        """Analisa o contexto econômico atual"""
        context = {
            'situation': 'normal',
            'urgency': 'low',
            'stagnation_cycles': 0,
            'volatility': 'low'
        }
        
        # Detecta crise
        if (current_data['daily_active_users'] < 10 or 
            current_data['token_price'] < 0.1 or
            current_data['community_sentiment'] == 'negative'):
            context['situation'] = 'crisis'
            context['urgency'] = 'high'
        
        # Detecta estagnação
        if len(history) >= 3:
            recent_dau = [h['daily_active_users'] for h in list(history)[-3:]]
            recent_price = [h['token_price'] for h in list(history)[-3:]]
            
            if (np.var(recent_dau) < 0.5 and np.var(recent_price) < 0.001):
                context['situation'] = 'stagnation'
                context['stagnation_cycles'] = self.count_stagnation_cycles(history)
                context['urgency'] = 'medium'
        
        # Detecta crescimento
        if len(history) >= 2:
            if (current_data['daily_active_users'] > history[-1]['daily_active_users'] and
                current_data['token_price'] > history[-1]['token_price']):
                if context['situation'] == 'normal':
                    context['situation'] = 'growth'
        
        return context
    
    def count_stagnation_cycles(self, history):
        """Conta ciclos consecutivos de estagnação"""
        count = 0
        for i in range(len(history) - 1, 0, -1):
            curr = history[i]
            prev = history[i-1]
            
            dau_change = abs(curr['daily_active_users'] - prev['daily_active_users'])
            price_change = abs(curr['token_price'] - prev['token_price'])
            
            if dau_change < 1 and price_change < 0.005:
                count += 1
            else:
                break
        
        return count
    
    def generate_adaptive_prompt(self, current_data, history, withdrawal_rate):
        """Gera prompt adaptativo baseado no contexto"""
        context = self.analyze_economic_context(current_data, history)
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Seleciona template baseado na situação
        situation_prompt = self.prompt_templates.get(context['situation'], "")
        
        # Formata com dados específicos
        if context['situation'] == 'crisis':
            situation_prompt = situation_prompt.format(**current_data)
        elif context['situation'] == 'stagnation':
            situation_prompt = situation_prompt.format(
                stagnation_cycles=context['stagnation_cycles'],
                recent_performance=performance_summary
            )
        
        # Prompt principal
        base_prompt = f"""
{situation_prompt}

CONTEXTO DE PERFORMANCE:
{json.dumps(performance_summary, indent=2)}

SITUAÇÃO ATUAL: {context['situation'].upper()}
URGÊNCIA: {context['urgency'].upper()}

HISTÓRICO ECONÔMICO (últimos 5 ciclos):
{json.dumps(list(history)[-5:], indent=2)}

ESTADO ATUAL (com taxa de saque aplicada):
{json.dumps({**current_data, 'current_withdrawal_rate': withdrawal_rate}, indent=2)}

INSTRUÇÕES ESPECÍFICAS:
1. Se situação = CRISIS: Faça mudanças de 15-30% nas métricas
2. Se situação = STAGNATION: Force variação mínima de 10%
3. Se situação = GROWTH: Mantenha momentum com ajustes de 5-15%
4. SEMPRE varie pelo menos 3 métricas significativamente

REGRAS CRÍTICAS:
- NUNCA repita valores exatos do estado anterior
- Community sentiment DEVE refletir as mudanças econômicas
- Daily_active_users deve ser > 0.1
- Token_price deve ser > 0.01
- New_players nunca deve ser exatamente 1e-05

Responda APENAS com o JSON no formato especificado.
"""
        
        return base_prompt.strip()

class MultiModelSimulator:
    """Simula múltiplos modelos e seleciona o melhor"""
    
    def __init__(self):
        self.api_key = "AIzaSyAQBnCIZD65b-SuLJOJ2iI28iuiI75KMm8"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        self.performance_tracker = PerformanceTracker()
        self.prompt_generator = AdvancedPromptGenerator(self.performance_tracker)
        
        # Sistema de memória expandido
        self.history_file = 'advanced_gemini_history.json'
        self.history = self._load_history()
        self.economy = EconomyState()
        
        # Configurações de simulação múltipla
        self.simulation_configs = [
            {'temperature': 0.7, 'name': 'conservative'},
            {'temperature': 0.9, 'name': 'aggressive'},
            {'temperature': 0.8, 'name': 'balanced'}
        ]
    
    def _load_history(self):
        """Carrega histórico expandido"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return deque(data.get('history', []), maxlen=50)
            except Exception as e:
                print(f"Erro ao carregar histórico: {e}")
        return deque(maxlen=50)
    
    def _save_history(self):
        """Salva histórico expandido"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'history': list(self.history),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar histórico: {e}")
    
    def _call_gemini_with_config(self, prompt, config):
        """Chama Gemini com configuração específica"""
        payload = {
            "contents": [
                { "role": "user", "parts": [{"text": prompt}] }
            ],
            "generationConfig": {
                "temperature": config.get('temperature', 0.8),
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "transaction_volume": { "type": "NUMBER" },
                        "nft_avg_price": { "type": "NUMBER" },
                        "daily_active_users": { "type": "NUMBER" },
                        "new_players": { "type": "NUMBER" },
                        "leaving_players": { "type": "NUMBER" },
                        "dex_liquidity": { "type": "NUMBER" },
                        "token_price": { "type": "NUMBER" },
                        "token_generation_rate": { "type": "NUMBER" },
                        "token_burn_rate": { "type": "NUMBER" },
                        "circulating_assets": { "type": "NUMBER" },
                        "community_sentiment": { "type": "STRING", "enum": ["positive", "neutral", "negative"] }
                    },
                    "required": ["transaction_volume", "nft_avg_price", "daily_active_users", "new_players", "leaving_players", "dex_liquidity", "token_price", "token_generation_rate", "token_burn_rate", "circulating_assets", "community_sentiment"]
                }
            }
        }
        
        try:
            response = requests.post(self.api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if result.get('candidates') and result['candidates'][0].get('content'):
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                return json.loads(json_string)
            return None
        except Exception as e:
            print(f"Erro na API Gemini ({config['name']}): {e}")
            return None
    
    def generate_multiple_scenarios(self, prompt):
        """Gera múltiplos cenários e seleciona o melhor"""
        scenarios = []
        
        print("[MULTI-MODEL] Gerando múltiplos cenários...")
        
        for config in self.simulation_configs:
            print(f"[MULTI-MODEL] Testando configuração: {config['name']}")
            result = self._call_gemini_with_config(prompt, config)
            
            if result:
                # Calcula score de diversidade
                diversity_score = self.calculate_diversity_score(result)
                scenarios.append({
                    'config': config['name'],
                    'data': result,
                    'diversity_score': diversity_score
                })
                print(f"[MULTI-MODEL] {config['name']}: Score diversidade = {diversity_score:.2f}")
            else:
                print(f"[MULTI-MODEL] {config['name']}: Falhou")
        
        if not scenarios:
            return None
        
        # Seleciona cenário com melhor score de diversidade
        best_scenario = max(scenarios, key=lambda x: x['diversity_score'])
        print(f"[MULTI-MODEL] Melhor cenário: {best_scenario['config']} (Score: {best_scenario['diversity_score']:.2f})")
        
        return best_scenario['data']
    
    def calculate_diversity_score(self, result):
        """Calcula score de diversidade comparado com estados recentes"""
        if len(self.history) == 0:
            return 10  # Máximo se não há histórico
        
        recent_state = self.history[-1]
        score = 0
        
        # Compara mudanças percentuais em métricas chave
        key_metrics = ['daily_active_users', 'token_price', 'transaction_volume', 'dex_liquidity']
        
        for metric in key_metrics:
            if recent_state[metric] != 0:
                change_pct = abs(result[metric] - recent_state[metric]) / recent_state[metric]
                score += change_pct * 100  # Converte para pontos
        
        # Bonus por mudança de sentimento
        if result['community_sentiment'] != recent_state.get('community_sentiment', 'neutral'):
            score += 5
        
        return min(score, 100)  # Cap em 100
    
    def run_advanced_simulation_cycle(self):
        """Executa ciclo avançado de simulação"""
        print("\n" + "=" * 80)
        print("[ADVANCED-AI] Iniciando ciclo avançado de simulação...")
        
        # Carrega estado atual
        withdrawal_rate = self.ler_dados('current_withdrawal_rate.txt', 0.20)
        current_data = self.economy.get_data()
        
        print(f"[ADVANCED-AI] Taxa de Saque: {withdrawal_rate:.2%}")
        print(f"[ADVANCED-AI] Estado Atual: DAU={current_data['daily_active_users']:.1f}, Preço=${current_data['token_price']:.3f}")
        
        # Salva estado atual no histórico
        previous_state = copy.deepcopy(current_data)
        self.history.append(previous_state)
        
        # Gera prompt adaptativo
        prompt = self.prompt_generator.generate_adaptive_prompt(current_data, self.history, withdrawal_rate)
        
        # Gera múltiplos cenários
        best_result = self.generate_multiple_scenarios(prompt)
        
        if best_result:
            # Avalia a decisão
            reward, metric = self.performance_tracker.evaluate_decision(
                previous_state, best_result, withdrawal_rate
            )
            
            # Atualiza economia
            self.economy.update_data(best_result)
            
            print(f"[ADVANCED-AI] Simulação concluída!")
            print(f"[ADVANCED-AI] Novo Estado: DAU={best_result['daily_active_users']:.1f}, Preço=${best_result['token_price']:.3f}")
            print(f"[ADVANCED-AI] Sentimento: {best_result['community_sentiment']}")
            print(f"[ADVANCED-AI] Reward da Decisão: {reward:.2f}")
            print(f"[ADVANCED-AI] Score Saúde Econômica: {metric['curr_score']:.1f}/100")
            
        else:
            print("[ADVANCED-AI] FALHA: Nenhum cenário foi gerado com sucesso!")
        
        # Salva histórico
        self._save_history()
        
        # Mostra resumo de performance
        perf_summary = self.performance_tracker.get_performance_summary()
        print(f"[ADVANCED-AI] Performance Média (últimas 10): {perf_summary['avg_recent_reward']:.2f}")
        print(f"[ADVANCED-AI] Tendência: {perf_summary['trend']}")
        
        print("=" * 80)
    
    def ler_dados(self, filepath, default_value):
        """Método auxiliar para ler dados"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return float(f.read().strip())
        except:
            return default_value

# Classe EconomyState original (mantida para compatibilidade)
class EconomyState:
    """Gerencia o estado da economia, lendo e escrevendo para arquivos."""
    def __init__(self):
        self.defaults = {
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
        self.data = {}
        self._load_initial_state()

    def _load_initial_state(self):
        """Carrega o estado atual da economia dos arquivos .txt."""
        for key, default in self.defaults.items():
            self.data[key] = self.ler_dados(f"{key}.txt", default, is_float=(key != 'community_sentiment'))

    def get_data(self):
        """Retorna os dados atuais da economia."""
        return self.data

    def update_data(self, new_data_dict):
        """Atualiza e salva os dados da economia."""
        for key, value in new_data_dict.items():
            if key in self.data:
                self.data[key] = value
        self._save_current_state()

    def _save_current_state(self):
        """Salva o estado atual da economia de volta nos arquivos .txt."""
        for key, value in self.data.items():
            self.escrever_dados(f"{key}.txt", value)
    
    def ler_dados(self, filepath, default_value, is_float=True):
        """Lê um valor de um arquivo, com fallback para um valor padrão."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if is_float:
                    return float(content)
                else:
                    return content
        except (FileNotFoundError, ValueError):
            return default_value

    def escrever_dados(self, filepath, value):
        """Escreve um valor em um arquivo."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(value))
        except IOError as e:
            print(f"Erro ao escrever no arquivo {filepath}: {e}")

def main_advanced_run():
    """Função principal para executar o sistema avançado"""
    simulator = MultiModelSimulator()
    simulator.run_advanced_simulation_cycle()

def run_multiple_cycles(num_cycles=10):
    """Executa múltiplos ciclos para teste"""
    simulator = MultiModelSimulator()
    
    print(f"[BATCH] Executando {num_cycles} ciclos de simulação...")
    
    for i in range(num_cycles):
        print(f"\n[BATCH] === CICLO {i+1}/{num_cycles} ===")
        simulator.run_advanced_simulation_cycle()
        
        # Pequena pausa entre ciclos
        import time
        time.sleep(1)
    
    # Relatório final
    perf_summary = simulator.performance_tracker.get_performance_summary()
    print(f"\n[BATCH] === RELATÓRIO FINAL ===")
    print(f"[BATCH] Total de decisões: {perf_summary['total_decisions']}")
    print(f"[BATCH] Performance média: {perf_summary['avg_recent_reward']:.2f}")
    print(f"[BATCH] Tendência geral: {perf_summary['trend']}")

if __name__ == "__main__":
    # Executa um ciclo único
    main_advanced_run()
    
    # Para testar múltiplos ciclos, descomente a linha abaixo:
    # run_multiple_cycles(5)
