import json
import os
import requests # Importar requests para fazer chamadas HTTP

# --- Funções Auxiliares ---

def ler_dados(filepath, default_value, is_float=True):
    """
    Lê um valor de um arquivo, com fallback para um valor padrão.
    Se o arquivo não existir ou o conteúdo for inválido, ele cria o arquivo
    com o valor padrão.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if is_float:
                return float(content)
            else:
                return content
    except (FileNotFoundError, ValueError):
        # Se o arquivo não existir ou for inválido, usa o valor padrão e tenta criar
        print(f"Arquivo '{filepath}' não encontrado ou inválido. Criando com valor padrão: {default_value}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(default_value))
        return default_value

def escrever_dados(filepath, value):
    """Escreve um valor em um arquivo."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(value))
    except IOError as e:
        print(f"Erro ao escrever no arquivo {filepath}: {e}")

# --- Definição do Estado Econômico ---

class EconomyState:
    """
    Gerencia o estado atual da economia do jogo, incluindo leitura e escrita
    de dados para arquivos de texto.
    """
    def __init__(self):
        # Valores padrão para cada parâmetro econômico
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
            'community_sentiment': "neutral" # pode ser "positive", "neutral", "negative"
        }
        self.data = {}
        self._load_initial_state()

    def _load_initial_state(self):
        """Carrega o estado atual da economia dos arquivos .txt."""
        for key, default in self.defaults.items():
            self.data[key] = ler_dados(f"{key}.txt", default, is_float=(key != 'community_sentiment'))
        # Garante que 'current_withdrawal_rate.txt' também exista
        ler_dados('current_withdrawal_rate.txt', 0.20, is_float=True)


    def get_data(self):
        """Retorna os dados atuais da economia."""
        return self.data

    def update_data(self, new_data_dict):
        """
        Atualiza os dados da economia com base em um dicionário fornecido
        e salva o estado atual de volta nos arquivos.
        """
        for key, value in new_data_dict.items():
            # Garante que o sentimento da comunidade seja um dos valores válidos
            if key == 'community_sentiment' and value not in ["positive", "neutral", "negative"]:
                print(f"Aviso: Sentimento '{value}' inválido recebido. Usando 'neutral'.")
                self.data[key] = "neutral"
            else:
                self.data[key] = value
        self._save_current_state()

    def _save_current_state(self):
        """Salva o estado atual da economia de volta nos arquivos .txt."""
        for key, value in self.data.items():
            escrever_dados(f"{key}.txt", value)

# --- Função Principal e de Controle ---

class GameEconomyController:
    """
    Controla o fluxo de simulação da economia do jogo, interagindo com o modelo Gemini.
    """
    def __init__(self):
        self.economy = EconomyState()
        # A API Key será fornecida pelo ambiente de execução.
        # Mantenha-a como uma string vazia aqui.
        self.api_key = "AIzaSyAQBnCIZD65b-SuLJOJ2iI28iuiI75KMm8" 
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"

    def _build_gemini_prompt(self, current_data, withdrawal_rate):
        """
        Constrói o prompt textual para o modelo Gemini, incluindo o estado
        econômico atual, a taxa de saque fornecida e o formato de saída JSON desejado.
        """
        # Adiciona a taxa de saque atual aos dados que serão enviados ao Gemini
        data_for_gemini = current_data.copy()
        data_for_gemini['current_withdrawal_rate'] = withdrawal_rate
        current_data_str = json.dumps(data_for_gemini, indent=2)

        prompt = f"""
Você é o cérebro de uma simulação econômica de um jogo. Dada a economia atual e a taxa de saque fornecida, preveja o estado da economia após um ciclo.

Estado Econômico Atual (incluindo a taxa de saque aplicada):
{current_data_str}

Com base neste estado e na taxa de saque aplicada, simule o próximo ciclo e forneça os novos valores para cada parâmetro.
O resultado deve ser um objeto JSON no seguinte formato:
{{
  "transaction_volume": <float>,
  "nft_avg_price": <float>,
  "daily_active_users": <float>,
  "new_players": <float>,
  "leaving_players": <float>,
  "dex_liquidity": <float>,
  "token_price": <float>,
  "token_generation_rate": <float>,
  "token_burn_rate": <float>,
  "circulating_assets": <float>,
  "community_sentiment": <string, um dos: "positive", "neutral", "negative">
}}
"""
        return prompt.strip() # Remove espaços em branco extras do prompt

    def _call_gemini_api(self, prompt):
        """
        Chama a API do Gemini com o prompt construído e retorna a resposta JSON.
        Inclui a configuração de geração para garantir uma resposta JSON estruturada.
        """
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
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
                    "required": [ # Garante que todos esses campos estejam presentes na resposta
                        "transaction_volume",
                        "nft_avg_price",
                        "daily_active_users",
                        "new_players",
                        "leaving_players",
                        "dex_liquidity",
                        "token_price",
                        "token_generation_rate",
                        "token_burn_rate",
                        "circulating_assets",
                        "community_sentiment"
                    ]
                }
            }
        }
        
        try:
            # Faz a requisição POST para a API do Gemini
            response = requests.post(self.api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)
            result = response.json()

            # Verifica se a resposta contém os dados esperados
            if result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and \
               result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                
                # O conteúdo da resposta é uma string JSON, então a decodificamos
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(json_string)
                return parsed_json
            else:
                print("Erro: Resposta inesperada ou incompleta da API do Gemini.")
                print(f"Resposta completa da API: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Erro na chamada da API do Gemini: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON da resposta do Gemini: {e}")
            print(f"Resposta bruta da API: {response.text if 'response' in locals() else 'N/A'}")
            return None

    def run_simulation_cycle(self):
        """
        Executa um ciclo completo da simulação: lê o estado, consulta o Gemini,
        e atualiza o estado com base na resposta da IA.
        """
        print("\n" + "=" * 70)
        print("[CONTROLLER] Iniciando ciclo de simulação da economia...")

        # 1. Obter a taxa de saque atual (fornecida externamente)
        withdrawal_rate = ler_dados('current_withdrawal_rate.txt', 0.20, is_float=True)
        print(f"[CONTROLLER] Taxa de Saque recebida (externa): {withdrawal_rate:.2%}")

        # 2. Ler o estado atual da economia dos arquivos
        current_data = self.economy.get_data()
        
        # Exibir alguns dados chave antes de enviar para a IA para contexto
        print(f"[CONTROLLER] Estado Atual: DAU={current_data['daily_active_users']:.0f}, Preço Token=${current_data['token_price']:.2f}, Sentimento='{current_data['community_sentiment']}'")

        # 3. Construir o prompt e chamar o modelo Gemini para a simulação, passando a taxa de saque
        prompt = self._build_gemini_prompt(current_data, withdrawal_rate)
        print("[CONTROLLER] Solicitando simulação ao modelo Gemini...")
        gemini_response_data = self._call_gemini_api(prompt)

        if gemini_response_data:
            # 4. Atualizar o estado da economia com os valores simulados pelo Gemini
            # A taxa de saque não é atualizada aqui, pois vem de outra IA
            self.economy.update_data(gemini_response_data)
            
            print(f"[CONTROLLER] Economia simulada pela IA. Novo Estado: DAU={self.economy.data['daily_active_users']:.0f}, Preço Token=${self.economy.data['token_price']:.2f}, Sentimento='{self.economy.data['community_sentiment']}'")
        else:
            print("[CONTROLLER] Falha na simulação do Gemini. O estado da economia não foi alterado.")
            # Em caso de falha da IA, o estado da economia permanece como estava antes do ciclo.

        print("=" * 70)

def main_single_run():
    """
    Função principal para executar um único ciclo de simulação do ambiente.
    """
    controller = GameEconomyController()
    
    # Executa um único ciclo de simulação da economia
    controller.run_simulation_cycle()

if __name__ == "__main__":
    main_single_run()
