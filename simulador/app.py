# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # Necessário para permitir requisições do frontend
import json
import os
import hashlib
import subprocess # Para chamar o blockchain_publisher.py

# Importa as classes da sua IA e a função auxiliar
# Certifique-se de que os arquivos .py da IA e do blockchain_publisher estão na mesma pasta
from AutonomousGameEconomyAI import AutonomousGameEconomyAI, create_sample_files_if_not_exist

app = Flask(__name__)
CORS(app) # Habilita CORS para permitir que seu frontend HTML/JS se comunique com o backend

# Instância global da IA (carregada uma vez na inicialização do servidor)
ai_instance = AutonomousGameEconomyAI()

# Caminho para o arquivo de snapshot (certifique-se de que corresponde ao da sua IA)
# O nome do arquivo é 'state_snapshots.json'
SNAPSHOT_FILE = 'state_snapshots.json' 

@app.route('/')
def index():
    """Serve o arquivo HTML principal."""
    return render_template('index.html')

@app.route('/run_ai_simulation', methods=['POST'])
def run_ai_simulation():
    """Endpoint para rodar a simulação da IA."""
    try:
        # Recebe os dados de entrada do frontend
        data_from_frontend = request.json['data']
        # A taxa de saque atual para a IA calcular o próximo estado
        # O frontend envia 'current_withdrawal_rate' como parte de 'data'.
        # Usamos o valor que o frontend está exibindo como a "taxa anterior" para a IA.
        current_withdrawal_rate_for_ai = data_from_frontend.get('current_withdrawal_rate', 0.20)

        # Chama a nova função da IA que recebe os dados diretamente do frontend
        # A IA processa, decide e armazena o histórico internamente.
        result = ai_instance.process_incoming_data(data_from_frontend, current_withdrawal_rate_for_ai)
        
        rate = result['rate']
        analysis = result['analysis']
        current_state_vector = result['state'] # Já é uma lista
        action = result['action']

        # Para o frontend, precisamos do hash do último snapshot para o histórico
        # Lemos o último snapshot salvo para obter o hash.
        # A função process_incoming_data da IA já salva o snapshot internamente.
        latest_snapshot_hash = None
        if os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    snapshot_data = json.loads(last_line)
                    # O 'state' já é uma lista de floats, convertemos para string para hash
                    state_string = json.dumps(snapshot_data['state'], sort_keys=True) 
                    latest_snapshot_hash = hashlib.sha256(state_string.encode('utf-8')).hexdigest()

        # Retorna os resultados para o frontend
        return jsonify({
            'rate': rate,
            'analysis': analysis,
            'state': current_state_vector, # Já é uma lista
            'state_hash': latest_snapshot_hash
        }), 200

    except Exception as e:
        print(f"Erro no endpoint /run_ai_simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/register_on_blockchain', methods=['POST'])
def register_on_blockchain():
    """Endpoint para registrar o último hash da IA na blockchain."""
    try:
        latest_snapshot_hash = None
        if os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    snapshot_data = json.loads(last_line)
                    state_string = json.dumps(snapshot_data['state'], sort_keys=True) 
                    latest_snapshot_hash = hashlib.sha256(state_string.encode('utf-8')).hexdigest()

        if not latest_snapshot_hash:
            return jsonify({'error': 'Nenhum snapshot de estado encontrado para registrar.'}), 400

        # Chama o script blockchain_publisher.py como um processo separado.
        # Ele lerá o último snapshot e enviará para a blockchain.
        process = subprocess.run(
            ["python", "blockchain_publisher.py"],
            capture_output=True,
            text=True,
            check=False # Não levanta exceção para códigos de retorno != 0
        )
        
        print(f"Saída do blockchain_publisher.py:\n{process.stdout}")
        print(f"Erros do blockchain_publisher.py:\n{process.stderr}")

        if process.returncode == 0:
            # Tenta extrair o hash da transação da saída
            tx_hash = None
            for line in process.stdout.split('\n'):
                if "Transação enviada! Hash:" in line:
                    tx_hash = line.split("Hash: ")[1].strip()
                    break
            
            if tx_hash:
                return jsonify({'message': 'Registro na blockchain iniciado com sucesso.', 'tx_hash': tx_hash}), 200
            else:
                return jsonify({'error': 'Registro na blockchain concluído, mas hash da transação não encontrado na saída.', 'details': process.stdout}), 500
        else:
            return jsonify({'error': f'Erro ao executar blockchain_publisher.py. Código de saída: {process.returncode}', 'details': process.stderr}), 500

    except Exception as e:
        print(f"Erro no endpoint /register_on_blockchain: {e}")
        return jsonify({'error': str(e)}), 500

# A função update_txt_files não é mais necessária, pois a IA recebe os dados diretamente
# def update_txt_files(data):
#     """Atualiza os arquivos .txt que a IA lê com os dados do frontend."""
#     for key, value in data.items():
#         filename = f"{key}.txt"
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 f.write(str(value))
#             print(f"Arquivo '{filename}' atualizado com '{value}'")
#         except Exception as e:
#             print(f"Erro ao escrever em '{filename}': {e}")

if __name__ == '__main__':
    # Garante que os arquivos .txt iniciais existam para a IA carregar histórico
    # e para o blockchain_publisher.py ter um snapshot para ler.
    create_sample_files_if_not_exist() 
    
    # Opcional: Você pode querer rodar um treinamento inicial da IA aqui
    # se você não tiver modelos .h5 pré-treinados e quiser que a IA comece com algum "conhecimento".
    # ai_instance.train_on_historical_data() 
    # ai_instance.save_models()

    print("Iniciando servidor Flask...")
    app.run(debug=True, port=5000) # Rode em modo debug para desenvolvimento