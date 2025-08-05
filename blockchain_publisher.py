import os
import json
import subprocess
import datetime
import hashlib
import sys

# --- Configurações da Blockchain Aleo ---
# Substitua pelo nome do seu programa implantado
PROGRAM_ID = 'record_economic_state.aleo'

# --- Configurações do Aleo CLI ---
# O caminho para o seu projeto Aleo
ALEO_PROJECT_PATH = 'C:\\Users\\agis\\Documents\\aleo_programs\\record_economic_state'

# A rede que você está usando (mainnet ou testnet)
NETWORK = 'mainnet'

# Sua chave privada (mantenha-a segura!)
# É altamente recomendável usar variáveis de ambiente para chaves privadas.
PRIVATE_KEY = os.environ.get('ALEO_PRIVATE_KEY', '')

# Verifique se a chave privada foi fornecida
if not PRIVATE_KEY:
    print("AVISO: Chave privada não encontrada. Configure a variável de ambiente ALEO_PRIVATE_KEY")
    print("Exemplo: set ALEO_PRIVATE_KEY=APrivateKey1zkp...")
    sys.exit(1)

# Endpoint da rede
ENDPOINT = 'https://api.explorer.provable.com/v1'

# --- Funções de Ajuda ---

def calculate_sha256_hash_as_aleo_field(data_string):
    """
    Calcula o hash SHA256 de uma string e o formata como um 'field' do Aleo.
    """
    sha256_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    # Garantir que o valor esteja dentro do limite de field do Aleo
    field_value = int(sha256_hash, 16) % (2**251 - 1)
    return str(field_value) + 'field'

def get_current_timestamp_as_aleo_u64():
    """
    Retorna o timestamp Unix atual e o formata como um 'u64' do Aleo.
    """
    timestamp = int(datetime.datetime.now().timestamp())
    # Garantir que esteja dentro do limite u64
    if timestamp > 2**64 - 1:
        timestamp = timestamp % (2**64 - 1)
    return str(timestamp) + 'u64'

def call_aleo_transition(function_name, inputs):
    """
    Chama uma função de transição do seu programa Aleo usando o CLI,
    e envia a transação para a rede.
    """
    # Primeiro, vamos tentar com leo run para teste local
    test_command = [
        'leo', 'run',
        function_name,
        *inputs,
        '--path', ALEO_PROJECT_PATH,
    ]
    
    print(f"Testando localmente com: {' '.join(test_command[:3])} ...")
    
    try:
        test_result = subprocess.run(
            test_command, 
            capture_output=True, 
            text=True, 
            cwd=ALEO_PROJECT_PATH,
            encoding='utf-8'
        )
        
        if test_result.returncode != 0:
            print("Erro durante teste local:")
            print(test_result.stderr)
            return None
        else:
            print("Teste local bem-sucedido!")
    except Exception as e:
        print(f"Erro ao executar teste local: {e}")
    
    # Agora vamos executar na rede
    command = [
        'leo', 'execute',
        function_name,
        *inputs,
        '--path', ALEO_PROJECT_PATH,
        '--private-key', PRIVATE_KEY,
        '--endpoint', ENDPOINT,
        '--network', NETWORK,
        '--broadcast',
        '--yes'  # Adiciona flag para não pedir confirmação
    ]

    print(f"\nExecutando na rede {NETWORK}...")

    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=ALEO_PROJECT_PATH,
            encoding='utf-8',
            timeout=120  # Timeout de 2 minutos
        )
        
        print("Saída do comando leo:")
        print("=" * 50)
        print(result.stdout)
        print("=" * 50)
        
        if result.returncode != 0:
            print("Erro detectado:")  
            print(result.stderr)
            return None
        
        # Procurar por indicadores de sucesso
        if any(success_indicator in result.stdout.lower() for success_indicator in 
               ["transaction", "broadcast", "success", "confirmed"]):
            print("\n✓ Transação enviada para a rede com sucesso!")
            
            # Tentar extrair o ID da transação
            for line in result.stdout.split('\n'):
                if 'transaction' in line.lower() and ('id' in line.lower() or 'hash' in line.lower()):
                    print(f"Info da transação: {line.strip()}")
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        print("Erro: Timeout ao executar o comando leo (excedeu 2 minutos)")
        return None
    except subprocess.CalledProcessError as e:
        print("Erro ao executar o comando leo:")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None

# --- Lógica Principal ---

if __name__ == "__main__":
    print("=== Iniciando registro de hash na blockchain Aleo ===\n")
    
    latest_snapshot_hash = None
    snapshot_info = {}
    
    try:
        if os.path.exists('state_snapshots.json'):
            with open('state_snapshots.json', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    snapshot_data = json.loads(last_line)
                    state_string = json.dumps(snapshot_data['state'], sort_keys=True) 
                    latest_snapshot_hash = calculate_sha256_hash_as_aleo_field(state_string)
                    
                    # Guardar informações adicionais para logging
                    snapshot_info = {
                        'timestamp': snapshot_data.get('timestamp', 'Unknown'),
                        'hash': latest_snapshot_hash
                    }
                    
                    print(f"✓ Snapshot encontrado:")
                    print(f"  - Timestamp: {snapshot_info['timestamp']}")
                    print(f"  - Hash (Aleo field): {latest_snapshot_hash}")
                else:
                    print("✗ Nenhum snapshot de estado encontrado no arquivo.")
        else:
            print("✗ Arquivo 'state_snapshots.json' não encontrado.")
    except json.JSONDecodeError as e:
        print(f"✗ Erro ao decodificar JSON: {e}")
    except Exception as e:
        print(f"✗ Erro ao ler ou processar 'state_snapshots.json': {e}")

    if latest_snapshot_hash:
        print("\n=== Preparando transação Aleo ===")
        timestamp_u64 = get_current_timestamp_as_aleo_u64()
        print(f"Timestamp atual (Aleo u64): {timestamp_u64}")
        
        print("\n=== Executando transação ===")
        aleo_output = call_aleo_transition('store_economic_hash', [latest_snapshot_hash, timestamp_u64])

        if aleo_output:
            print("\n✓ Processo de transação Aleo concluído!")
            
            # Salvar log da transação
            try:
                log_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'snapshot_info': snapshot_info,
                    'aleo_timestamp': timestamp_u64,
                    'status': 'success'
                }
                
                with open('blockchain_submissions.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    
                print("✓ Log da transação salvo em 'blockchain_submissions.log'")
            except Exception as e:
                print(f"Aviso: Não foi possível salvar o log: {e}")
        else:
            print("\n✗ Falha ao executar a transação Aleo.")
    else:
        print("\n✗ Nenhum hash disponível para registro na blockchain Aleo.")
    
    print("\n=== Processo finalizado ===")