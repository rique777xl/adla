# blockchain_publisher.py
import os
import json
from web3 import Web3
from web3.middleware.geth_poa import geth_poa_middleware
import hashlib

# --- Configurações da Blockchain ---
# Substitua pelo endereço do SEU contrato TextRegistry na Polygon
CONTRACT_ADDRESS = '0xa6630379eBd4daFb35939a93b4e02824816a4f1F' 

# Substitua pelo ABI do SEU contrato TextRegistry
# Este é o JSON que você copiou do Remix
CONTRACT_ABI = json.loads('''
[
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "internalType": "string",
                "name": "_textOrHash",
                "type": "string"
            },
            {
                "indexed": true,
                "internalType": "address",
                "name": "_registrant",
                "type": "address"
            },
            {
                "indexed": false,
                "internalType": "uint256",
                "name": "_timestamp",
                "type": "uint256"
            }
        ],
        "name": "TextRegistered",
        "type": "event"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "_textOrHash",
                "type": "string"
            }
        ],
        "name": "getRegistrant",
        "outputs": [
            {
                "internalType": "address",
                "name": "",
                "type": "address"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "_textOrHash",
                "type": "string"
            }
        ],
        "name": "getRegistrationTimestamp",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "_textOrHash",
                "type": "string"
            }
        ],
        "name": "registerText",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
''') # Cole o ABI completo aqui dentro das aspas triplas

# URL do node Polygon (substitua por um provedor real, como Infura, Alchemy, ou seu próprio nó local)
# Use a URL da Mainnet ou Testnet (Amoy/Mumbai) dependendo de onde seu contrato está.
# Exemplo para Polygon Mainnet (via Infura, substitua YOUR_INFURA_PROJECT_ID):
# POLYGON_RPC_URL = "https://polygon-mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
# Exemplo para Polygon Amoy Testnet (via Infura, substitua YOUR_INFURA_PROJECT_ID):
# POLYGON_RPC_URL = "https://polygon-amoy.infura.io/v3/YOUR_INFURA_PROJECT_ID"
# Exemplo para um nó local de testes (Ganache/Hardhat local)
POLYGON_RPC_URL = "https://polygon-rpc.com" # Exemplo para um ambiente de desenvolvimento local

# Sua chave privada (NUNCA DEIXE EM CÓDIGO FONTE EM PRODUÇÃO! USE VARIÁVEIS DE AMBIENTE!)
# Esta é a chave privada da conta que irá pagar o gás das transações.
# Exemplo: private_key = os.environ.get("PRIVATE_KEY")
PRIVATE_KEY = "64cc542b7e96058d67a5b294b6eda0fd3acdcc8fe1a9de8657c525f63222466f" # !!! SUBSTITUA PELA SUA CHAVE PRIVADA (COM CUIDADO DE SEGURANÇA) !!!

# Endereço da sua carteira (gerado a partir da chave privada)
ACCOUNT_ADDRESS = Web3.to_checksum_address("0xBbBdE2FB9F1e23B04AE4dDbF197B75c84865Fde6") # !!! SUBSTITUA PELO SEU ENDEREÇO DE CARTEIRA !!!


def get_web3_instance():
    """Retorna uma instância configurada do Web3."""
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC_URL))
    
    # Adiciona o middleware PoA para redes como Polygon
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    if not w3.is_connected():
        raise Exception(f"Não foi possível conectar à rede Polygon em {POLYGON_RPC_URL}")
    print(f"Conectado à rede Polygon em {POLYGON_RPC_URL}")
    return w3

def calculate_sha256_hash(data_string):
    """Calcula o hash SHA256 de uma string."""
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

def register_data_on_blockchain(data_to_register_hash):
    """
    Registra um hash (ou texto) na blockchain usando o contrato TextRegistry.
    Args:
        data_to_register_hash (str): O hash SHA256 dos dados da IA (ou um texto curto).
    Returns:
        str: O hash da transação, ou None em caso de falha.
    """
    w3 = get_web3_instance()
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

    try:
        # 1. Obter o nonce da transação (número de transações enviadas pela sua conta)
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)

        # 2. Construir a transação
        # gas_price = w3.eth.gas_price # Você pode deixar o node sugerir ou definir manualmente
        # É uma boa prática definir um gas_price e gas_limit razoáveis para a Polygon
        # Você pode consultar https://gasnow.org/ ou Polygonscan para valores atuais
        
        # Para Polygon, usar 'maxFeePerGas' e 'maxPriorityFeePerGas' é mais comum (EIP-1559)
        # Se o seu nó RPC não suportar EIP-1559, você precisará usar 'gasPrice'
        
        # Exemplo com EIP-1559 (preferencial na Polygon):
        # max_priority_fee_per_gas = w3.to_wei('30', 'gwei') # Exemplo de 30 Gwei
        # latest_block = w3.eth.get_block('latest')
        # base_fee_per_gas = latest_block['baseFeePerGas']
        # max_fee_per_gas = base_fee_per_gas + max_priority_fee_per_gas # ou um valor fixo maior

        transaction = contract.functions.registerText(data_to_register_hash).build_transaction({
            'chainId': w3.eth.chain_id,
            'from': ACCOUNT_ADDRESS,
            'nonce': nonce,
            # 'gasPrice': w3.to_wei('30', 'gwei'), # Exemplo: 30 Gwei, ajuste conforme a rede
            'gas': 200000 # Limite de gás, ajuste se necessário, mas 200k é geralmente suficiente
            # 'maxFeePerGas': max_fee_per_gas, # Descomente para EIP-1559
            # 'maxPriorityFeePerGas': max_priority_fee_per_gas, # Descomente para EIP-1559
        })

        # 3. Assinar a transação
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=PRIVATE_KEY)

        # 4. Enviar a transação
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        print(f"Transação enviada! Hash: {tx_hash.hex()}")

        # 5. Esperar pela confirmação da transação
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status == 1:
            print(f"Dados registrados com sucesso! Transação confirmada em bloco {receipt.blockNumber}")
            return tx_hash.hex()
        else:
            print(f"Falha na transação. Status do recibo: {receipt.status}")
            return None

    except Exception as e:
        print(f"Erro ao registrar dados na blockchain: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso:
    # 1. Digamos que sua IA gerou um hash de seus dados de entrada/saída
    # (No seu cenário, você leria isso do 'state_snapshots.json' ou similar)
    
    # Exemplo de como você obteria o hash dos dados mais recentes da IA:
    latest_snapshot_hash = None
    try:
        # Lendo o último snapshot do arquivo
        if os.path.exists('state_snapshots.json'):
            with open('state_snapshots.json', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    snapshot_data = json.loads(last_line)
                    # O 'state' já é uma lista de floats, convertemos para string para hash
                    state_string = json.dumps(snapshot_data['state'], sort_keys=True) 
                    latest_snapshot_hash = calculate_sha256_hash(state_string)
                    print(f"Hash do último snapshot de estado da IA: {latest_snapshot_hash}")
                else:
                    print("Nenhum snapshot de estado encontrado.")
        else:
            print("Arquivo 'state_snapshots.json' não encontrado.")
    except Exception as e:
        print(f"Erro ao ler ou processar 'state_snapshots.json': {e}")

    if latest_snapshot_hash:
        # Agora registre esse hash na blockchain
        transaction_hash = register_data_on_blockchain(latest_snapshot_hash)
        if transaction_hash:
            print(f"Hash da IA registrado na blockchain! Você pode vê-lo em Polygonscan: https://polygonscan.com/tx/{transaction_hash}")
        else:
            print("Não foi possível registrar o hash da IA na blockchain.")
    else:
        print("Nenhum hash disponível para registro na blockchain.")