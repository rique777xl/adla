
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
# POLYGON_RPC_URL = "https://polygon-amoy.infura.io/v3/YOUR_INFURA_PROJECT_I
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
            RIVATE_KEY)

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
        

        print("Nenhum hash disponível para registro na blockchain.")
