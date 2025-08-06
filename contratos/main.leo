// record_economic_state/src/main.leo
// Certifique-se de que este arquivo está dentro de record_economic_state/src/
program record_economic_state.aleo {

    // Define um registro público para armazenar o hash do estado econômico
    record EconomicStateHash {
        owner: address,         // Endereço do proprietário do registro
        state_hash: field,      // O hash do estado econômico (público)
        timestamp: u64,         // Timestamp da gravação (público)
    }

    // Função para armazenar o hash do estado econômico
    // Agora é uma 'transition' para poder manipular registros
    transition store_economic_hash(input_hash: field, timestamp_val: u64) -> EconomicStateHash {
        // Retorna um novo registro EconomicStateHash
        return EconomicStateHash {
            owner: self.caller,     // Usa 'self.caller' para obter o endereço do chamador
            state_hash: input_hash, // O hash fornecido como entrada
            timestamp: timestamp_val, // O timestamp fornecido como entrada
        };
    }
}